"""PrecisionTimer — precise key-hold timing tool for Windows.

TAP the action key once — no need to keep it held.  The physical press
passes straight through to the game/app, the physical release is
suppressed, and a synthetic release fires exactly `hold_ms` (+ offset)
after the press.  The app always sees a hold of precisely the target
duration, whether you tap for 30 ms or keep the key pressed.

Precision engine (all timing is fixed — nothing self-adjusts):
  * the press is timestamped at hook entry, before any locking
  * a single persistent TIME_CRITICAL worker owns every release —
    nothing is spawned inside the low-level hook callback
  * coarse wait on a high-resolution waitable timer (Win10 1803+,
    falls back to time.sleep), busy-spin for the final `spin_margin_ms`
  * the release is a pre-built SendInput call (no per-call key parsing),
    and error is measured at the injection point, not after it

If the timing needs trimming, tune `hold_ms` with F6/F7 or set
`release_offset_ms` in precision_timer.json (applies on restart).

Controls:
    F5  pause / resume        F6 / F7   hold -/+ step ms
    F8  print status          F10       reset statistics
    F9  quit

Requires the `keyboard` package and (typically) administrator privileges.
"""

from __future__ import annotations

import copy
import ctypes
import ctypes.wintypes as wt
import faulthandler
import json
import logging
import math
import os
import threading
import time
import traceback

import keyboard

__version__ = "3.1"

# --------------------------------------------------------------------------- #
# Paths / crash logging
# --------------------------------------------------------------------------- #
try:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _crash_log = open(os.path.join(_script_dir, "crash.log"), "w")
    faulthandler.enable(file=_crash_log, all_threads=True)
except OSError:
    _script_dir = "."

CONFIG_PATH = os.path.join(_script_dir, "precision_timer.json")

DEFAULT_CONFIG = {
    "action_key": "k",
    "hold_ms": 555,
    "release_offset_ms": 0,
    "tune_step_ms": 5,
    "spin_margin_ms": 3.0,   # busy-wait window for sub-ms release accuracy
    "debug": True,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("PrecisionTimer")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def _clamp_config(cfg: dict) -> dict:
    """Coerce/clamp all fields so malformed JSON can't crash timing."""
    def _num(key, default, lo, hi):
        try:
            val = float(cfg.get(key, default))
        except (TypeError, ValueError):
            val = float(default)
        val = max(lo, min(hi, val))
        cfg[key] = int(val) if val.is_integer() else val

    _num("hold_ms", 555, 0, 5000)
    _num("release_offset_ms", 0, -2000, 2000)
    _num("tune_step_ms", 5, 1, 1000)
    _num("spin_margin_ms", 3.0, 0.0, 50.0)
    cfg["debug"] = bool(cfg.get("debug", DEFAULT_CONFIG["debug"]))
    key = cfg.get("action_key", DEFAULT_CONFIG["action_key"])
    cfg["action_key"] = (
        key.strip().lower()
        if isinstance(key, str) and key.strip()
        else DEFAULT_CONFIG["action_key"]
    )
    # Keep only known fields (drops legacy preset/movement entries).
    return {k: cfg.get(k, DEFAULT_CONFIG[k]) for k in DEFAULT_CONFIG}


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        return copy.deepcopy(DEFAULT_CONFIG)
    try:
        with open(CONFIG_PATH, "r") as f:
            loaded = json.load(f)
            if not isinstance(loaded, dict):
                raise ValueError("config is not an object")
            return _clamp_config(loaded)
    except (OSError, ValueError):
        log.warning("[config] unreadable, using defaults")
        return copy.deepcopy(DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    try:
        tmp = CONFIG_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cfg, f, indent=4)
        os.replace(tmp, CONFIG_PATH)
    except OSError as exc:
        log.warning("[config] save failed: %s", exc)


config = load_config()
active = True
_shutdown = threading.Event()


# --------------------------------------------------------------------------- #
# Windows timer resolution + priority
# --------------------------------------------------------------------------- #
winmm = None
try:
    winmm = ctypes.WinDLL("winmm")
    winmm.timeBeginPeriod(1)
except Exception:
    pass

HIGH_PRIORITY_CLASS = 0x00000080
try:
    _k32 = ctypes.windll.kernel32
    _k32.SetPriorityClass(_k32.GetCurrentProcess(), HIGH_PRIORITY_CLASS)
except Exception:
    pass

THREAD_PRIORITY_TIME_CRITICAL = 15


def _boost_current_thread() -> None:
    try:
        h = ctypes.windll.kernel32.GetCurrentThread()
        ctypes.windll.kernel32.SetThreadPriority(h, THREAD_PRIORITY_TIME_CRITICAL)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# High-resolution waiting
# --------------------------------------------------------------------------- #
_CREATE_WAITABLE_TIMER_HIGH_RESOLUTION = 0x00000002
_TIMER_ALL_ACCESS = 0x1F0003


class PrecisionWaiter:
    """Coarse wait on a high-resolution waitable timer (fallback: time.sleep),
    then busy-spin to the exact perf_counter deadline."""

    def __init__(self) -> None:
        self._timer = None
        self._k32 = None
        try:
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.CreateWaitableTimerExW.restype = wt.HANDLE
            k32.CreateWaitableTimerExW.argtypes = (
                ctypes.c_void_p, wt.LPCWSTR, wt.DWORD, wt.DWORD)
            k32.SetWaitableTimer.restype = wt.BOOL
            k32.SetWaitableTimer.argtypes = (
                wt.HANDLE, ctypes.POINTER(wt.LARGE_INTEGER), wt.LONG,
                ctypes.c_void_p, ctypes.c_void_p, wt.BOOL)
            k32.WaitForSingleObject.restype = wt.DWORD
            k32.WaitForSingleObject.argtypes = (wt.HANDLE, wt.DWORD)
            handle = k32.CreateWaitableTimerExW(
                None, None, _CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                _TIMER_ALL_ACCESS)
            if handle:
                self._timer, self._k32 = handle, k32
        except Exception:
            pass

    @property
    def high_res(self) -> bool:
        return self._timer is not None

    def _coarse(self, sec: float) -> None:
        if self._timer:
            due = wt.LARGE_INTEGER(-int(sec * 10_000_000))  # relative, 100 ns
            if self._k32.SetWaitableTimer(
                    self._timer, ctypes.byref(due), 0, None, None, False):
                self._k32.WaitForSingleObject(
                    self._timer, wt.DWORD(int(sec * 1000) + 100))
                return
        time.sleep(sec)

    def wait_until(self, deadline: float, margin_ms: float) -> None:
        """Block until perf_counter() reaches `deadline`."""
        coarse = deadline - margin_ms / 1000.0 - time.perf_counter()
        if coarse > 0:
            self._coarse(coarse)
        while time.perf_counter() < deadline:
            pass


# --------------------------------------------------------------------------- #
# Timing statistics (Welford online mean/variance)
# --------------------------------------------------------------------------- #
class Stats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.count = 0
            self._mean = 0.0
            self._m2 = 0.0
            self.min_err = math.inf
            self.max_err = -math.inf
            self.last_err = 0.0
            self.misses = 0  # releases landing more than 1 ms off target

    def record(self, err_ms: float) -> None:
        with self._lock:
            self.count += 1
            delta = err_ms - self._mean
            self._mean += delta / self.count
            self._m2 += delta * (err_ms - self._mean)
            self.min_err = min(self.min_err, err_ms)
            self.max_err = max(self.max_err, err_ms)
            self.last_err = err_ms
            if abs(err_ms) > 1.0:
                self.misses += 1

    def snapshot(self) -> dict:
        with self._lock:
            std = math.sqrt(self._m2 / self.count) if self.count > 1 else 0.0
            return {
                "count": self.count,
                "mean": self._mean if self.count else 0.0,
                "std": std,
                "min": self.min_err if self.count else 0.0,
                "max": self.max_err if self.count else 0.0,
                "last": self.last_err,
                "misses": self.misses,
            }


stats = Stats()

# Injection/timing state.  A lock protects cycle start/end handoff between
# the hook callback and the release worker.
_state_lock = threading.Lock()
_injecting = False
_physically_held = False   # physical key state (filters auto-repeat)
_passthrough_held = False  # press was passed through untimed (paused)
_cycle_active = False      # a timed hold is in flight; the timer owns the key


# --------------------------------------------------------------------------- #
# Synthetic release via pre-built SendInput (no per-call key parsing)
# --------------------------------------------------------------------------- #
_ULONG_PTR = wt.WPARAM

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
INPUT_KEYBOARD = 1


class MOUSEINPUT(ctypes.Structure):
    _fields_ = (("dx", wt.LONG), ("dy", wt.LONG), ("mouseData", wt.DWORD),
                ("dwFlags", wt.DWORD), ("time", wt.DWORD),
                ("dwExtraInfo", _ULONG_PTR))


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD),
                ("time", wt.DWORD), ("dwExtraInfo", _ULONG_PTR))


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (("uMsg", wt.DWORD), ("wParamL", wt.WORD), ("wParamH", wt.WORD))


class _INPUTUNION(ctypes.Union):
    _fields_ = (("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT))


class INPUT(ctypes.Structure):
    _fields_ = (("type", wt.DWORD), ("union", _INPUTUNION))


_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.SendInput.restype = wt.UINT
_user32.SendInput.argtypes = (wt.UINT, ctypes.POINTER(INPUT), ctypes.c_int)

_release_input: INPUT | None = None

# Keys needing the 0xE0 extended prefix even though the keyboard module
# reports a bare scan code for them.
_EXTENDED_KEY_NAMES = {
    "up", "down", "left", "right", "insert", "delete", "home", "end",
    "page up", "page down", "print screen", "right ctrl", "right alt",
    "left windows", "right windows", "menu", "apps",
}


def _build_key_input(key: str, is_up: bool) -> INPUT:
    try:
        sc = keyboard.key_to_scan_codes(key)[0]
    except Exception:
        sc = 0
    flags = KEYEVENTF_SCANCODE
    if is_up:
        flags |= KEYEVENTF_KEYUP
    if sc > 0xFF or key.strip().lower() in _EXTENDED_KEY_NAMES:
        flags |= KEYEVENTF_EXTENDEDKEY
        sc &= 0xFF
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki = KEYBDINPUT(0, sc, flags, 0, 0)
    return inp

# Build Pro Stick Down (.) and Pro Stick Up (;) inputs based on your keybinds
_down_press = _build_key_input(".", is_up=False)
_down_release = _build_key_input(".", is_up=True)
_up_press = _build_key_input(";", is_up=False)
_up_release = _build_key_input(";", is_up=True)


def _inject_release() -> None:
    global _injecting
    _injecting = True
    try:
        inp = _release_input
        sent = (inp is not None and
                _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT)) == 1)
        if not sent:
            keyboard.release(config["action_key"])
    finally:
        _injecting = False


# --------------------------------------------------------------------------- #
# Release worker — one persistent TIME_CRITICAL thread owns all releases
# --------------------------------------------------------------------------- #
class ReleaseWorker:
    def __init__(self) -> None:
        self._evt = threading.Event()
        self._press = 0.0
        self._hold = 0.0
        self._waiter = PrecisionWaiter()
        self.hr_timer = self._waiter.high_res
        threading.Thread(target=self._run, name="release", daemon=True).start()

    def schedule(self, press_time: float, hold_ms: float) -> None:
        self._press = press_time
        self._hold = float(hold_ms)
        self._evt.set()

    def _run(self) -> None:
        global _cycle_active
        _boost_current_thread()
        while not _shutdown.is_set():
            if not self._evt.wait(0.5):
                continue
            self._evt.clear()
            try:
                press, hold = self._press, self._hold

                # 1. Start sequence: Press Pro Stick Down (.)
                _injecting = True
                _user32.SendInput(1, ctypes.byref(_down_press), ctypes.sizeof(INPUT))
                _injecting = False

                # 2. Precision wait to reach the green release window
                deadline = press + hold / 1000.0
                self._waiter.wait_until(deadline, float(config["spin_margin_ms"]))

                # 3. Flick Up: Release Down (.), instantly press Up (;)
                t_rel = time.perf_counter()
                _injecting = True
                _user32.SendInput(1, ctypes.byref(_down_release), ctypes.sizeof(INPUT))
                _user32.SendInput(1, ctypes.byref(_up_press), ctypes.sizeof(INPUT))
                _injecting = False

                # 4. Hold Pro Stick Up long enough for 2K to read it (80ms)
                self._waiter.wait_until(time.perf_counter() + 0.08, float(config["spin_margin_ms"]))

                # 5. Clean up: Release Up (;)
                _injecting = True
                _user32.SendInput(1, ctypes.byref(_up_release), ctypes.sizeof(INPUT))
                _injecting = False

                err = (t_rel - press) * 1000.0 - hold
                stats.record(err)

                if config["debug"]:
                    log.info(
                        "[rhythm] tempo held %.2f ms (target %.0f, err %+.2f ms)",
                        (t_rel - press) * 1000.0, hold, err)
            except Exception:
                log.exception("[worker] release failed")
            finally:
                with _state_lock:
                    _cycle_active = False


worker = ReleaseWorker()


# --------------------------------------------------------------------------- #
# Status display
# --------------------------------------------------------------------------- #
def print_status() -> None:
    s = stats.snapshot()
    print("\n" + "=" * 56)
    print(f"     PrecisionTimer v{__version__}  (precision timer)")
    print("=" * 56)
    print(f"  Status    : {'ACTIVE' if active else 'PAUSED'}")
    print(f"  Action key: {config['action_key']}  (tap once — no holding)")
    print(f"  Hold      : {config['hold_ms']} ms"
          f"  (offset {int(config['release_offset_ms']):+d})")
    print(f"  Engine    : spin {float(config['spin_margin_ms']):.1f} ms"
          f"   timer {'high-res' if worker.hr_timer else 'standard'}")
    print(f"  Actions   : {s['count']}   misses(|err|>1ms): {s['misses']}")
    if s["count"]:
        print(f"  Error     : last {s['last']:+.2f}  avg {s['mean']:+.2f}"
              f"  sd {s['std']:.2f}  min {s['min']:+.2f}"
              f"  max {s['max']:+.2f} ms")
    print("-" * 56)
    print("  F5 pause   F6 -hold   F7 +hold   F8 status")
    print("  F10 reset stats   F9 quit")
    print("=" * 56 + "\n")


# --------------------------------------------------------------------------- #
# Action-key hook
# --------------------------------------------------------------------------- #
def _hook(event) -> bool:
    now = time.perf_counter()  # timestamp first: before locks, before logic
    try:
        return _hook_impl(event, now)
    except Exception:
        log.exception("[hook] error")
        return True


def _hook_impl(event, now: float) -> bool:
    global _physically_held, _passthrough_held, _cycle_active
    if _injecting:
        return True  # let our own synthetic release pass through

    if event.event_type == keyboard.KEY_DOWN:
        if _physically_held:
            return _passthrough_held  # auto-repeat: transparent only if paused
        _physically_held = True
        if not active:
            _passthrough_held = True
            return True                # paused: key behaves normally
        with _state_lock:
            if not _cycle_active:
                _cycle_active = True
                hold = config["hold_ms"] + config["release_offset_ms"]
                worker.schedule(now, hold)
                return False           # Block K from firing a normal shot
        return False                   # tap during an active cycle: swallow

    # KEY_UP
    was_held = _physically_held
    _physically_held = False
    if _passthrough_held:
        _passthrough_held = False
        return True                    # matching release of a passed press
    if not was_held and not _cycle_active:
        return True                    # unmatched release (press predates hook)
    return False                       # timer owns the release; suppress


# --------------------------------------------------------------------------- #
# Runtime controls
# --------------------------------------------------------------------------- #
def adjust_hold(delta: int) -> None:
    config["hold_ms"] = int(max(0, min(5000, config["hold_ms"] + delta)))
    save_config(config)
    log.info("[tune] hold = %d ms", config["hold_ms"])


def toggle_active() -> None:
    global active
    active = not active
    print_status()


def reset_stats() -> None:
    stats.reset()
    log.info("[stats] reset")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _install_hook(key: str) -> None:
    keyboard.hook_key(key, _hook, suppress=True)


def main() -> None:
    _install_hook(config["action_key"])

    controls = {
        "f5": toggle_active,
        "f6": lambda: adjust_hold(-config["tune_step_ms"]),
        "f7": lambda: adjust_hold(+config["tune_step_ms"]),
        "f8": print_status,
        "f10": reset_stats,
        "f9": _shutdown.set,
    }
    for hk, fn in controls.items():
        try:
            keyboard.add_hotkey(hk, fn)
        except Exception:
            log.warning("[hotkey] could not register '%s'", hk)

    print_status()
    log.info("[start] Timer mode ready. Tap '%s' to act.", config["action_key"])

    # Idle until F9 (or Ctrl-C); all work happens in hook/worker threads.
    try:
        _shutdown.wait()
    except KeyboardInterrupt:
        pass

    log.info("[exit] stopping")
    try:
        keyboard.unhook_all()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("[fatal]")
        traceback.print_exc()
        input("\nFatal error — see above / crash.log.  Press Enter.")
    finally:
        if winmm:
            try:
                winmm.timeEndPeriod(1)
            except Exception:
                pass
