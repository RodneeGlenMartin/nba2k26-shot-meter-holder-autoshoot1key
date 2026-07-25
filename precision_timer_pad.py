"""PrecisionTimer Pad — precise rhythm-stick timing driven by a controller.

Controller edition of precision_timer.py.  TAP the trigger button once
(default: button 3 = X on an Xbox 360 pad, joy.cpl numbering) — the
script owns the whole motion: the virtual right stick snaps DOWN
instantly, holds for exactly `hold_ms` (+ offset), flicks UP for
`flick_ms`, then returns to neutral.  Every other physical input
(buttons, sticks, triggers) is mirrored to the virtual pad in real
time; the trigger button itself is never forwarded, so the game only
ever sees the timed stick motion.

    physical X360 pad --XInput poll (1 kHz)--> this script
                                                   |  mirror everything
                                                   |  except the trigger
                                                   v
                        game <--ViGEmBus-- virtual X360 pad

IMPORTANT — the game must see ONLY the virtual pad.  Hide the physical
controller from the game with HidHide and add python.exe to HidHide's
application allowlist (so this script keeps access); otherwise the game
sees both pads: the raw trigger press AND the stick motion.

Requires:
    pip install vgamepad     (virtual pad; needs the ViGEmBus driver —
                              https://github.com/nefarius/ViGEmBus/releases)
    pip install keyboard     (optional — only for the F-key hotkeys)

Precision engine (same as the keyboard edition):
  * the press is timestamped at the XInput poll that detects the edge
    (polling adds at most one tick ≈1 ms of latency; trim with offset)
  * stick-down is injected synchronously in the poll thread; a single
    persistent TIME_CRITICAL worker owns the flick-up
  * coarse wait on a high-resolution waitable timer (Win10 1803+,
    falls back to time.sleep), busy-spin for the final `spin_margin_ms`
  * error is measured at the flick-up injection point

Buttons (joy.cpl numbering for an X360 pad):
    1=A  2=B  3=X  4=Y  5=LB  6=RB  7=Back  8=Start  9=LS  10=RS

Controls:
    F5  pause / resume        F6 / F7   hold -/+ step ms (coarse)
    F3 / F4  hold -/+ 1 ms    F2        fatigue mode on/off
    F8  print status          F10       reset statistics
    F9  quit   (Ctrl-C also works)

Fatigue mode (F2) adds `fatigue_offset_ms` to the hold — flip it on when
your player's stamina is low (2K slows the jumper animation as you tire,
so the ideal tempo gets longer), and off again when fresh.
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

try:
    import keyboard          # optional: F-key hotkeys only
except Exception:
    keyboard = None

try:
    import vgamepad as vg    # virtual X360 pad via ViGEmBus
    _vg_error = None
except Exception as exc:     # ImportError, or ViGEm client DLL failure
    vg = None
    _vg_error = exc

__version__ = "1.0"

# --------------------------------------------------------------------------- #
# Paths / crash logging
# --------------------------------------------------------------------------- #
try:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _crash_log = open(os.path.join(_script_dir, "crash_pad.log"), "w")
    faulthandler.enable(file=_crash_log, all_threads=True)
except OSError:
    _script_dir = "."

CONFIG_PATH = os.path.join(_script_dir, "precision_timer_pad.json")

DEFAULT_CONFIG = {
    "trigger_button": 3,     # joy.cpl number or name ("x", "rb", ...)
    "hold_ms": 555,
    "release_offset_ms": 0,
    "tune_step_ms": 5,
    "spin_margin_ms": 3.0,   # busy-wait window for sub-ms flick accuracy
    "flick_ms": 80,          # how long the up-flick is held
    "fatigue_offset_ms": 8,  # extra hold when fatigue mode (F2) is on
    "poll_hz": 1000,         # physical pad polling rate
    "pad_index": -1,         # XInput slot 0-3, -1 = first connected
    "debug": True,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("PrecisionTimerPad")

# Persistent log (dated) — lets tempo_learn.py match recorded shots to the
# exact press time and hold value that produced them.
try:
    _fh = logging.FileHandler(
        os.path.join(_script_dir, "pad_log.txt"), encoding="utf-8")
    _fh.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(_fh)
except OSError:
    pass


# --------------------------------------------------------------------------- #
# Buttons — joy.cpl numbering (as shown in the controller test dialog)
# mapped to XInput wButtons masks
# --------------------------------------------------------------------------- #
_BUTTON_MASKS = {
    1: 0x1000,   # A
    2: 0x2000,   # B
    3: 0x4000,   # X
    4: 0x8000,   # Y
    5: 0x0100,   # LB
    6: 0x0200,   # RB
    7: 0x0020,   # Back
    8: 0x0010,   # Start
    9: 0x0040,   # LS click
    10: 0x0080,  # RS click
}
_BUTTON_NAMES = {1: "A", 2: "B", 3: "X", 4: "Y", 5: "LB", 6: "RB",
                 7: "Back", 8: "Start", 9: "LS", 10: "RS"}
_NAME_TO_NUM = {v.lower(): k for k, v in _BUTTON_NAMES.items()}

STICK_MIN = -32768   # full down
STICK_MAX = 32767    # full up


def _parse_trigger(val) -> int:
    """Accept a joy.cpl button number (1-10) or a name like 'x'."""
    if isinstance(val, str):
        v = val.strip().lower()
        if v in _NAME_TO_NUM:
            return _NAME_TO_NUM[v]
        val = v
    try:
        n = int(val)
    except (TypeError, ValueError):
        return DEFAULT_CONFIG["trigger_button"]
    return n if n in _BUTTON_MASKS else DEFAULT_CONFIG["trigger_button"]


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
    _num("flick_ms", 80, 10, 1000)
    _num("fatigue_offset_ms", 8, 0, 100)
    _num("poll_hz", 1000, 125, 2000)
    _num("pad_index", -1, -1, 3)
    cfg["debug"] = bool(cfg.get("debug", DEFAULT_CONFIG["debug"]))
    cfg["trigger_button"] = _parse_trigger(
        cfg.get("trigger_button", DEFAULT_CONFIG["trigger_button"]))
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
fatigued = False   # F2: player is tired -> add fatigue_offset_ms to hold
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
            self.misses = 0  # flicks landing more than 1 ms off target
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

# A lock protects cycle start/end handoff between the poll loop and the
# flick worker; _pad_lock serializes all virtual-pad report writes/updates.
_state_lock = threading.Lock()
_pad_lock = threading.Lock()
_cycle_active = False      # a timed motion is in flight; the timer owns RS

_vpad = None               # vgamepad.VX360Gamepad, created in main()
worker = None              # FlickWorker, created in main()
loop = None                # PadLoop, created in main()


# --------------------------------------------------------------------------- #
# XInput — reading the physical pad
# --------------------------------------------------------------------------- #
class _XINPUT_GAMEPAD(ctypes.Structure):
    _fields_ = (("wButtons", wt.WORD), ("bLeftTrigger", ctypes.c_ubyte),
                ("bRightTrigger", ctypes.c_ubyte), ("sThumbLX", ctypes.c_short),
                ("sThumbLY", ctypes.c_short), ("sThumbRX", ctypes.c_short),
                ("sThumbRY", ctypes.c_short))


class _XINPUT_STATE(ctypes.Structure):
    _fields_ = (("dwPacketNumber", wt.DWORD), ("Gamepad", _XINPUT_GAMEPAD))


def _load_xinput():
    for dll in ("XInput1_4", "xinput1_3", "XInput9_1_0"):
        try:
            lib = ctypes.WinDLL(dll)
            lib.XInputGetState.restype = wt.DWORD
            lib.XInputGetState.argtypes = (wt.DWORD, ctypes.POINTER(_XINPUT_STATE))
            return lib
        except OSError:
            continue
    return None


_xinput = _load_xinput()

# XInput slots occupied by our own virtual pad — the poll loop must never
# mirror the virtual pad back into itself.  Filled in at startup in main().
_virtual_slots: set = set()


def _connected_slots() -> set:
    st = _XINPUT_STATE()
    return {i for i in range(4)
            if _xinput.XInputGetState(i, ctypes.byref(st)) == 0}


# --------------------------------------------------------------------------- #
# Flick worker — one persistent TIME_CRITICAL thread owns the up-flick.
# (The stick-down is injected directly in the poll thread at edge detect;
# this worker only waits out the tempo and reverses the stick.)
# --------------------------------------------------------------------------- #
class FlickWorker:
    def __init__(self) -> None:
        self._evt = threading.Event()
        self._press = 0.0
        self._hold = 0.0
        self._waiter = PrecisionWaiter()
        self.hr_timer = self._waiter.high_res
        threading.Thread(target=self._run, name="flick", daemon=True).start()

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

                # 1. Precision wait to reach the green release window
                deadline = press + hold / 1000.0
                self._waiter.wait_until(deadline, float(config["spin_margin_ms"]))

                # 2. Flick up: stick reverses in one atomic report
                t_rel = time.perf_counter()
                with _pad_lock:
                    _vpad.report.sThumbRX = 0
                    _vpad.report.sThumbRY = STICK_MAX
                    _vpad.update()

                # 3. Hold the up-flick long enough for the game to read it
                self._waiter.wait_until(
                    t_rel + float(config["flick_ms"]) / 1000.0,
                    float(config["spin_margin_ms"]))

                # 4. Clean up: stick back to neutral (poll loop then resumes
                #    mirroring the physical right stick)
                with _pad_lock:
                    _vpad.report.sThumbRX = 0
                    _vpad.report.sThumbRY = 0
                    _vpad.update()

                err = (t_rel - press) * 1000.0 - hold
                stats.record(err)

                if config["debug"]:
                    log.info(
                        "[rhythm] tempo held %.2f ms (target %.0f, err %+.2f ms)",
                        (t_rel - press) * 1000.0, hold, err)
            except Exception:
                log.exception("[worker] flick failed")
            finally:
                with _state_lock:
                    _cycle_active = False


# --------------------------------------------------------------------------- #
# Pad loop — polls the physical pad, mirrors it to the virtual pad,
# and starts a timed cycle on the trigger button's press edge
# --------------------------------------------------------------------------- #
class PadLoop:
    def __init__(self, flick_worker: FlickWorker) -> None:
        self.worker = flick_worker
        self.pad_index = None
        self._period = 1.0 / float(config["poll_hz"])
        self._waiter = PrecisionWaiter()
        threading.Thread(target=self._run, name="padloop", daemon=True).start()

    def _find_pad(self, state: _XINPUT_STATE):
        cfg_idx = int(config["pad_index"])
        slots = (cfg_idx,) if 0 <= cfg_idx <= 3 else (0, 1, 2, 3)
        for i in slots:
            if i in _virtual_slots:
                continue
            if _xinput.XInputGetState(i, ctypes.byref(state)) == 0:
                return i
        return None

    def _neutralize(self) -> None:
        """Zero the mirrored inputs so nothing sticks after a disconnect.
        The right stick is left alone if a timed cycle still owns it."""
        with _pad_lock:
            r = _vpad.report
            r.wButtons = 0
            r.bLeftTrigger = 0
            r.bRightTrigger = 0
            r.sThumbLX = 0
            r.sThumbLY = 0
            if not _cycle_active:
                r.sThumbRX = 0
                r.sThumbRY = 0
            _vpad.update()

    def _run(self) -> None:
        global _cycle_active
        _boost_current_thread()
        state = _XINPUT_STATE()
        prev_buttons = 0
        last_sent = None
        next_tick = time.perf_counter()

        while not _shutdown.is_set():
            if self.pad_index is None:
                idx = self._find_pad(state)
                if idx is None:
                    _shutdown.wait(0.25)
                    continue
                self.pad_index = idx
                prev_buttons = 0
                last_sent = None
                next_tick = time.perf_counter()
                log.info("[pad] physical controller connected (slot %d)", idx)

            if _xinput.XInputGetState(self.pad_index, ctypes.byref(state)) != 0:
                log.warning("[pad] controller disconnected (slot %d)",
                            self.pad_index)
                self.pad_index = None
                self._neutralize()
                continue
            now = time.perf_counter()  # timestamp first: before any logic

            gp = state.Gamepad
            buttons = gp.wButtons
            trigger_mask = _BUTTON_MASKS[config["trigger_button"]]

            # Press edge on the trigger button -> start a timed cycle
            if (active and (buttons & trigger_mask)
                    and not (prev_buttons & trigger_mask)):
                start = False
                with _state_lock:
                    if not _cycle_active:
                        _cycle_active = True
                        start = True
                if start:
                    # Snap the virtual right stick DOWN right now, in this
                    # thread — no handoff latency on the press side.
                    with _pad_lock:
                        _vpad.report.sThumbRX = 0
                        _vpad.report.sThumbRY = STICK_MIN
                        _vpad.update()
                    hold = config["hold_ms"] + config["release_offset_ms"]
                    if fatigued:
                        hold += config["fatigue_offset_ms"]
                    self.worker.schedule(now, hold)
                    if config["debug"]:
                        log.info("[trigger] press -> gather %.0f ms"
                                 "  (phys LS %+d,%+d  RS %+d,%+d)",
                                 hold, gp.sThumbLX, gp.sThumbLY,
                                 gp.sThumbRX, gp.sThumbRY)
                elif config["debug"]:
                    # tap during an active cycle: swallowed, never forwarded
                    log.info("[trigger] tap ignored — previous shot still "
                             "in flight")

            # Mirror the physical pad. The trigger button is stripped while
            # active (paused: fully transparent). The right stick belongs to
            # the timer while a cycle is in flight.
            forward = buttons if not active else (buttons & ~trigger_mask)
            with _pad_lock:
                r = _vpad.report
                r.wButtons = forward
                r.bLeftTrigger = gp.bLeftTrigger
                r.bRightTrigger = gp.bRightTrigger
                r.sThumbLX = gp.sThumbLX
                r.sThumbLY = gp.sThumbLY
                if not _cycle_active:
                    r.sThumbRX = gp.sThumbRX
                    r.sThumbRY = gp.sThumbRY
                snap = (r.wButtons, r.bLeftTrigger, r.bRightTrigger,
                        r.sThumbLX, r.sThumbLY, r.sThumbRX, r.sThumbRY)
                if snap != last_sent:
                    _vpad.update()
                    last_sent = snap
            prev_buttons = buttons

            next_tick += self._period
            if next_tick < now:                 # fell behind: resync
                next_tick = now + self._period
            self._waiter.wait_until(next_tick, 0.0)


# --------------------------------------------------------------------------- #
# Status display
# --------------------------------------------------------------------------- #
def print_status() -> None:
    s = stats.snapshot()
    n = config["trigger_button"]
    slot = loop.pad_index if loop else None
    print("\n" + "=" * 56)
    print(f"     PrecisionTimer Pad v{__version__}  (controller edition)")
    print("=" * 56)
    print(f"  Status    : {'ACTIVE' if active else 'PAUSED'}")
    print(f"  Trigger   : button {n} ({_BUTTON_NAMES[n]})  (tap once — no holding)")
    print(f"  Hold      : {config['hold_ms']} ms"
          f"  (offset {int(config['release_offset_ms']):+d})"
          f"   flick {int(config['flick_ms'])} ms")
    print(f"  Fatigue   : {'ON +' + str(int(config['fatigue_offset_ms'])) + ' ms'
                           if fatigued else 'off'}  (F2 toggles)")
    print(f"  Engine    : poll {int(config['poll_hz'])} Hz"
          f"   spin {float(config['spin_margin_ms']):.1f} ms"
          f"   timer {'high-res' if worker and worker.hr_timer else 'standard'}")
    print(f"  Pads      : physical "
          f"{'slot ' + str(slot) if slot is not None else 'NOT CONNECTED'}"
          f"   |   virtual OK")
    print(f"  Actions   : {s['count']}   misses(|err|>1ms): {s['misses']}")
    if s["count"]:
        print(f"  Error     : last {s['last']:+.2f}  avg {s['mean']:+.2f}"
              f"  sd {s['std']:.2f}  min {s['min']:+.2f}"
              f"  max {s['max']:+.2f} ms")
    print("-" * 56)
    if keyboard:
        print(f"  F5 pause   F6/F7 hold -/+{int(config['tune_step_ms'])}ms"
              f"   F3/F4 hold -/+1ms   F2 fatigue")
        print("  F8 status   F10 reset stats   F9 quit")
    else:
        print("  hotkeys disabled ('keyboard' package missing) — Ctrl-C quits")
    print("  XInput pads can't be cloaked by HidHide — use slot order:")
    print("  unplug pad -> start this script -> replug pad -> start game")
    print("  (virtual pad takes slot 0, so the game binds to it).")
    print("=" * 56 + "\n")


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


def toggle_fatigue() -> None:
    global fatigued
    fatigued = not fatigued
    if fatigued:
        log.info("[tempo] fatigue mode ON  (hold %d +%d ms)",
                 config["hold_ms"], config["fatigue_offset_ms"])
    else:
        log.info("[tempo] fatigue mode OFF (hold %d ms)", config["hold_ms"])


def reset_stats() -> None:
    stats.reset()
    log.info("[stats] reset")


# --------------------------------------------------------------------------- #
# Config hot-reload — lets live_coach.py (or manual JSON edits) retune the
# running timer. Only timing fields are applied live; device fields
# (trigger_button, poll_hz, pad_index) still need a restart.
# --------------------------------------------------------------------------- #
_HOT_FIELDS = ("hold_ms", "release_offset_ms", "fatigue_offset_ms",
               "tune_step_ms", "spin_margin_ms", "flick_ms", "debug")


def _config_watcher() -> None:
    last_mtime = None
    while not _shutdown.is_set():
        _shutdown.wait(1.0)
        try:
            mtime = os.path.getmtime(CONFIG_PATH)
        except OSError:
            continue
        if mtime == last_mtime:
            continue
        last_mtime = mtime
        fresh = load_config()
        changed = {k: fresh[k] for k in _HOT_FIELDS
                   if fresh[k] != config[k]}
        if changed:
            config.update(changed)
            log.info("[config] reloaded: %s  (hold now %d ms)",
                     ", ".join(f"{k}={v}" for k, v in changed.items()),
                     config["hold_ms"])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
_ERROR_ALREADY_EXISTS = 183


def _single_instance() -> bool:
    """Hold a named mutex for the process lifetime; False if another
    instance already owns it (two copies would double the virtual pads,
    double every hotkey press, and race over the config file)."""
    try:
        k32 = ctypes.windll.kernel32
        k32.CreateMutexW(None, False, "PrecisionTimerPad_single_instance")
        return k32.GetLastError() != _ERROR_ALREADY_EXISTS
    except Exception:
        return True


def main() -> None:
    global _vpad, worker, loop

    if not _single_instance():
        log.error("[fatal] another PrecisionTimer Pad is already running — "
                  "close that window first")
        return
    if _xinput is None:
        log.error("[fatal] no XInput DLL found — cannot read the controller")
        return
    if vg is None:
        log.error("[fatal] the 'vgamepad' package is required:")
        log.error("            pip install vgamepad")
        log.error("        (import error: %s)", _vg_error)
        return
    before = _connected_slots()
    try:
        _vpad = vg.VX360Gamepad()
    except Exception as exc:
        log.error("[fatal] could not create the virtual pad: %s", exc)
        log.error("        Install the ViGEmBus driver:")
        log.error("            https://github.com/nefarius/ViGEmBus/releases")
        return

    # Identify which XInput slot the virtual pad landed in, so the poll
    # loop never picks it up as the "physical" controller.
    deadline = time.perf_counter() + 2.0
    while time.perf_counter() < deadline:
        new = _connected_slots() - before
        if new:
            _virtual_slots.update(new)
            break
        time.sleep(0.05)
    if _virtual_slots:
        log.info("[pad] virtual pad occupies slot %s",
                 ", ".join(map(str, sorted(_virtual_slots))))
    else:
        log.warning("[pad] could not identify the virtual pad's XInput slot")

    worker = FlickWorker()
    loop = PadLoop(worker)
    threading.Thread(target=_config_watcher, name="cfgwatch",
                     daemon=True).start()

    if keyboard:
        controls = {
            "f5": toggle_active,
            "f6": lambda: adjust_hold(-config["tune_step_ms"]),
            "f7": lambda: adjust_hold(+config["tune_step_ms"]),
            "f3": lambda: adjust_hold(-1),
            "f4": lambda: adjust_hold(+1),
            "f2": toggle_fatigue,
            "f8": print_status,
            "f10": reset_stats,
            "f9": _shutdown.set,
        }
        for hk, fn in controls.items():
            try:
                keyboard.add_hotkey(hk, fn)
            except Exception:
                log.warning("[hotkey] could not register '%s'", hk)
    else:
        log.warning("[hotkey] 'keyboard' package missing — "
                    "F-keys disabled, Ctrl-C quits")

    print_status()
    log.info("[start] Pad mode ready. Tap button %d (%s) to act.",
             config["trigger_button"], _BUTTON_NAMES[config["trigger_button"]])

    # Idle until F9 (or Ctrl-C); all work happens in poll/worker threads.
    try:
        _shutdown.wait()
    except KeyboardInterrupt:
        _shutdown.set()

    log.info("[exit] stopping")
    if keyboard:
        try:
            keyboard.unhook_all()
        except Exception:
            pass
    try:
        with _pad_lock:
            _vpad.reset()
            _vpad.update()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("[fatal]")
        traceback.print_exc()
        input("\nFatal error — see above / crash_pad.log.  Press Enter.")
    finally:
        if winmm:
            try:
                winmm.timeEndPeriod(1)
            except Exception:
                pass
