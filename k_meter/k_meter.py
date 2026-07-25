"""K-Meter — keyboard button-shot perfection from the purple meter.

TAP K once. The physical press passes straight through to NBA 2K26 (the
shot starts and the horizontal PURPLE meter appears), the physical
release is suppressed, and a synthetic K-release fires at the perfect
moment — read live off the meter: the magenta fill racing left-to-right
into the GREEN TICK at the bar's tip. Release = fill front reaches the
tick (scheduled predictively from the fill speed, sub-ms injection).

    tap K -> K held for the game -> purple bar fills ->
          vision predicts fill-front @ green tick -> synthetic K-up

Meter signature (measured from real 2K26 screenshots):
    fill  : magenta H 150-151, bar ~128x24 px at 1600x900
    tick  : green H 53-65, fixed at the bar's right tip
    track : dark, to the right of the fill

Fallbacks (in order): meter lost mid-fill -> release immediately;
no meter and nothing magenta on screen within `no_meter_ms` -> hold for
`hold_ms` measured from the press (pressed without a shot); hard safety
at `meter_max_ms` so the key can never stay held forever.

Requires:  pip install keyboard opencv-python mss numpy
Run as administrator (keyboard hook + game window focus).
Do NOT run this at the same time as precision_timer_pad.py — both
register F-key hotkeys.

Controls (all prefixed by `hotkey_prefix`, default Ctrl+Alt — bare
F-keys collide with other tools and silently retune latency mid-game):
    Ctrl+Alt+F5  pause / resume (K behaves normally while paused)
    Ctrl+Alt+F7  latency +10 ms (EARLIER)   Ctrl+Alt+F6  -10 ms (LATER)
    Ctrl+Alt+F4  latency  +3 ms (EARLIER)   Ctrl+Alt+F3   -3 ms (LATER)
    Ctrl+Alt+F8  status   Ctrl+Alt+F10  reset stats   Ctrl+Alt+F9  quit

meter_offset_ms: ms after (positive) or before (negative) the fill
reaches the tick. meter_lead_ms compensates capture/display latency.
Both hot-reload from k_meter.json while running.
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

try:
    import numpy as _np
    import cv2 as _cv2
    from mss import MSS as _MSS
    _VISION_OK = True
except Exception:
    _VISION_OK = False

__version__ = "1.2"      # 1.2: velocity-based latency compensation

# --------------------------------------------------------------------------- #
# Paths / crash logging / config
# --------------------------------------------------------------------------- #
try:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _crash_log = open(os.path.join(_script_dir, "crash.log"), "w")
    faulthandler.enable(file=_crash_log, all_threads=True)
except OSError:
    _script_dir = "."

CONFIG_PATH = os.path.join(_script_dir, "k_meter.json")

DEFAULT_CONFIG = {
    "action_key": "k",       # tap once; also the key held for the game
    "hold_ms": 650,          # fallback hold when the meter is never seen
    "no_meter_ms": 400,      # give up looking for the meter after this
    "spin_margin_ms": 3.0,   # busy-wait window for sub-ms release accuracy
    "latency_ms": 35,        # MAIN KNOB: pipeline lag to cancel
    "release_pct": 97,       # backstop cap if speed can't be measured
    "min_lead_pct": 82,      # minimum percentage of fill before velocity-based release can trigger
    "fast_min_lead_pct": 88, # same floor for fast fills (v >= 0.41)
    "meter_offset_ms": 0,    # extra ms after that point (+ only, delays)
    "meter_lead_ms": 0,      # capture/display latency compensation
    "meter_max_ms": 1200,    # hard safety: never hold K longer than this
    "predict": False,        # False = release ONLY from the fill itself
    "autocal": False,        # broken error signal — see MeterEye [measure]
    "hotkey_prefix": "ctrl+alt+",  # "" = bare F-keys (collides with other tools)
    "game_window": "NBA 2K26",
    "force_mss": True,       # force super stable CPU capture backend
    "persistent_span": 118,  # fallback arrow length if tick span unseen
    "arrow_ratio": 1.227,    # arrow-tip length / fill->tick span
    "lookahead_ms": 25,      # sub-frame release scheduling window
    "v_lsq": True,           # fit v over ALL samples, not just two endpoints
    "v_long_weight": 0.75,   # weight of long-baseline fill speed vs 60ms window
    "stall_cap_ms": 20,      # frame gaps longer than this are capture stalls, not fill time
    "frame_align": True,     # release at the CENTRE of a game frame
    "game_frame_ms": 13.3,   # measured: this rig renders at 75 fps
    "subframe_fill": True,   # extrapolate across the game's 60fps staircase
    "step_cap_ms": 18,       # max extrapolation past the last observed step
    "autotune": True,        # self-correct latency_ms from measured overshoot
    "autotune_shots": 8,     # shots per correction batch (median of these)
    "autotune_deadband_ms": 6,  # ignore errors smaller than this
    "autotune_target_ms": 12, # measured overshoot that means "centred"
    "trust_min_frames": 8,   # fewer tracked frames -> discard the sample
    "trust_min_span": 95,    # smaller meter read -> truncated, discard
    "debug": True,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("KMeter")

try:
    _fh = logging.FileHandler(
        os.path.join(_script_dir, "k_meter_log.txt"), encoding="utf-8")
    _fh.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(_fh)
except OSError:
    pass


def _clamp_config(cfg: dict) -> dict:
    def _num(key, default, lo, hi):
        try:
            val = float(cfg.get(key, default))
        except (TypeError, ValueError):
            val = float(default)
        val = max(lo, min(hi, val))
        cfg[key] = int(val) if val.is_integer() else val

    _num("hold_ms", 650, 100, 3000)
    _num("no_meter_ms", 400, 100, 1500)
    _num("spin_margin_ms", 3.0, 0.0, 50.0)
    _num("latency_ms", 60, 0, 300)
    _num("release_pct", 97, 50, 100)
    _num("min_lead_pct", 82, 50, 100)
    _num("fast_min_lead_pct", 88, 50, 100)
    _num("meter_offset_ms", 0, -300, 800)
    _num("meter_lead_ms", 0, 0, 200)
    _num("meter_max_ms", 1200, 600, 2500)
    _num("persistent_span", 118, 20, 500)
    _num("arrow_ratio", 1.227, 1.0, 2.0)
    _num("lookahead_ms", 25, 0, 60)
    _num("v_long_weight", 0.75, 0.0, 1.0)
    _num("stall_cap_ms", 20, 8, 100)
    _num("step_cap_ms", 18, 0, 40)
    _num("game_frame_ms", 13.3, 5, 40)
    _num("autotune_shots", 8, 3, 50)
    _num("autotune_deadband_ms", 6, 0, 40)
    _num("autotune_target_ms", 12, -40, 40)
    _num("trust_min_frames", 8, 2, 60)
    _num("trust_min_span", 95, 40, 200)
    cfg["autotune"] = bool(cfg.get("autotune", DEFAULT_CONFIG["autotune"]))
    cfg["v_lsq"] = bool(cfg.get("v_lsq", DEFAULT_CONFIG["v_lsq"]))
    cfg["frame_align"] = bool(cfg.get("frame_align", DEFAULT_CONFIG["frame_align"]))
    cfg["subframe_fill"] = bool(cfg.get("subframe_fill", DEFAULT_CONFIG["subframe_fill"]))
    cfg["predict"] = bool(cfg.get("predict", DEFAULT_CONFIG["predict"]))
    cfg["force_mss"] = bool(cfg.get("force_mss", DEFAULT_CONFIG["force_mss"]))
    cfg["debug"] = bool(cfg.get("debug", DEFAULT_CONFIG["debug"]))
    key = cfg.get("action_key", DEFAULT_CONFIG["action_key"])
    cfg["action_key"] = (key.strip().lower()
                         if isinstance(key, str) and key.strip()
                         else DEFAULT_CONFIG["action_key"])
    hp = cfg.get("hotkey_prefix", DEFAULT_CONFIG["hotkey_prefix"])
    cfg["hotkey_prefix"] = hp if isinstance(hp, str) else DEFAULT_CONFIG["hotkey_prefix"]
    gw = cfg.get("game_window", DEFAULT_CONFIG["game_window"])
    cfg["game_window"] = (gw.strip() if isinstance(gw, str) and gw.strip()
                          else DEFAULT_CONFIG["game_window"])
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
        ctypes.windll.kernel32.SetThreadPriority(
            h, THREAD_PRIORITY_TIME_CRITICAL)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# High-resolution waiting
# --------------------------------------------------------------------------- #
_CREATE_WAITABLE_TIMER_HIGH_RESOLUTION = 0x00000002
_TIMER_ALL_ACCESS = 0x1F0003


class PrecisionWaiter:
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
            due = wt.LARGE_INTEGER(-int(sec * 10_000_000))
            if self._k32.SetWaitableTimer(
                    self._timer, ctypes.byref(due), 0, None, None, False):
                self._k32.WaitForSingleObject(
                    self._timer, wt.DWORD(int(sec * 1000) + 100))
                return
        time.sleep(sec)

    def wait_until(self, deadline: float, margin_ms: float) -> None:
        coarse = deadline - margin_ms / 1000.0 - time.perf_counter()
        if coarse > 0:
            self._coarse(coarse)
        while time.perf_counter() < deadline:
            pass


# --------------------------------------------------------------------------- #
# Timing statistics
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
            self.misses = 0

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
            return {"count": self.count,
                    "mean": self._mean if self.count else 0.0,
                    "std": std,
                    "min": self.min_err if self.count else 0.0,
                    "max": self.max_err if self.count else 0.0,
                    "last": self.last_err, "misses": self.misses}


stats = Stats()

_state_lock = threading.Lock()
_injecting = False
_physically_held = False
_passthrough_held = False
_cycle_active = False

# --------------------------------------------------------------------------- #
# Synthetic key release via pre-built SendInput
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
    _fields_ = (("uMsg", wt.DWORD), ("wParamL", wt.WORD),
                ("wParamH", wt.WORD))


class _INPUTUNION(ctypes.Union):
    _fields_ = (("mi", MOUSEINPUT), ("ki", KEYBDINPUT),
                ("hi", HARDWAREINPUT))


class INPUT(ctypes.Structure):
    _fields_ = (("type", wt.DWORD), ("union", _INPUTUNION))


_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.SendInput.restype = wt.UINT
_user32.SendInput.argtypes = (wt.UINT, ctypes.POINTER(INPUT), ctypes.c_int)

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


_release_input = _build_key_input(config["action_key"], is_up=True)


def _inject_release() -> None:
    global _injecting
    _injecting = True
    try:
        sent = _user32.SendInput(
            1, ctypes.byref(_release_input), ctypes.sizeof(INPUT)) == 1
        if not sent:
            keyboard.release(config["action_key"])
    finally:
        _injecting = False


# --------------------------------------------------------------------------- #
# Release worker — persistent TIME_CRITICAL thread, retargetable deadline
# --------------------------------------------------------------------------- #
class ReleaseWorker:
    def __init__(self) -> None:
        self._evt = threading.Event()
        self._retarget = threading.Event()
        self._new_deadline = None
        self._press = 0.0
        self._hold = 0.0
        self._waiter = PrecisionWaiter()
        self.hr_timer = self._waiter.high_res
        self.active_cycle = False
        threading.Thread(target=self._run, name="release",
                         daemon=True).start()

    def schedule(self, press_time: float, hold_ms: float) -> None:
        self._press = press_time
        self._hold = float(hold_ms)
        self.active_cycle = True
        self._evt.set()

    def cancel(self) -> None:
        self.active_cycle = False
        self._retarget.set()

    def retarget(self, deadline: float) -> None:
        self._new_deadline = deadline
        self._retarget.set()

    def _run(self) -> None:
        global _cycle_active
        _boost_current_thread()
        while not _shutdown.is_set():
            if not self._evt.wait(0.5):
                continue
            self._evt.clear()
            self._retarget.clear()
            self._new_deadline = None
            try:
                press = self._press
                deadline = press + self._hold / 1000.0
                spin = float(config["spin_margin_ms"]) / 1000.0
                while not _shutdown.is_set() and self.active_cycle:
                    rem = deadline - spin - time.perf_counter()
                    if rem <= 0:
                        break
                    if self._retarget.wait(min(rem, 0.05)):
                        self._retarget.clear()
                        nd = self._new_deadline
                        if nd:
                            # accept past-due: release ASAP, never ignore
                            deadline = max(nd,
                                           time.perf_counter() + 0.002)
                if self.active_cycle:
                    self._waiter.wait_until(
                        deadline, float(config["spin_margin_ms"]))
                    if self.active_cycle:
                        self.active_cycle = False
                        t_rel = time.perf_counter()
                        _inject_release()
                        held = (t_rel - press) * 1000.0
                        err = held - (deadline - press) * 1000.0
                        stats.record(err)
                        if config["debug"]:
                            log.info("[shot] K held %.2f ms (err %+.2f ms)",
                                     held, err)
            except Exception:
                log.exception("[worker] release failed")
            finally:
                with _state_lock:
                    _cycle_active = False


# --------------------------------------------------------------------------- #
# MeterEye — watches the horizontal PURPLE meter, schedules the release
# when the magenta fill front reaches the green tick at the bar's tip.
# --------------------------------------------------------------------------- #
class MeterEye:
    # Magenta fill: measured H 150-151. The upper bound used to run to 162,
    # which reaches into court reds (Bulls logo sits right of the meter) —
    # stray red past the fill front inflated both the span and the measured
    # fill speed, firing the release early. Kept tight around the real hue.
    MAG_LO = (146, 120, 150)
    MAG_HI = (157, 255, 255)
    TICK_LO = (45, 60, 90)       # green tick: measured H 53-65
    TICK_HI = (80, 255, 255)
    BAR_FRAC = 0.082             # bar length / window width (measured)

    def __init__(self, release_worker: "ReleaseWorker") -> None:
        self.worker = release_worker
        self._armed = threading.Event()
        self._over_hist = []   # measured overshoot per shot, for _autotune
        self._press = 0.0
        self._rect = None
        self._rect_t = -10.0
        self.last = "idle"
        self.persistent_span = 118  # Learned default span for player's custom meter size
        self._tick_span = None      # per-shot fill->tick distance (scales)
        threading.Thread(target=self._run, name="metereye",
                         daemon=True).start()

    def arm(self, press_time: float) -> None:
        self._press = press_time
        self._last_v = 0.30
        self._armed.set()

    def _game_rect(self):
        try:
            u32 = ctypes.windll.user32
            hwnd = u32.FindWindowW(None, config["game_window"])
            if not hwnd:
                return None
            r = wt.RECT()
            if not u32.GetClientRect(hwnd, ctypes.byref(r)):
                return None
            pt = wt.POINT(0, 0)
            u32.ClientToScreen(hwnd, ctypes.byref(pt))
            w, h = r.right - r.left, r.bottom - r.top
            if w < 400 or h < 300:
                return None
            return pt.x, pt.y, w, h
        except Exception:
            return None

    def _grab(self, camera, sct, x, y, w, h):
        if camera is not None:
            region = (int(x), int(y), int(x + w), int(y + h))
            frame = camera.grab(region=region)
            if frame is not None:
                return _cv2.cvtColor(frame, _cv2.COLOR_RGB2HSV)
        shot = sct.grab({"left": int(x), "top": int(y),
                         "width": int(w), "height": int(h)})
        img = _np.frombuffer(shot.bgra, _np.uint8).reshape(
            shot.height, shot.width, 4)[:, :, :3]
        return _cv2.cvtColor(img, _cv2.COLOR_BGR2HSV)

    def _autotune(self, over_ms: float) -> None:
        """Nudge latency_ms from measured overshoot.

        `over_ms` is how late the release landed: positive means the fill
        overshot the target (late), negative means it stopped short
        (early). Raising latency_ms releases EARLIER, so the correction
        moves with the sign of the error.

        Acts as soon as the error is real, not after a fixed shot count.
        A single shot carries ~one game frame (~17 ms) of noise, so one
        sample can never justify a move — but four shots all reading -13 ms
        obviously can, and waiting for a magic 8th is just lost games.

        So it asks the only question that matters after EVERY shot: is the
        error bigger than the noise in the samples I have? It compares the
        median error against its own standard error (from the MAD, so a
        single wild shot can't fake confidence). A large consistent error
        clears that bar at 2-3 shots and gets corrected immediately; a
        small ambiguous one keeps collecting until it either becomes clear
        or hits `autotune_shots` and gets acted on anyway.

        Step size scales with the error too — a 30 ms miss is not corrected
        in 8 ms crawls, but the gain is under 1.0 so it converges instead
        of oscillating.
        """
        if not config.get("autotune", True):
            return
        self._over_hist.append(float(over_ms))
        h = self._over_hist
        n = len(h)
        if n < 3:                       # 1-2 samples are never evidence
            return
        s = sorted(h)
        median = s[n // 2]
        # Aim at a setpoint, not at zero. The magenta keeps animating for a
        # beat after 2K registers the release, so a fill that stops exactly
        # on target was actually released BEFORE it — 2K scores that "Early".
        # Badge-verified shots at -35, -13, -4 and +7 ms ALL came back Early,
        # which is what proved the offset is real. Chasing zero walked
        # latency_ms 53 -> 67 in the wrong direction over two batches.
        aim = float(config.get("autotune_target_ms", 25))
        err = median - aim
        dead = float(config.get("autotune_deadband_ms", 6))
        cap = int(config.get("autotune_shots", 8))

        # Spread of the samples themselves, via MAD -> robust sigma.
        mad = sorted(abs(x - median) for x in h)[n // 2]
        # Floor at the real per-shot noise (~1 game frame). Claiming
        # less makes 2 identical readings look like proof and the
        # loop lurches past the target on a coincidence.
        sigma = max(1.4826 * mad, 14.0)
        stderr = sigma / math.sqrt(n)
        # Real if it clears both the deadband and 2 standard errors.
        sure = abs(err) > max(dead, 2.0 * stderr)
        if not sure and n < cap:
            return

        self._over_hist = []
        if abs(err) < dead:
            log.info("[autotune] %d shots, median %+.0f ms (aim %+.0f) — "
                     "inside deadband, holding latency_ms=%d",
                     n, median, aim, int(config["latency_ms"]))
            return
        step = max(-15.0, min(15.0, 0.7 * err))   # scale, but damped
        new = int(round(config["latency_ms"] + step))
        new = max(0, min(300, new))
        if new == int(config["latency_ms"]):
            return
        log.info("[autotune] %d shots, median %+.0f ms vs aim %+.0f "
                 "= %+.0f ms (%s, +-%.0f) — latency_ms %d -> %d",
                 n, median, aim, err,
                 "late" if err > 0 else "early", stderr,
                 int(config["latency_ms"]), new)
        config["latency_ms"] = new
        save_config(config)

    def _find_tick(self, hsv, x_from, x_to, y0, y1):
        """Green tick inside the row band, to the right of the fill.
        Returns tick centre x (crop coords) or None."""
        x_from = max(0, x_from); x_to = min(hsv.shape[1], x_to)
        y0 = max(0, y0); y1 = min(hsv.shape[0], y1)
        if x_to - x_from < 3 or y1 - y0 < 3:
            return None
        g = _cv2.inRange(hsv[y0:y1, x_from:x_to],
                         self.TICK_LO, self.TICK_HI)
        
        # Filter out the thin green player indicator ring using connected components
        ncc, lab, st, ce = _cv2.connectedComponentsWithStats(g)
        best_tick = None
        best_area = 0
        for i in range(1, ncc):
            tx, ty, tw, th, ta = st[i]
            # Extremely sensitive tick detection for thin Hall of Fame green ticks
            if th >= 3 and ta >= 2:
                if ta > best_area:
                    best_area = ta
                    comp_mask = (lab == i)
                    xs = _np.nonzero(comp_mask.any(axis=0))[0]
                    best_tick = x_from + int(xs.mean())
        return best_tick

    def _locate(self, camera, sct, rect):
        """Find the purple bar. Returns
        ((bar_y0, bar_y1, front_x, end_x) or None, n_magenta_blobs)."""
        gx, gy, gw, gh = rect
        hsv = self._grab(camera, sct, gx, gy, gw, gh)
        m = _cv2.inRange(hsv, self.MAG_LO, self.MAG_HI)
        ncc, lab, st, ce = _cv2.connectedComponentsWithStats(m)
        cands = []
        for i in range(1, ncc):
            x, y, w, h, a = st[i]
            if a >= 40 and 8 <= h <= 45 and w >= 3 and w <= 220:
                cands.append((a, x, y, w, h))
        
        span_px = float(config.get("persistent_span", 118))
        for a, x, y, w, h in sorted(cands, reverse=True)[:3]:
            solid = a / max(w * h, 1)
            if 12 <= h <= 40 and solid >= 0.70:
                end = x + int(span_px)
                return (y - 6, y + h + 6, x + w, end), len(cands)
        return None, len(cands)

    def _release(self, press, t_anchor, why, offset_ms):
        target = press + t_anchor + (offset_ms
                                     - config["meter_lead_ms"]) / 1000.0
        target = max(target, press + 0.15)
        target = min(target,
                     press + config["meter_max_ms"] / 1000.0 - 0.03)
        
        # Fast-path: cancel the background worker thread and release directly
        self.worker.cancel()
        
        # High-precision wait in this thread (since it's already time-critical)
        waiter = PrecisionWaiter()
        waiter.wait_until(target, float(config["spin_margin_ms"]))
        
        t_rel = time.perf_counter()
        _inject_release()
        
        held = (t_rel - press) * 1000.0
        err = held - (target - press) * 1000.0
        stats.record(err)
        
        self.last = f"{why} {t_anchor*1000:.0f} ms"
        if config["debug"]:
            log.info("[meter] %s at %.0f ms -> released directly at %.2f ms (err %+.2f ms)",
                     why, t_anchor * 1000, (t_rel - press) * 1000, err)

    def _run(self) -> None:
        _boost_current_thread()
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
        camera = None
        if not config.get("force_mss", False):
            try:
                import dxcam
                camera = dxcam.create()
                log.info("[capture] backend: dxcam (GPU)")
            except Exception as e:
                log.info("[capture] dxcam initialization failed: %s (falling back to mss)", e)
                camera = None
        else:
            log.info("[capture] backend: mss (CPU) forced by config")
        try:
            with _MSS() as sct:
                while not _shutdown.is_set():
                    if not self._armed.wait(0.5):
                        continue
                    self._armed.clear()
                    try:
                        self._track(camera, sct)
                    except Exception:
                        log.exception("[meter] tracking failed")
                        # Never let a tracking bug ride the 1200 ms safety
                        # guard — that holds K nearly twice as long as a
                        # normal shot and 2K scores it a hard "Late". Fall
                        # back to the same press-anchored hold the no-meter
                        # path uses, so a crash costs a normal-length shot
                        # instead of a guaranteed miss.
                        try:
                            fallback = (self._press
                                        + float(config["hold_ms"]) / 1000.0)
                            self.worker.retarget(
                                max(fallback, time.perf_counter() + 0.005))
                            log.warning("[meter] fell back to %d ms hold",
                                        int(config["hold_ms"]))
                        except Exception:
                            log.exception("[meter] fallback retarget failed")
        finally:
            if camera is not None:
                try:
                    camera.release()
                    log.info("[capture] released dxcam instance")
                except Exception:
                    pass

    def _track(self, camera, sct) -> None:
        press = self._press
        now = time.perf_counter()
        if self._rect is None or now - self._rect_t > 5.0:
            self._rect = self._game_rect()
            self._rect_t = now
        rect = self._rect
        if rect is None:
            self.last = "no game window"
            return
        gx, gy, gw, gh = rect
        guard = press + config["meter_max_ms"] / 1000.0 - 0.06
        lock = None            # (band_y0, band_y1, front_x, target_x) window
        cands_seen = 0
        gave_up = False        # blind fallback armed, but still watching
        gave_up_at = 0.0       # when that fallback release is due
        seen = 0
        lost = 0
        hist = []              # (t, fill_w_px)
        tick_offset = None     # locked-in green tick offset from left edge
        dbg = (-1, -1)         # last front, target
        t_found = None
        front_at_found = None
        # Fill LENGTH at the first tracked frame. Velocity must be measured
        # from this, never from the absolute front: the meter is anchored to
        # the shooter, so 2K's camera pan slides the whole bar sideways while
        # it fills. The front then moves for two reasons and we would credit
        # all of it to fill speed — panning toward centre (shooter on the
        # left) inflated v by ~40%, and since lead_px and the scheduled wait
        # both divide by v, that fired the release early on every left-side
        # shot. front - left cancels the translation out.
        fw_at_found = None
        t_fw = None
        prev_fill_w = None     # staircase step tracking (see fill_est)
        step_gaps = []         # observed gaps between fill steps = game frames
        # Running sums for a least-squares fit of fill vs time. Both of the
        # old velocity estimates were TWO-POINT: they subtracted the ends of
        # a window and threw away the ~55 samples between. The fill is a
        # staircase, so quantising either endpoint by one step is a ~9%
        # error on v over a 60 ms window, which becomes ~2 ms of release
        # error once multiplied by the prediction horizon. Fitting a line
        # through every sample averages the staircase out instead.
        lsq_n = 0
        lsq_sx = lsq_sy = lsq_sxx = lsq_sxy = 0.0
        last_step_fill = None  # fill at the last REAL (>=2px) advance
        last_step_t = None
        fill_at_step = None
        t_step = None
        t_prev = None
        active_ms = 0.0        # elapsed time with capture stalls removed
        stalls_ms = 0.0        # how much stall we discarded, for the log
        while not _shutdown.is_set():
            now = time.perf_counter()
            if now >= guard:
                self.last = ("meter never found" if lock is None
                             else "no release signal")
                log.warning("[meter] %s — safety release at %d ms "
                            "(last front %d tick %d)",
                            self.last, int(config["meter_max_ms"]),
                            dbg[0], dbg[1])
                return
            if lock is None:
                # Stop once the blind fallback has fired. Without this the
                # thread spins to the 1200 ms guard and logs a "meter never
                # found — safety release" line for a shot that already
                # released correctly at hold_ms.
                if gave_up and now >= gave_up_at - 0.02:
                    return
                if now - press < 0.080:  # Ignore first 80ms to avoid fading meters or quick false locks
                    time.sleep(0.002)
                    continue
                found, ncand = self._locate(camera, sct, rect)
                cands_seen = max(cands_seen, ncand)
                if found:
                    lock = found
                    seen = 1
                    t_found = now
                    front_at_found = found[2]
                    # The meter renders larger when the player is nearer the
                    # camera (span ~110 close, ~106 far). fill_w and v are in
                    # real pixels and scale with it, so a fixed target length
                    # is short on big meters (early) and long on small ones
                    # (late). Scale the arrow target by the span we just
                    # measured instead.
                    _ts = found[3] - found[2]
                    self._tick_span = _ts if 60 <= _ts <= 200 else None
                    if config["debug"]:
                        log.info("[meter] purple bar found at +%.0f ms "
                                 "(fill %d -> tick %d, span %d, target %d)",
                                 (now - press) * 1000,
                                 found[2], found[3], found[3] - found[2],
                                 int((found[3] - found[2])
                                     * float(config.get("arrow_ratio", 1.227)))
                                 if self._tick_span
                                 else int(config.get("persistent_span", 130)))
                elif (not gave_up
                      and now - press > float(config["no_meter_ms"]) / 1000.0
                      and cands_seen == 0):
                    # No meter this shot. Fall back to a normal-length hold
                    # anchored on the PRESS, not on the moment we gave up —
                    # otherwise the give-up delay is added to the hold and
                    # the shot releases far later than a metered one.
                    target = press + float(config["hold_ms"]) / 1000.0
                    self.worker.retarget(max(target, now + 0.005))
                    self.last = "no meter — quick abort"
                    if config["debug"]:
                        log.info("[meter] no meter within %d ms — holding "
                                 "%d ms total (release in %.0f ms)",
                                 int(config["no_meter_ms"]),
                                 int(config["hold_ms"]),
                                 (target - now) * 1000.0)
                    # Arm the fallback but KEEP LOOKING. The release isn't
                    # until press+hold_ms, and meters have been observed
                    # appearing as late as +596 ms — giving up at
                    # no_meter_ms threw away every meter that landed in
                    # the gap between the two, for no gain. If one shows
                    # up before the fallback fires, tracking takes over
                    # and retargets; if not, the blind hold stands.
                    gave_up = True
                    gave_up_at = target
                continue
            band_y0, band_y1, front_x, target_x = lock
            # Tight crop around the BAR ONLY — a small grab keeps the
            # capture+process loop short, which is pure release latency.
            cx0 = max(0, target_x - int(gw * self.BAR_FRAC) - 40)
            cx1 = min(gw, target_x + 35)
            cy0 = max(0, band_y0 - 20)
            cy1 = min(gh, band_y1 + 20)
            hsv = self._grab(camera, sct, gx + cx0, gy + cy0,
                             cx1 - cx0, cy1 - cy0)
            t_now = time.perf_counter() - press
            m = _cv2.inRange(hsv, self.MAG_LO, self.MAG_HI)
            npx = int(_np.count_nonzero(m))
            if npx >= 25:
                seen += 1
                lost = 0
                xs = _np.nonzero(m.any(axis=0))[0]
                ys = _np.nonzero(m.any(axis=1))[0]
                front = cx0 + int(xs.max())
                left = cx0 + int(xs.min())
                # Fill LENGTH, not front position — see fw_at_found above.
                fill_w = front - left
                if fw_at_found is None:
                    fw_at_found = fill_w
                    t_fw = now
                    t_prev = now
                else:
                    # Cap each frame's contribution: anything longer than
                    # stall_cap_ms is a dropped/late frame, not fill time.
                    _cap = float(config.get("stall_cap_ms", 20)) / 1000.0
                    _gap = now - t_prev
                    if _gap > _cap:
                        stalls_ms += (_gap - _cap) * 1000.0
                    active_ms += min(_gap, _cap) * 1000.0
                    t_prev = now
                # Static tracking safety: if we have tracked for > 150ms but the fill has not grown by at least 8px,
                # we are likely locked onto a static background element or jersey.
                if now - t_fw > 0.150 and fill_w - fw_at_found < 8:
                    self._release(press, t_now, "static lock-on abort", 0)
                    return
                band_y0 = cy0 + int(ys.min()) - 20
                band_y1 = cy0 + int(ys.max()) + 20
                if getattr(self, "_tick_span", None):
                    expected_len = (float(self._tick_span)
                                    * float(config.get("arrow_ratio", 1.227)))
                else:
                    expected_len = float(config.get("persistent_span", 130))
                target = left + expected_len
                lock = (band_y0, band_y1, front, target)
                dist = target - front
                dbg = (front, target)
                hist.append((t_now, fill_w))
                while len(hist) > 1 and t_now - hist[0][0] > 0.060:
                    hist.pop(0)
                # Release when the fill is `latency_ms` short of full,
                # measured in the fill's OWN speed. The screen we read
                # lags the game's internal state by a fixed time, so the
                # compensation must be a time converted to pixels — a
                # fixed % cannot track it when the fill speed changes.
                span = expected_len
                # Timestamp each staircase step here; the extrapolation
                # itself happens below, once v is known.
                if fill_w != prev_fill_w:
                    # Frame-period sampling. Require a REAL advance: fill_w
                    # is front-minus-left and the camera pans both edges, so
                    # a 1 px pan drift looks like a step and would drag the
                    # measured period below the true frame time. One rendered
                    # frame moves the fill ~5 px, so >=2 px separates a real
                    # step from pan noise.
                    if last_step_fill is None:
                        last_step_fill, last_step_t = fill_w, now
                    elif fill_w - last_step_fill >= 2:
                        _g = now - last_step_t
                        if 0.005 < _g < 0.040:
                            step_gaps.append(_g)
                        last_step_fill, last_step_t = fill_w, now
                    prev_fill_w = fill_w
                    fill_at_step = fill_w
                    t_step = now
                remain = span - fill_w
                # px/ms from the last few fill-length samples (60ms window)
                lsq_n += 1
                lsq_sx += t_now
                lsq_sy += fill_w
                lsq_sxx += t_now * t_now
                lsq_sxy += t_now * fill_w

                v = 0.0
                if len(hist) >= 2:
                    dt = (hist[-1][0] - hist[0][0]) * 1000.0
                    adv = hist[-1][1] - hist[0][1]
                    if dt > 15 and adv > 0:
                        v = adv / dt
                # The 60 ms window is only 3-5 frames, so one noisy reading
                # skews it badly — and both lead_px and the scheduled wait
                # divide by v, so the error lands on the release twice.
                # Prefer a long baseline over the whole fill: same slope,
                # far less noise.
                if t_fw is not None:
                    # Elapsed time with capture stalls excluded. Wall clock
                    # keeps running when frames stop arriving, but the fill
                    # we can SEE does not advance — so a stall inflates the
                    # denominator only, reads v too low, and a low v shrinks
                    # lead_px and releases late. One 70 ms hiccup did exactly
                    # that: v 0.27 (session low) and a 725 ms hold, ~95 ms
                    # longer than every other shot. Capping each frame's
                    # contribution keeps a stall from being counted as time
                    # the fill was observed standing still.
                    dt_long = active_ms
                    adv_long = fill_w - fw_at_found
                    if dt_long > 120 and adv_long > 0:
                        v_long = adv_long / dt_long
                        w = float(config.get("v_long_weight", 0.75))
                        v = (w * v_long + (1.0 - w) * v) if v > 0.05 else v_long
                if v > 0.05:
                    # Least-squares slope wins when we have enough of the
                    # fill to fit. Guarded on the denominator and on a sane
                    # result so a degenerate fit can never poison the
                    # release; the two-point estimate stays as the fallback.
                    v_fit = 0.0
                    if config.get("v_lsq", True) and lsq_n >= 12:
                        _den = lsq_n * lsq_sxx - lsq_sx * lsq_sx
                        if _den > 1e-9:
                            _sl = (lsq_n * lsq_sxy - lsq_sx * lsq_sy) / _den
                            _sl /= 1000.0          # px/s -> px/ms
                            if 0.05 < _sl < 1.0:
                                v_fit = _sl
                                v = _sl
                    self._last_v = v
                elif hasattr(self, "_last_v"):
                    v = self._last_v
                else:
                    v = 0.30  # Default fallback
                # The game renders at ~60 fps while we capture at ~200 Hz,
                # so fill_w is a staircase: it holds for ~3 captures, then
                # jumps ~5 px. Reading it raw means the fill is stale by
                # 0-16.7 ms depending purely on where in the game's frame
                # we happened to look — ~8 ms late on average with ~5 ms
                # of spread. The constant part gets absorbed by
                # arrow_ratio; the SPREAD is what makes one setting land
                # early on one shot and late on the next. Extrapolating
                # from the last step keeps the estimate continuous
                # instead of lagging the last drawn frame.
                fill_est = fill_w
                if (config.get("subframe_fill", True)
                        and v > 0.05 and t_step is not None):
                    # Never extrapolate past one game frame: a missing
                    # step means the fill stopped, not that it kept going.
                    age_ms = min((now - t_step) * 1000.0,
                                 float(config.get("step_cap_ms", 18)))
                    fill_est = fill_w + v * age_ms
                    remain = span - fill_est
                # Use the configured latency for both jumpshots and layups to keep timing consistent
                lead_px = v * float(config["latency_ms"])
                cap_px = span * float(config["release_pct"]) / 100.0
                # Stall detection: front has not advanced by more than 1px for at least 30ms near the end (>= 95% full)
                stalled = False
                if len(hist) >= 2:
                    t_latest, pos_latest = hist[-1]
                    for t_old, pos_old in reversed(hist[:-1]):
                        if t_latest - t_old >= 0.030:
                            if pos_latest - pos_old <= 1 and fill_w >= 0.95 * span:
                                stalled = True
                            break
                base_min_pct = float(config.get("min_lead_pct", 82))
                if v >= 0.41:
                    # Fast fills need a BIGGER lead in pixels, not a later
                    # release — forcing 95% here made every fast meter fire
                    # after the lead the speed had already earned.
                    min_lead_pct = max(base_min_pct,
                                       float(config.get("fast_min_lead_pct",
                                                        88.0))) / 100.0
                else:
                    min_lead_pct = base_min_pct / 100.0
                # Sub-frame scheduling: fire on the frame BEFORE the ideal
                # crossing and schedule the release for the exact moment it
                # happens. Waiting for `remain <= lead_px` meant the trigger
                # landed wherever the next capture happened to fall — up to a
                # frame of fill (~4-5px, ~15 ms) past the ideal point, and
                # always on the late side.
                look_ms = float(config.get("lookahead_ms", 25))
                lead_ready = (v > 0 and fill_w >= min_lead_pct * span
                              and remain <= lead_px + v * look_ms)
                if seen >= 2 and t_now >= 0.10 and (
                        lead_ready or fill_w >= cap_px - 1 or stalled):
                    pct = 100.0 * fill_w / span
                    why = ("LEAD" if lead_ready
                           else ("CAP" if not stalled else "STALL"))
                    # Seconds until remain == lead_px at the current speed.
                    wait_s = 0.0
                    if lead_ready and v > 0:
                        wait_s = max(0.0, (remain - lead_px) / v / 1000.0)
                    # Land in the MIDDLE of a game frame, not near its edge.
                    #
                    # 2K polls input once per rendered frame, so releasing
                    # anywhere inside a frame gives the same result — only
                    # which frame matters. Scheduling to a bare computed
                    # instant can put us a hair from a boundary, where 3 ms
                    # of pipeline jitter tips the release into the next
                    # frame. That is not a 3 ms error, it is a full 16.7 ms
                    # one, and it is what turns an Excellent into a Late
                    # with nothing changed.
                    #
                    # Every step in fill_w IS a frame boundary, so t_step
                    # gives us the phase for free. Shifting to frame centre
                    # costs at most half a frame and buys ~8 ms of margin
                    # on both sides.
                    align_ms = 0.0
                    frame_ms = 0.0
                    frame_spread = "-"
                    if (config.get("frame_align", True)
                            and lead_ready and t_step is not None):
                        _cfg_fr = float(config.get("game_frame_ms",
                                                   13.3)) / 1000.0
                        frame = _cfg_fr
                        if len(step_gaps) >= 6:
                            _sg = sorted(step_gaps)
                            _med = _sg[len(_sg) // 2]
                            # Only trust a measurement in a plausible band.
                            # Outside it, something is polluting the steps
                            # and the configured period is the safer bet.
                            if 0.0100 <= _med <= 0.0230:
                                frame = _med
                            # Spread of the step gaps. A real render clock
                            # is tight (p25~p50~p75); a wide spread means
                            # something is splitting frames and the median
                            # is an underestimate, not a frame time.
                            _q = (_sg[len(_sg) // 4] * 1000.0,
                                  _sg[len(_sg) * 3 // 4] * 1000.0)
                            frame_spread = "%.1f/%.1f" % _q
                        d = ((press + t_now + wait_s) - t_step) % frame
                        shift = frame / 2.0 - d
                        if shift <= -frame / 2.0:
                            shift += frame
                        wait_s = max(0.0, wait_s + shift)
                        align_ms = shift * 1000.0
                        frame_ms = frame * 1000.0
                    # How far the whole bar slid sideways since detection —
                    # this is the camera pan the old front-based velocity
                    # was mistaking for fill speed.
                    pan_px = (left - front_at_found
                              if front_at_found is not None else 0.0)
                    # Mean interval between usable frames. This sets the
                    # floor on release precision: we can only ever act on
                    # the frames we actually get, and lookahead_ms has to
                    # be at least one interval to fire BEFORE the crossing.
                    frame_dt = (active_ms / max(1, seen - 1)) if seen > 1 else 0.0
                    self._release(
                        press, t_now + wait_s,
                        f"{why} {pct:.0f}% rem{remain}px "
                        f"v{v:.2f}px/ms comp{int(config['latency_ms'])}ms "
                        f"sched+{wait_s * 1000:.1f}ms x{int(left)} "
                        f"pan{pan_px:+.0f}px stall{stalls_ms:.0f}ms "
                        f"frm{seen} dt{frame_dt:.1f}ms sub{(fill_est - fill_w):+.1f}px "
                        f"align{align_ms:+.1f}ms fr{frame_ms:.1f}ms"
                        f"[{frame_spread}]n{len(step_gaps)}",
                        config["meter_offset_ms"])
                    
                    # --- Post-release latency & error measurement ---
                    try:
                        t_inject = time.perf_counter()
                        t_limit = t_inject + 0.150  # Grab for 150ms
                        post_hist = []
                        
                        while time.perf_counter() < t_limit:
                            hsv_post = self._grab(camera, sct, gx + cx0, gy + cy0, cx1 - cx0, cy1 - cy0)
                            t_frame = time.perf_counter()
                            m_post = _cv2.inRange(hsv_post, self.MAG_LO, self.MAG_HI)
                            npx_post = int(_np.count_nonzero(m_post))
                            if npx_post >= 25:
                                xs_post = _np.nonzero(m_post.any(axis=0))[0]
                                curr_front = cx0 + int(xs_post.max())
                                post_hist.append((t_frame, curr_front))
                            time.sleep(0.001)
                            
                        if post_hist:
                            # Robustly find when the bar stopped growing by finding the first stable frame
                            stop_idx = None
                            for i in range(len(post_hist)):
                                is_stable = True
                                for j in range(i + 1, len(post_hist)):
                                    if post_hist[j][1] > post_hist[i][1] + 1:
                                        is_stable = False
                                        break
                                if is_stable:
                                    stop_idx = i
                                    break
                            
                            if stop_idx is not None:
                                t_observed = post_hist[stop_idx][0]
                                max_pos = post_hist[stop_idx][1]
                                measured_latency_ms = (t_observed - t_inject) * 1000.0
                                
                                # Determine the shot timing direction
                                # If it stopped at or past the target boundary
                                if max_pos >= target - 1:
                                    # If it stabilized almost immediately, it hit the boundary before release registered (LATE)
                                    if measured_latency_ms < 25.0:
                                        direction = "Late"
                                        time_error_ms = -15.0  # Fixed error to trigger a step adjustment
                                    else:
                                        direction = "Perfect"
                                        time_error_ms = 0.0
                                else:
                                    direction = "Early"
                                    # Calculate actual time error based on fill speed
                                    time_error_ms = (target - max_pos) / v if v > 0.05 else 0.0

                                # Debug-only, and no longer says "Perfect":
                                # this check only asks whether the fill
                                # reached `target`, which magenta always
                                # does, so it labelled shots 2K scored
                                # "Slightly Late" as perfect. Kept as raw
                                # numbers for diagnostics; never trust the
                                # verdict, and never tune from it.
                                # The VERDICT here is meaningless (it only
                                # asks whether the fill reached target,
                                # which magenta always does). But
                                # stopped-minus-target is a real, direct
                                # measurement of how late the release
                                # landed, in pixels of fill — the only
                                # ground truth available without reading
                                # 2K's badge. Positive = overshot = late.
                                over = max_pos - target
                                over_ms = over / v if v > 0.05 else 0.0
                                # Only feed the controller shots it actually
                                # tracked. A meter picked up late gives a
                                # truncated span and a 2-frame velocity guess,
                                # and its overshoot is noise dressed up as a
                                # measurement — one of those in a batch of 8
                                # can drag latency_ms 15 ms off target.
                                min_frm = int(config.get("trust_min_frames", 8))
                                min_span = int(config.get("trust_min_span", 95))
                                bad = ("frames" if seen < min_frm
                                       else "span" if span < min_span
                                       else None)
                                log.info("[overshoot] %+d px (%+.0f ms) "
                                         "target %d stopped %d "
                                         "span %d frm %d%s",
                                         over, over_ms, target, max_pos,
                                         span, seen,
                                         f" — UNTRUSTED ({bad})" if bad else "")
                                if bad:
                                    log.info("[autotune] sample discarded: "
                                             "meter was not tracked cleanly")
                                else:
                                    self._autotune(over_ms)
                                
                                # --- Closed-loop Auto-calibration ---
                                # Only calibrate on jumpshots to avoid layup anomalies
                                # Disabled by default: the "Perfect" test
                                # below only checks that the fill reached
                                # `target`, which magenta always does — so it
                                # scores late shots as perfect and walks
                                # latency_ms the wrong way. Set "autocal":
                                # true to re-enable.
                                if config.get("autocal", False) and 0.05 < v < 0.41:
                                    if direction == "Early":
                                        # Proportional correction: release later
                                        correction = -time_error_ms * 0.5
                                    elif direction == "Late":
                                        # Fixed step correction: release earlier
                                        correction = +4.0
                                    else:
                                        correction = 0.0
                                    
                                    if abs(correction) > 0.1:
                                        old_latency = config["latency_ms"]
                                        new_latency = max(10.0, min(80.0, old_latency + correction))
                                        new_latency = round(new_latency, 1)
                                        if new_latency != old_latency:
                                            config["latency_ms"] = new_latency
                                            save_config(config)
                                            log.info("[measure] Auto-calibrated latency: %.1f ms -> %.1f ms (applied %+.1f ms correction)",
                                                     old_latency, new_latency, correction)
                    except Exception as e:
                        log.debug("Post-release measurement failed: %s", e)
                    
                    return
                # Optional look-ahead (config "predict": true). Off by
                # default — prediction estimates the arrival time and
                # any error in it shows up as early/late shots.
                if config["predict"] and len(hist) >= 3:
                    dt = hist[-1][0] - hist[0][0]
                    adv = hist[-1][1] - hist[0][1]   # px advanced
                    if dt > 0.02 and adv >= 2:
                        v = adv / dt                 # px per second
                        if v >= 25 and dist > 0 and dist / v <= 0.30:
                            self._release(
                                press, t_now + dist / v,
                                f"FULL predicted d{dist} v{v:.0f}",
                                config["meter_offset_ms"])
                            return
            else:
                lost += 1
                # Meter vanished mid-fill: release immediately rather
                # than hold blind.
                if seen >= 4 and lost >= 3:
                    self._release(press, t_now, "meter lost", 0)
                    return
                if lost >= 10:
                    lock = None
                    hist = []
                    seen = 0
                    lost = 0
                    tick_age = 99


worker = None
meter_eye = None

# --------------------------------------------------------------------------- #
# K hook — pass the press through, suppress the physical release;
# the worker owns the synthetic release.
# --------------------------------------------------------------------------- #
def _hook(event) -> bool:
    now = time.perf_counter()
    try:
        return _hook_impl(event, now)
    except Exception:
        log.exception("[hook] error")
        return True


def _game_focused() -> bool:
    """Is NBA 2K26 the foreground window?

    Typing a 'k' in any other app used to arm a shot: the hook is global,
    so it fired while the game sat idle in the background, the meter was
    never found, and the log filled with phantom failures. Cheap Win32
    call, and on any error it returns True so a broken check can never
    silently stop the tool from working in-game.
    """
    try:
        u32 = ctypes.windll.user32
        fg = u32.GetForegroundWindow()
        if not fg:
            return True
        n = u32.GetWindowTextLengthW(fg)
        buf = ctypes.create_unicode_buffer(n + 1)
        u32.GetWindowTextW(fg, buf, n + 1)
        return config["game_window"].lower() in buf.value.lower()
    except Exception:
        return True


def _hook_impl(event, now: float) -> bool:
    global _physically_held, _passthrough_held, _cycle_active
    if _injecting:
        return True

    if event.event_type == keyboard.KEY_DOWN:
        if _physically_held:
            return _passthrough_held      # auto-repeat
        _physically_held = True
        if not active or not _game_focused():
            _passthrough_held = True
            return True
        with _state_lock:
            if not _cycle_active:
                _cycle_active = True
                worker.schedule(now, config["meter_max_ms"])
                if meter_eye is not None:
                    meter_eye.arm(now)
                if config["debug"]:
                    log.info("[trigger] K down -> shot armed "
                             "(meter read)")
                return True               # game receives the press
        return False                      # tap during active cycle

    # KEY_UP
    was_held = _physically_held
    _physically_held = False
    if _passthrough_held:
        _passthrough_held = False
        return True
    if not was_held and not _cycle_active:
        return True
    return False                          # timer owns the release


# --------------------------------------------------------------------------- #
# Config hot-reload
# --------------------------------------------------------------------------- #
_HOT_FIELDS = ("hold_ms", "no_meter_ms", "fast_min_lead_pct", "arrow_ratio", "lookahead_ms", "v_long_weight", "v_lsq", "stall_cap_ms", "subframe_fill", "step_cap_ms", "frame_align", "game_frame_ms", "autotune", "autotune_shots", "autotune_deadband_ms", "autotune_target_ms", "trust_min_frames", "trust_min_span", "spin_margin_ms", "latency_ms", "release_pct", "min_lead_pct",
               "meter_offset_ms", "meter_lead_ms", "meter_max_ms",
               "predict", "force_mss", "debug")


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
            log.info("[config] reloaded: %s",
                     ", ".join(f"{k}={v}" for k, v in changed.items()))


# --------------------------------------------------------------------------- #
# Status / runtime controls
# --------------------------------------------------------------------------- #
def print_status() -> None:
    s = stats.snapshot()
    print("\n" + "=" * 56)
    print(f"     K-Meter v{__version__}  (purple-meter button shot)")
    print("=" * 56)
    print(f"  Status    : {'ACTIVE' if active else 'PAUSED'}")
    print(f"  Key       : {config['action_key']}  (tap once)")
    print(f"  Meter     : fire {int(config['latency_ms'])} ms before the "
          f"shape fills  (cap {int(config['release_pct'])}%)")
    print(f"  Safety    : {int(config['meter_max_ms'])} ms max hold"
          f"   engine spin {float(config['spin_margin_ms']):.1f} ms"
          f" {'high-res' if worker and worker.hr_timer else 'standard'}")
    if meter_eye is not None:
        print(f"  Last shot : {meter_eye.last}")
    print(f"  Shots     : {s['count']}   inject err >1ms: {s['misses']}")
    if s["count"]:
        print(f"  Inject err: last {s['last']:+.2f}  avg {s['mean']:+.2f}"
              f"  sd {s['std']:.2f} ms")
    print("-" * 56)
    print("  F5 pause   F7 earlier +10ms   F6 later -10ms")
    print("             F4 earlier +3ms    F3 later -3ms")
    print("  F8 status   F10 reset stats   F9 quit")
    print("=" * 56 + "\n")


def adjust_offset(delta: int) -> None:
    config["meter_offset_ms"] = int(
        max(-300, min(800, config["meter_offset_ms"] + delta)))
    save_config(config)
    log.info("[tune] meter offset = %+d ms", config["meter_offset_ms"])


def adjust_latency(delta: int) -> None:
    """+delta = compensate more = release EARLIER."""
    config["latency_ms"] = int(
        max(0, min(300, config["latency_ms"] + delta)))
    save_config(config)
    log.info("[tune] latency comp = %d ms  (shots fire %s)",
             config["latency_ms"], "EARLIER" if delta > 0 else "LATER")


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
_ERROR_ALREADY_EXISTS = 183


def _single_instance() -> bool:
    try:
        k32 = ctypes.windll.kernel32
        k32.CreateMutexW(None, False, "KMeter_single_instance")
        return k32.GetLastError() != _ERROR_ALREADY_EXISTS
    except Exception:
        return True


def main() -> None:
    global worker, meter_eye

    if not _single_instance():
        log.error("[fatal] another K-Meter is already running — close "
                  "that window first")
        return
    if not _VISION_OK:
        log.error("[fatal] vision deps missing: "
                  "pip install opencv-python mss numpy")
        return

    worker = ReleaseWorker()
    meter_eye = MeterEye(worker)
    threading.Thread(target=_config_watcher, name="cfgwatch",
                     daemon=True).start()

    keyboard.hook_key(config["action_key"], _hook, suppress=True)

    controls = {
        "f5": toggle_active,
        "f6": lambda: adjust_latency(-10),   # later
        "f7": lambda: adjust_latency(+10),   # earlier
        "f3": lambda: adjust_latency(-3),
        "f4": lambda: adjust_latency(+3),
        "f8": print_status,
        "f10": reset_stats,
        "f9": _shutdown.set,
    }
    # Bare F-keys collide with other tools' hotkeys, which silently walks
    # latency_ms around mid-session. A modifier prefix makes every binding
    # deliberate. Set "hotkey_prefix": "" to go back to bare F-keys.
    prefix = str(config.get("hotkey_prefix", "ctrl+alt+"))
    for hk, fn in controls.items():
        combo = prefix + hk
        try:
            keyboard.add_hotkey(combo, fn)
        except Exception:
            log.warning("[hotkey] could not register '%s'", combo)
    log.info("[hotkey] tuning keys bound as %sF3/F4/F6/F7 "
             "(F5 pause, F8 status, F9 quit)", prefix or "")

    print_status()
    log.info("[start] K-Meter v%s ready — firing %d ms before the shape "
             "fills. Tap '%s' to shoot.",
             __version__, config["latency_ms"], config["action_key"])

    try:
        _shutdown.wait()
    except KeyboardInterrupt:
        _shutdown.set()

    log.info("[exit] stopping")
    try:
        keyboard.unhook_all()
    except Exception:
        pass
    if _physically_held or _cycle_active:
        _inject_release()      # never leave K stuck down


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
