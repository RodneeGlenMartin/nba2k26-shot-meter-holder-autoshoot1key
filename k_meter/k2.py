"""k2 — NBA 2K26 shot release, phase-locked to the game's render clock.

WHY THIS EXISTS
    The previous tool estimated a fill *velocity* and extrapolated a smooth
    line to work out when the meter would reach the target. But the meter is
    not smooth: 2K redraws it once per rendered frame, so the fill is a
    STAIRCASE. Fitting a continuous line through a staircase and
    extrapolating scatters the release by ~4 ms (measured, sd 4.31 ms) —
    against a Hall of Fame timing window of roughly 3 ms. That is why no
    amount of latency tuning ever produced consistent Excellents: the
    estimator was noisier than the target.

THE METHOD HERE
    The fill crosses the target on ONE specific rendered frame, not at some
    continuous instant. So instead of estimating a speed, this fits the
    staircase itself:

        observe step transitions  ->  least-squares fit  t_k = t0 + k * T
        step size                 ->  median fill increment per step
        crossing step             ->  K = ceil(target / step_px)
        release                   ->  t0 + K*T  minus the pipeline latency

    Every rendered frame is a sample of the same clock, so ~20 steps pin the
    period and phase far tighter than two endpoints pin a velocity.
    Simulated against the same capture jitter: sd 1.25 ms vs 4.31 ms, and
    shots landing >3 ms off drop from 55% to 1%.

    Run `python k2.py --selftest` to re-run that comparison.
    Run `python k2.py <video|image|folder> [--dump DIR]` for offline
    evaluation against recordings or Game Bar screenshots: it runs the same
    detection as the live loop and reports bar/tick/tip geometry per frame.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import statistics
import sys
import time

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DEFAULTS = {
    "action_key": "k",
    "game_window": "NBA 2K26",
    "latency_ms": 66.0,      # MAIN KNOB: pipeline lag to fire ahead of the crossing
    "arrow_ratio": 1.022,    # release target = green-window RIGHT-EDGE span * this
                             # (corpus-measured arrow tip / right edge; the old
                             #  1.045 belonged to the centre-of-green basis)
    "hold_ms": 650,          # blind hold when no meter is ever found
    "no_meter_ms": 550,      # give up looking (but keep watching until hold fires)
    "max_hold_ms": 1200,     # hard safety
    "snap_to_frame": False,  # release on the rendered frame vs the true instant
    "min_steps": 6,          # steps needed before the clock fit is trusted
    "min_lead_ms": 6,        # never schedule closer than this; fire now instead
    "spin_ms": 3.0,          # busy-wait window for sub-ms release
    "force_mss": False,
    "debug": True,
}

HOT = ("latency_ms", "arrow_ratio", "hold_ms", "no_meter_ms",
       "min_steps", "min_lead_ms", "spin_ms", "debug")

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(HERE, "k2.json")
cfg = dict(DEFAULTS)


def load_cfg() -> None:
    try:
        with open(CFG_PATH, encoding="utf-8") as fh:
            cfg.update({k: v for k, v in json.load(fh).items() if k in DEFAULTS})
    except FileNotFoundError:
        save_cfg()
    except Exception:
        log.exception("[cfg] unreadable, using defaults")


def save_cfg() -> None:
    try:
        with open(CFG_PATH, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
    except OSError:
        log.exception("[cfg] could not save")


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s.%(msecs)03d  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("k2")
try:
    _fh = logging.FileHandler(os.path.join(HERE, "k2_log.txt"), encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d  %(message)s",
                                       datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(_fh)
except OSError:
    pass


# --------------------------------------------------------------------------- #
# Vision: meter detection. Lives here (not in the runtime) so the offline
# evaluation and the live loop run EXACTLY the same code — a detection bug
# found on a screenshot is a detection bug fixed in the game.
# --------------------------------------------------------------------------- #
try:
    import cv2
    import numpy as np
except ImportError:                      # the timing selftest needs neither
    cv2 = None
    np = None

if np is not None:
    # Magenta fill, measured from real 2K26 frames (H 146-157).
    MAG_LO = np.array((146, 120, 150), np.uint8)
    MAG_HI = np.array((157, 255, 255), np.uint8)
    # Green tick sits at the bar's tip; only used to size the bar.
    TICK_LO = np.array((45, 60, 90), np.uint8)
    TICK_HI = np.array((80, 255, 255), np.uint8)
else:
    MAG_LO = MAG_HI = TICK_LO = TICK_HI = None


def find_bar(hsv):
    """Locate the magenta fill. Returns (y0, y1, left, right) or None."""
    m = cv2.inRange(hsv, MAG_LO, MAG_HI)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None

    valid = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 15 or not (6 <= h <= 45) or not (3 <= w <= 220):
            continue
        valid.append((x, y, w, h, a))

    if not valid:
        return None

    # Group components sharing the same horizontal row band (+/- 8 px)
    # to reconstruct fill length when legs/shoes split the bar
    best_group = None
    best_area = 0
    for x, y, w, h, a in valid:
        y_ctr = y + h / 2.0
        group = [c for c in valid if abs((c[1] + c[3] / 2.0) - y_ctr) <= 8.0]
        total_a = sum(c[4] for c in group)
        if total_a > best_area:
            best_area = total_a
            best_group = group

    if not best_group or best_area < 40:
        return None

    min_x = min(c[0] for c in best_group)
    max_x = max(c[0] + c[2] for c in best_group)
    min_y = min(c[1] for c in best_group)
    max_y = max(c[1] + c[3] for c in best_group)
    return min_y - 6, max_y + 6, min_x, max_x


def find_tick(hsv, y0, y1, from_x=None):
    """RIGHT EDGE of the green window within a horizontal band, or None.

    The right edge, not the centre. The green window's WIDTH varies per
    shot (a wide-open look draws ~27 px of green, a contested one a 4 px
    sliver), so its centre wanders ~11 px between shots — measured on the
    Captures corpus, where centre-based spans read 122 on a wide window
    and 133 on a narrow one, i.e. a ~40 ms target error on wide windows.
    The window always ENDS at the arrow, so its right edge is the stable
    ruler for sizing the bar.

    Two more hazards, both seen in the corpus:
      * a white court line can cross the window diagonally and SPLIT it
        into components — hence fragments near the seed are merged;
      * green-painted courts (Celtics) put arbitrary large green regions
        in the band — hence the blob-shape filter and the width refusal.
    """
    if from_x is not None:
        x0 = max(0, int(from_x))
        x1 = min(hsv.shape[1], int(from_x + 250))
        band = hsv[max(0, y0):max(1, y1), x0:x1]
    else:
        band = hsv[max(0, y0):max(1, y1)]
        x0 = 0
    m = cv2.inRange(band, TICK_LO, TICK_HI)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    cands = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if h >= 4 and a >= 8 and w <= 40:
            cands.append((x, w, a))
    if not cands:
        return None
    sx, sw, _ = max(cands, key=lambda c: c[2])
    lo, hi = sx, sx + sw
    for x, w, _ in cands:
        if x <= hi + 12 and x + w >= lo - 12:      # same window, split apart
            lo = min(lo, x)
            hi = max(hi, x + w)
    if hi - lo > 45:                               # green floor, not a tick
        return None
    return x0 + hi


def find_tip(hsv, y0, y1, from_x, limit=18):
    """DIAGNOSTIC ONLY — not used for targeting. This is how arrow_ratio
    was measured, and how to re-measure it if 2K changes the meter art;
    the offline evaluation reports it for every settled screenshot.

    Rightmost edge of the BAR — the point of the arrow.

    The green chevron is only a landmark; the bar continues past it to a
    dark outline, and that outline is what "fill the arrow" means. Measured
    on a real frame: green centre 989, bar tip 995 — six pixels, ~22 ms of
    release, and the difference between a full arrow and a Slightly Early.

    Scans the whole bar BAND, not one row: the arrow converges to a point,
    so its apex is a single pixel at the vertical centre and a one-row scan
    misses it whenever the band midpoint is off by one.
    """
    lo = max(0, y0)
    hi = min(hsv.shape[0], y1 + 1)
    x1 = min(hsv.shape[1], from_x + limit)
    if lo >= hi or from_x >= x1:
        return None
    dark = hsv[lo:hi, max(0, from_x):x1, 2] < 130
    cols = np.nonzero(dark.any(axis=0))[0]
    if not len(cols):
        return None
    tip = int(from_x + cols.max())
    # Sanity bound. Anything dark further out is a player, a court line or a
    # shadow, not the bar. Measured gap green->tip is 6-7 px; a frame with a
    # defender behind the meter offered a "tip" 58 px out (~211 ms of error)
    # before this check existed.
    gap = tip - (from_x - 1)
    return tip if 2 <= gap <= 12 else None


# --------------------------------------------------------------------------- #
# THE CORE: staircase clock
# --------------------------------------------------------------------------- #
# Render-clock priors. These only steady the fit while the staircase is
# short; once steps accumulate, the data outweighs them (count-weighted
# blend below). Every completed shot feeds its RAW fit back in, so after a
# few shots the prior tracks THIS machine and THIS build of the game.
#
# Why not fixed clamps: live logs showed every shot pinned at the clamp
# edges (T 13.34, step 3.60 — exactly the blend of prior and lower clamp),
# i.e. the real clock sat outside the allowed window and every crossing
# estimate carried a systematic error that varied with lock-on time.
PRIOR_PERIOD = 13.70     # ms per rendered frame (75 Hz), cold-start value
PRIOR_STEP = 3.85        # px of fill per rendered frame, cold-start value
PRIOR_WEIGHT = 4.0       # the prior counts as this many observed steps
_learned = {"period": None, "step": None}


def clock_prior() -> tuple[float, float]:
    return (_learned["period"] or PRIOR_PERIOD,
            _learned["step"] or PRIOR_STEP)


def update_clock_prior(period_raw: float, step_raw: float,
                       alpha: float = 0.3) -> None:
    """EMA of raw per-shot fits -> prior for the next shot's early frames."""
    if not (10.5 <= period_raw <= 18.0 and 2.4 <= step_raw <= 6.0):
        return
    for key, v in (("period", period_raw), ("step", step_raw)):
        cur = _learned[key]
        _learned[key] = v if cur is None else (1.0 - alpha) * cur + alpha * v


class StaircaseClock:
    """Fits the game's render clock from observed fill steps.

    Each time the fill jumps, that jump IS a rendered frame. Collecting the
    (time, fill) of each jump and fitting `t = t0 + k*T` recovers the render
    period and phase. Both are then known well enough to name the exact
    future frame on which the fill will reach a target.
    """

    def __init__(self) -> None:
        self.steps: list[tuple[float, float]] = []   # (t, fill) at each jump
        self._last_fill: float | None = None
        self._max_fill: float = 0.0
        self.period = 0.0        # ms per rendered frame (prior-blended)
        self.step_px = 0.0       # fill advance per rendered frame (blended)
        self.period_raw = 0.0    # pure least-squares values, fed back as
        self.step_raw = 0.0      # the learned prior after a good shot
        self.t0 = 0.0            # fitted time of step index 0
        self.base_fill = 0.0
        self.ok = False

    def reset(self) -> None:
        self.steps.clear()
        self._last_fill = None
        self._max_fill = 0.0
        self.period = 0.0
        self.step_px = 0.0
        self.period_raw = 0.0
        self.step_raw = 0.0
        self.t0 = 0.0
        self.base_fill = 0.0
        self.ok = False

    def add(self, t_ms: float, fill: float) -> None:
        """Feed one capture. Only valid rising jumps are kept."""
        if self._last_fill is None:
            self._last_fill = fill
            self._max_fill = fill
            return
        # Ignore recoil noise: if fill dropped below max_fill - 3.0, the shot is recoiling
        if fill < self._max_fill - 3.0:
            return
        if fill > self._last_fill + 0.5 and fill > self._max_fill + 0.2:
            self.steps.append((t_ms, fill))
            self._last_fill = fill
            self._max_fill = fill
            self._fit()

    def _fit(self) -> None:
        n = len(self.steps)
        if n < 4:
            return
        ts = [t for t, _ in self.steps]
        fy = [f for _, f in self.steps]

        span_f = fy[-1] - fy[0]
        if span_f <= 0.5:
            return

        diffs = [fy[i+1] - fy[i] for i in range(n - 1) if fy[i+1] > fy[i] + 0.5]
        if not diffs:
            return

        prior_T, prior_S = clock_prior()

        # Seed the frame-index quantisation. A single seed (span/(n-1))
        # breaks when the capture runs slower than the render clock: two
        # rendered frames merge into one observed jump, the naive seed
        # doubles, every index halves and the fitted period lands at 2T.
        # So try the plausible seeds and keep the one whose fitted clock
        # explains the observed times best.
        med = statistics.median(diffs)
        best = None
        for step0 in {prior_S, med, med / 2.0, span_f / (n - 1)}:
            if not (2.2 <= step0 <= 6.0):
                continue
            idx = [round((f - fy[0]) / step0) for f in fy]
            if len(set(idx)) < 4 or idx[-1] <= 0:
                continue
            if any(b < a for a, b in zip(idx, idx[1:])):
                continue
            m = n
            sx = sum(idx)
            sy = sum(ts)
            sxx = sum(a * a for a in idx)
            sxy = sum(a * b for a, b in zip(idx, ts))
            den = m * sxx - sx * sx
            if den <= 1e-9:
                continue
            period_raw = (m * sxy - sx * sy) / den
            if not (9.0 <= period_raw <= 20.0):
                continue
            t0_raw = (sy - period_raw * sx) / m
            rms = sum((t - (t0_raw + period_raw * k)) ** 2
                      for t, k in zip(ts, idx)) / m
            if best is None or rms < best[0]:
                best = (rms, idx, period_raw, sx, sy, den)
        if best is None:
            return
        _, idx, period_raw, sx, sy, den = best

        m = n
        sfy = sum(fy)
        sxfy = sum(a * b for a, b in zip(idx, fy))
        step_raw = (m * sxfy - sx * sfy) / den
        # Degenerate batch (occlusion garbage, mis-quantised indices):
        # keep the previous good fit rather than poison it.
        if not (10.5 <= period_raw <= 18.0) or not (2.4 <= step_raw <= 6.0):
            return

        # Count-weighted prior: with few steps the prior steadies the fit,
        # with a full staircase the data speaks for itself (m=36 observed
        # steps outweigh the prior 9:1). No hard clamps — see note above.
        w = PRIOR_WEIGHT
        period = (w * prior_T + m * period_raw) / (w + m)
        step = (w * prior_S + m * step_raw) / (w + m)

        mean_k = sx / m
        mean_t = sy / m
        mean_f = sfy / m

        self.period = period
        self.t0 = mean_t - period * mean_k
        self.step_px = step
        self.base_fill = mean_f - step * mean_k

        # Edge-anchor the phase. A step is only ever OBSERVED at the first
        # capture after the render tick, so observation noise is one-sided:
        # the mean-anchored t0 runs late by half the capture interval, and
        # that bias moves whenever capture cadence moves (crop ~5 ms grabs
        # vs full-frame ~12 ms vs mss fallback). Anchoring t0 to the lower
        # envelope of the residuals ties the fit to the tick itself, making
        # the release independent of how fast the screen was sampled.
        # Second-smallest residual, not the min: one glitched early sample
        # must not drag the whole phase down.
        res = sorted(t - (self.t0 + self.period * k)
                     for t, k in zip(ts, idx))
        edge = res[1] if m >= 8 else res[0]
        if -15.0 <= edge < 0.0:
            self.t0 += edge

        self.period_raw = period_raw
        self.step_raw = step_raw
        min_s = int(cfg.get("min_steps", 6))
        self.ok = m >= min_s

    def cross_time(self, target_fill: float) -> float | None:
        """Time (ms since press) at which the fill reaches target."""
        if not self.ok or self.step_px <= 0:
            return None
        frames = (target_fill - self.base_fill) / self.step_px
        if cfg.get("snap_to_frame"):
            frames = float(int(frames) if abs(frames - round(frames)) < 1e-9
                           else int(frames) + 1)
        return self.t0 + frames * self.period


# --------------------------------------------------------------------------- #
# Win32: high-resolution wait + synthetic key release (Guarded for cross-platform)
# --------------------------------------------------------------------------- #
_TIMER_ALL_ACCESS = 0x1F0003
_HIGH_RES = 0x00000002
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
INPUT_KEYBOARD = 1

if sys.platform == "win32":
    import ctypes.wintypes as wt
    try:
        ctypes.WinDLL("winmm").timeBeginPeriod(1)
    except Exception:
        pass

    _ULONG_PTR = wt.WPARAM

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = (("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD),
                    ("time", wt.DWORD), ("dwExtraInfo", _ULONG_PTR))

    class _U(ctypes.Union):
        _fields_ = (("ki", KEYBDINPUT), ("pad", ctypes.c_byte * 24))

    class INPUT(ctypes.Structure):
        _fields_ = (("type", wt.DWORD), ("u", _U))

    try:
        _u32 = ctypes.WinDLL("user32", use_last_error=True)
        _u32.SendInput.argtypes = (wt.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
    except Exception:
        _u32 = None
else:
    wt = None
    INPUT = None
    KEYBDINPUT = None
    _u32 = None


class Waiter:
    def __init__(self) -> None:
        self.h = None
        self.k32 = None
        if sys.platform == "win32" and wt is not None:
            try:
                k32 = ctypes.WinDLL("kernel32", use_last_error=True)
                k32.CreateWaitableTimerExW.restype = wt.HANDLE
                k32.CreateWaitableTimerExW.argtypes = (ctypes.c_void_p, wt.LPCWSTR,
                                                       wt.DWORD, wt.DWORD)
                k32.SetWaitableTimer.argtypes = (wt.HANDLE,
                                                 ctypes.POINTER(wt.LARGE_INTEGER),
                                                 wt.LONG, ctypes.c_void_p,
                                                 ctypes.c_void_p, wt.BOOL)
                h = k32.CreateWaitableTimerExW(None, None, _HIGH_RES,
                                               _TIMER_ALL_ACCESS)
                if h:
                    self.h, self.k32 = h, k32
            except Exception:
                pass

    def until(self, deadline: float, spin_ms: float) -> None:
        coarse = deadline - spin_ms / 1000.0 - time.perf_counter()
        if coarse > 0:
            if self.h and self.k32 and wt is not None:
                due = wt.LARGE_INTEGER(-int(coarse * 10_000_000))
                if self.k32.SetWaitableTimer(self.h, ctypes.byref(due), 0,
                                             None, None, False):
                    self.k32.WaitForSingleObject(self.h,
                                                 wt.DWORD(int(coarse * 1000) + 100))
                else:
                    time.sleep(coarse)
            else:
                time.sleep(coarse)
        while time.perf_counter() < deadline:
            pass


def game_focused() -> bool:
    """K must never arm a shot while you are typing in another window."""
    if sys.platform != "win32" or _u32 is None or wt is None:
        return True
    try:
        fg = _u32.GetForegroundWindow()
        if not fg:
            return True
        n = _u32.GetWindowTextLengthW(fg)
        buf = ctypes.create_unicode_buffer(n + 1)
        _u32.GetWindowTextW(fg, buf, n + 1)
        return cfg["game_window"].lower() in buf.value.lower()
    except Exception:
        return True


def game_rect():
    if sys.platform != "win32" or _u32 is None or wt is None:
        return None
    try:
        hwnd = _u32.FindWindowW(None, cfg["game_window"])
        if not hwnd:
            return None
        r = wt.RECT()
        if not _u32.GetClientRect(hwnd, ctypes.byref(r)):
            return None
        pt = wt.POINT(0, 0)
        _u32.ClientToScreen(hwnd, ctypes.byref(pt))
        w, h = r.right - r.left, r.bottom - r.top
        return (pt.x, pt.y, w, h) if w > 400 and h > 300 else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Self-test: prove the timing core before trusting it in a game
# --------------------------------------------------------------------------- #
def selftest() -> int:
    import random

    TARGET = 132.0
    # (name, render period ms, fill px/frame, capture ms, capture jitter,
    #  sd gate ms)
    # Case 2 places the true clock OUTSIDE the old clamp windows — the
    # regression that live logs exposed (every real shot pinned at the
    # clamp edges). Case 3 simulates capture running slower than the render
    # clock (observed p90 27 ms full-frame grabs), where merged steps used
    # to mis-quantise the frame indices.
    CASES = (
        ("75 Hz, nominal capture", 13.70, 3.85, 5.2, 0.15, 2.0),
        ("81 Hz, off-prior clock", 12.40, 3.40, 5.2, 0.15, 2.0),
        ("75 Hz, slow capture   ", 13.70, 3.85, 18.0, 0.25, 4.0),
    )

    ok = True
    biases = []
    for name, T, STEP, CAP, JIT, sd_gate in CASES:
        rnd = random.Random(11)

        def shot():
            phase = rnd.uniform(0, T)
            step = STEP * rnd.uniform(0.97, 1.03)
            s, t = [], 0.0
            while t < 420:
                s.append((t, float(int(int((t + phase) // T) * step))))
                t += CAP * rnd.uniform(1 - JIT, 1 + JIT)
            return s, (TARGET / step) * T - phase

        def velocity(s, i):
            p = s[:i + 1]
            n = len(p)
            sx = sum(a for a, _ in p); sy = sum(b for _, b in p)
            sxx = sum(a * a for a, _ in p); sxy = sum(a * b for a, b in p)
            den = n * sxx - sx * sx
            if den <= 0:
                return None
            v = (n * sxy - sx * sy) / den
            if v <= 0:
                return None
            t, f = p[-1]
            return t + (TARGET - f) / v

        def staircase(s, i):
            c = StaircaseClock()
            for t, f in s[:i + 1]:
                c.add(t, f)
            return c.cross_time(TARGET)

        ea, eb = [], []
        for _ in range(1500):
            s, truth = shot()
            i = next((j for j, (t, f) in enumerate(s) if f >= TARGET - 60),
                     None)
            if i is None or i < 10:
                continue
            a, b = velocity(s, i), staircase(s, i)
            if a:
                ea.append(a - truth)
            if b:
                eb.append(b - truth)

        print(f"\n  [{name}]  T={T} ms  step={STEP} px  capture~{CAP} ms")
        res = {}
        for lbl, e in (("velocity extrapolation", ea),
                       ("staircase clock       ", eb)):
            mu = statistics.mean(e)
            sd = statistics.pstdev(e)
            bad = 100 * sum(1 for x in e if abs(x - mu) > 3) / len(e)
            print(f"    {lbl:24s} bias {mu:+6.2f} ms  sd {sd:5.2f} ms  "
                  f">3ms off: {bad:5.1f}%")
            res[lbl.strip()] = (mu, sd)
        mu_b, sd_b = res["staircase clock"]
        _, sd_a = res["velocity extrapolation"]
        biases.append(mu_b)
        if sd_b > sd_gate:
            print(f"    FAIL: staircase sd {sd_b:.2f} > gate {sd_gate}")
            ok = False
        if abs(mu_b) > 4.0:
            print(f"    FAIL: staircase bias {mu_b:+.2f} exceeds 4 ms")
            ok = False
        if sd_b >= sd_a:
            print("    FAIL: staircase no better than velocity fit")
            ok = False

    # A constant bias folds into latency_ms and is harmless. A bias that
    # MOVES with capture cadence is an invisible timing shift every time
    # capture degrades — that is what the edge anchor exists to kill.
    spread = max(biases) - min(biases)
    print(f"\n  bias spread across capture regimes: {spread:.2f} ms")
    if spread > 3.0:
        print("  FAIL: bias moves with capture cadence (>3 ms spread)")
        ok = False

    print("\n  SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Offline evaluation: run the LIVE detection over screenshots / recordings.
# This is how detection constants get validated — the Captures folder is a
# free labelled corpus: every Game Bar screenshot of the game either shows
# the meter (detect it, measure it) or does not (anything found is a false
# positive to chase down with --dump).
# --------------------------------------------------------------------------- #
IMG_EXT = (".png", ".jpg", ".jpeg")
VID_EXT = (".mkv", ".mp4", ".avi")


def _imread(path):
    """cv2.imread chokes on non-ASCII Windows paths; decode from bytes."""
    try:
        return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def _imwrite(path, img) -> bool:
    try:
        okf, buf = cv2.imencode(".png", img)
        if okf:
            buf.tofile(path)
            return True
    except Exception:
        pass
    return False


def _probe_frame(hsv):
    """One frame -> everything the meter shows, or None if no bar."""
    bar = find_bar(hsv)
    if bar is None:
        return None
    y0, y1, l, r = bar
    tick = find_tick(hsv, y0, y1, from_x=l)
    tip = find_tip(hsv, y0, y1, from_x=r)
    span = (tick - l) if (tick is not None and tick > l) else None
    # Only a dark edge BEYOND the green window's right edge is the arrow
    # outline; anything at or inside it is shading at the fill front on a
    # non-full bar and would poison the arrow_ratio statistic.
    ratio = ((tip - l) / span) if (tip is not None and span
                                   and tip - l > span) else None
    return {"y0": y0, "y1": y1, "l": l, "r": r, "fill": r - l,
            "tick": tick, "tip": tip, "span": span, "ratio": ratio}


def _dump_probe(img, p, path) -> None:
    """Annotated 4x crop of one detection, for eyeballing."""
    h, w = img.shape[:2]
    right = max(p["r"], p["tick"] or 0, p["tip"] or 0)
    x0 = max(0, p["l"] - 50)
    x1 = min(w, right + 50)
    y0 = max(0, p["y0"] - 25)
    y1 = min(h, p["y1"] + 25)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return
    s = 4
    vis = cv2.resize(crop, (crop.shape[1] * s, crop.shape[0] * s),
                     interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(vis, ((p["l"] - x0) * s, (p["y0"] - y0) * s),
                  ((p["r"] - x0) * s, (p["y1"] - y0) * s), (255, 0, 255), 1)
    for key, col in (("tick", (0, 255, 0)), ("tip", (255, 128, 0))):
        x = p[key]
        if x is not None:
            cv2.line(vis, ((x - x0) * s, 0), ((x - x0) * s, vis.shape[0]),
                     col, 1)
    _imwrite(path, vis)


def _eval_images(paths: list[str], dump_dir: str | None) -> None:
    try:
        # Capture filenames carry arbitrary window-title characters; never
        # let a cp1252 console kill the run over one of them.
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    print(f"\n[eval] {len(paths)} images", flush=True)
    probes = []
    for pth in paths:
        name = os.path.basename(pth)
        img = _imread(pth)
        if img is None:
            print(f"  {name}: unreadable", flush=True)
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        p = _probe_frame(hsv)
        probes.append(p)
        if p is None:
            print(f"  {name} ({img.shape[1]}x{img.shape[0]}): no meter",
                  flush=True)
            continue
        span = f"{p['span']}" if p["span"] else "—"
        tip = f"{p['tip'] - p['l']}" if p["tip"] is not None else "—"
        ratio = f"{p['ratio']:.3f}" if p["ratio"] else "—"
        print(f"  {name} ({img.shape[1]}x{img.shape[0]}): "
              f"fill {p['fill']:3d}px  span {span}  tip {tip}  "
              f"tip/span {ratio}", flush=True)
        if dump_dir:
            stem = os.path.splitext(name)[0]
            _dump_probe(img, p, os.path.join(dump_dir, stem + ".probe.png"))

    det = [p for p in probes if p]
    spans = sorted(p["span"] for p in det if p["span"])
    ratios = sorted(p["ratio"] for p in det if p["ratio"])
    print(f"\n[eval] meter detected in {len(det)}/{len(probes)} readable "
          f"images", flush=True)
    if spans:
        print(f"[eval] span px: median {statistics.median(spans):.0f}  "
              f"range {spans[0]}..{spans[-1]}  n={len(spans)}", flush=True)
    if ratios:
        # tip/span across settled screenshots IS the arrow_ratio the live
        # loop should aim at; compare against cfg before trusting either.
        print(f"[eval] tip/span (arrow_ratio candidates): "
              f"median {statistics.median(ratios):.3f}  "
              f"range {ratios[0]:.3f}..{ratios[-1]:.3f}  n={len(ratios)}  "
              f"(cfg arrow_ratio={cfg['arrow_ratio']})", flush=True)


def _eval_shot(shot_id: int, frames: list, fps: float) -> None:
    clock = StaircaseClock()
    spans = []
    for fno, p in frames:
        if p["span"] and 100 <= p["span"] <= 240:
            spans.append(p["span"])
        clock.add((fno / fps) * 1000.0, float(p["fill"]))

    if not spans:
        print(f"\n  --- Shot #{shot_id} ({len(frames)} frames): no tick, "
              f"skipped ---", flush=True)
        return
    med_span = float(statistics.median(spans))
    target = med_span * cfg["arrow_ratio"]

    print(f"\n  --- Shot #{shot_id} ({len(frames)} frames) ---", flush=True)
    print(f"  Span median: {med_span:.1f} px | Target: {target:.1f} px",
          flush=True)
    if clock.ok:
        cross = clock.cross_time(target)
        print(f"  CLOCK FIT: T={clock.period:.2f} ms (raw {clock.period_raw:.2f})"
              f" | step={clock.step_px:.2f} px (raw {clock.step_raw:.2f})"
              f" | steps={len(clock.steps)}", flush=True)
        if cross:
            print(f"  Target crossing: {cross:.1f} ms after first frame",
                  flush=True)
    else:
        print(f"  CLOCK FIT: FAILED (steps collected: {len(clock.steps)})",
              flush=True)


def _eval_video(path: str) -> None:
    print(f"\n[eval] video {os.path.basename(path)}", flush=True)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    frame_idx = 0
    shot_count = 0
    current = []
    gap = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        p = _probe_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
        if p is not None:
            current.append((frame_idx, p))
            gap = 0
        elif current:
            gap += 1
            if gap > 5:                      # shot over; brief occlusions ok
                if len(current) > 5:
                    shot_count += 1
                    _eval_shot(shot_count, current, fps)
                current = []
                gap = 0
        frame_idx += 1
    if len(current) > 5:
        shot_count += 1
        _eval_shot(shot_count, current, fps)
    cap.release()
    print(f"[eval] finished {os.path.basename(path)}: {shot_count} shots, "
          f"{frame_idx} frames @ {fps:.0f} fps", flush=True)


def evaluate_files(paths: list[str], dump_dir: str | None = None) -> None:
    """Offline analysis of videos (.mkv/.mp4/.avi), images, or folders."""
    if cv2 is None:
        print("[eval] OpenCV (cv2) + numpy required for offline evaluation.")
        return

    imgs, vids = [], []
    for path in paths:
        if not os.path.exists(path):
            print(f"[eval] file/folder not found: {path}")
        elif os.path.isdir(path):
            for f in sorted(os.listdir(path)):
                if f.lower().endswith(IMG_EXT):
                    imgs.append(os.path.join(path, f))
                elif f.lower().endswith(VID_EXT):
                    vids.append(os.path.join(path, f))
        elif path.lower().endswith(IMG_EXT):
            imgs.append(path)
        elif path.lower().endswith(VID_EXT):
            vids.append(path)
        else:
            print(f"[eval] unsupported file type: {path}")

    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
    if imgs:
        _eval_images(imgs, dump_dir)
    for v in vids:
        _eval_video(v)


# --------------------------------------------------------------------------- #
# Main Entry Point
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="k2 — NBA 2K26 release timer. "
                    "No arguments: live mode. With paths: offline evaluation.")
    ap.add_argument("paths", nargs="*",
                    help="videos, images or folders to evaluate offline")
    ap.add_argument("--selftest", action="store_true",
                    help="prove the timing core against simulated capture")
    ap.add_argument("--dump", metavar="DIR",
                    help="with paths: save annotated detection crops to DIR")
    args = ap.parse_args()

    load_cfg()
    if args.selftest:
        sys.exit(selftest())

    if args.paths:
        print(f"k2 — offline evaluation: {args.paths}")
        evaluate_files(args.paths, args.dump)
        return

    try:
        import k2_runtime
        k2_runtime.main()
    except ImportError:
        log.error("k2_runtime module not found. Live tracking unavailable.")
        sys.exit(1)


if __name__ == "__main__":
    main()
