"""k2_runtime — the live half of k2. Import-free of the old tool.

Split from k2.py so the timing core stays testable with no game, no screen
and no Windows-only imports in the way. k2.py owns StaircaseClock, meter
detection and the self-test/offline evaluation; this owns capture, tracking
and the key hook. Detection living in k2.py is deliberate: the offline
evaluation exercises the exact code that fires shots.

Shot flow, one thread, no worker:

    K down (game focused)  ->  arm
      |
      +-- hunt for the magenta bar, up to no_meter_ms
      |     not found  ->  blind release at press + hold_ms
      |
      +-- found: target = tick_span * arrow_ratio
            feed every capture into StaircaseClock
            once the clock is fitted, ask it which rendered frame the fill
            reaches target on, subtract latency_ms, wait, release.

The clock is re-fitted on every step, so the release time is refined right
up to the moment it fires — no commit-and-hope from a single early frame.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import os
import threading
import time

import cv2
import keyboard
import numpy as np
from mss import MSS

from k2 import (CFG_PATH, INPUT, KEYBDINPUT, KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE,
                INPUT_KEYBOARD, StaircaseClock, Waiter, cfg, clock_prior,
                find_bar, find_tick, game_focused, game_rect, load_cfg, log,
                save_cfg, update_clock_prior, _u32)

_injecting = False
_held = False
_active = False
_lock = threading.Lock()
_press_t = 0.0
_armed = threading.Event()
_stop = threading.Event()

_rel = INPUT()
_rel.type = INPUT_KEYBOARD
_rel.u.ki = KEYBDINPUT(0, keyboard.key_to_scan_codes(cfg["action_key"])[0],
                       KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, 0, 0)


def inject_release() -> None:
    global _injecting
    _injecting = True
    try:
        if _u32.SendInput(1, ctypes.byref(_rel), ctypes.sizeof(INPUT)) != 1:
            keyboard.release(cfg["action_key"])
    finally:
        _injecting = False


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #
class Screen:
    def __init__(self) -> None:
        self.cam = None
        if not cfg.get("force_mss"):
            try:
                import dxcam
                self.cam = dxcam.create(output_color="BGR")
                log.info("[capture] dxcam (GPU)")
            except Exception:
                self.cam = None
        if self.cam is None:
            log.info("[capture] mss (CPU)")
        self.sct = MSS()

    def hsv(self, x, y, w, h):
        if self.cam is not None:
            f = self.cam.grab(region=(int(x), int(y), int(x + w), int(y + h)))
            if f is not None:
                return cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        raw = self.sct.grab({"left": int(x), "top": int(y),
                             "width": int(w), "height": int(h)})
        img = np.frombuffer(raw.rgb, np.uint8).reshape(raw.height, raw.width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def settle(scr: Screen, crop, t_rel: float) -> None:
    """Watch the meter AFTER the release and log where it comes to rest.

    2K's meter does not stop where you released. It runs on to the tip,
    collides, then the fill RECEDES to a mark showing the true release
    point. That means a screenshot only carries timing information once
    the recoil has finished — comparing an early frame against a settled
    one is comparing two different things, which is exactly how a shot
    measured at 0.985 of span got graded later than one at 0.992.

    Purely diagnostic. Nothing here feeds the release decision; it exists
    to find out whether the settled ratio tracks the badge. If it does,
    it is a continuous per-shot error signal to replace 3-bucket badge
    reports. If it does not, this gets deleted rather than trusted.

    Runs after inject_release(), so it costs the timing path nothing, and
    bails the moment the next shot arms.
    """
    if crop is None:
        return
    cx, cy, cw, ch = crop
    samples = []
    last, still_since = None, None
    while time.perf_counter() - t_rel < 1.2 and not _armed.is_set():
        hsv = scr.hsv(cx, cy, cw, ch)
        bar = find_bar(hsv)
        if bar is not None:
            y0, y1, left, right = bar
            tick = find_tick(hsv, y0, y1)
            ms = (time.perf_counter() - t_rel) * 1000.0
            samples.append((ms, right - left + 1,
                            (tick - left) if tick else None))
            fill = right - left + 1
            if last is not None and abs(fill - last) <= 1:
                if still_since is None:
                    still_since = ms
                elif ms - still_since >= 180:
                    break
            else:
                still_since = None
            last = fill
        time.sleep(0.004)

    if len(samples) < 4:
        log.info("[settle] no read (%d samples)", len(samples))
        return
    peak = max(s[1] for s in samples)
    ms, fill, span = samples[-1]
    spans = [s[2] for s in samples if s[2]]
    span = spans[-1] if spans else span
    # OVERSHOOT is the number that carries timing. The rest value saturates
    # at the tick and jitters +-1.5% frame to frame (measured on a static,
    # already-settled bar: 0.985 0.985 1.000 0.985 1.000 1.000 0.992), so
    # differences smaller than ~2 px in the rest are noise, not information.
    log.info("[settle] peak %d span %s OVERSHOOT %s | rest %d after %.0f ms, "
             "%d samples", peak, span,
             f"{peak - span:+d}" if span else "?", fill, ms, len(samples))


# --------------------------------------------------------------------------- #
# The shot
# --------------------------------------------------------------------------- #
def track(scr: Screen, waiter: Waiter) -> None:
    press = _press_t
    rect = game_rect()
    if rect is None:
        log.warning("[shot] no game window")
        return
    gx, gy, gw, gh = rect

    deadline = press + cfg["max_hold_ms"] / 1000.0
    give_up = press + cfg["no_meter_ms"] / 1000.0
    blind = press + cfg["hold_ms"] / 1000.0

    clock = StaircaseClock()
    crop = None
    target = None
    seen = 0
    misses = 0

    while not _stop.is_set():
        now = time.perf_counter()
        if now >= deadline:
            inject_release()
            log.warning("[shot] safety release at %d ms",
                        cfg["max_hold_ms"])
            return

        # Hunt on the full frame, TRACK on a tight crop. Full-frame grabs
        # measure 12.1 ms median (p90 27.4) vs 4.7 ms for a crop — slower
        # than the 13.7 ms render frame, which would starve the clock fit of
        # the several samples per frame it needs to see steps cleanly.
        if crop is None:
            hsv = scr.hsv(gx, gy, gw, gh)
            ox = oy = 0
        else:
            cx, cy, cw, ch = crop
            hsv = scr.hsv(cx, cy, cw, ch)
            ox, oy = cx - gx, cy - gy
        bar = find_bar(hsv)
        if bar is not None and crop is not None:
            _y0, _y1, _l, _r = bar
            # Bar touching a crop edge means the camera panned it partly out
            # and the measured length is a lie. Drop back to full-frame.
            if _l <= 1 or _r >= crop[2] - 2:
                crop = None
                continue

        if bar is None:
            if target is None and now >= blind:
                waiter.until(blind, cfg["spin_ms"])
                inject_release()
                log.info("[shot] no meter — blind %d ms", cfg["hold_ms"])
                return
            if target is None and now >= give_up:
                # Keep watching until the blind release is actually due:
                # meters have been seen appearing as late as +596 ms, and
                # giving up early threw those away for nothing.
                time.sleep(0.001)
                continue
            if target is not None and crop is not None:
                # A dead crop must not eat the shot. The edge check below
                # only helps when the bar is SEEN touching the crop edge; a
                # fast pan (or a body walking through) can remove it from
                # the crop entirely between two grabs, and before this
                # counter existed the loop spun on the empty crop until the
                # 1200 ms safety — a guaranteed hard Late. ~8 misses is
                # ~45 ms: long enough to ride out one occluded frame, short
                # enough to re-acquire and keep the fit alive.
                misses += 1
                if misses >= 8:
                    crop = None
                    misses = 0
            time.sleep(0.001)
            continue
        misses = 0

        y0, y1, left, right = bar
        # Fill LENGTH, never absolute position: 2K pans the camera and the
        # meter slides with the shooter. Absolute front mistakes pan for fill.
        fill = float(right - left)
        seen += 1

        if target is None:
            tick = find_tick(hsv, y0, y1, from_x=left)
            if tick is None or tick <= right:
                time.sleep(0.001)
                continue
            span = tick - left
            # Reject pop-in animation artifacts: valid meter span is 128..200 px
            if not (128 <= span <= 200):
                time.sleep(0.001)
                continue
            target = span * float(cfg["arrow_ratio"])
            log.info("[shot] bar at +%.0f ms  span %d  target %.0f",
                     (now - press) * 1000, span, target)

        if crop is None:
            # (Re)establish the tight crop. Runs on lock-on AND after a pan
            # pushed the bar against an edge — otherwise one pan would drop
            # us to 12 ms full-frame grabs for the rest of the shot.
            pad = 120
            crop = (gx + ox + left - pad, gy + oy + y0 - 16,
                    int(target + 2 * pad), int(y1 - y0 + 32))

        clock.add((now - press) * 1000.0, fill)

        # Release decision. Preferred: the fitted clock names the crossing
        # frame. Until the fit exists (meter appeared late, too few steps)
        # a prior-rate projection stands in — same maths the old emergency
        # path used, but recomputed every capture and scheduled through the
        # waiter for a sub-ms release instead of fired bare the instant a
        # threshold flipped, which landed those shots up to a frame late.
        cross = clock.cross_time(target)
        fitted = cross is not None
        if not fitted:
            prior_T, prior_S = clock_prior()
            cross = (now - press) * 1000.0 + ((target - fill) / prior_S) * prior_T
        fire = press + (cross - float(cfg["latency_ms"])) / 1000.0
        lead = (fire - time.perf_counter()) * 1000.0
        if lead <= float(cfg["min_lead_ms"]):
            waiter.until(max(fire, time.perf_counter()), cfg["spin_ms"])
            t = time.perf_counter()
            inject_release()
            if fitted:
                log.info("[shot] RELEASE hold %.1f ms | fill %.0f/%.0f "
                         "T %.2f (raw %.2f) step %.2f (raw %.2f) steps %d "
                         "frames %d cross %.1f lead %.1f",
                         (t - press) * 1000, fill, target,
                         clock.period, clock.period_raw,
                         clock.step_px, clock.step_raw,
                         len(clock.steps), seen, cross, lead)
                # A full staircase is a measurement of this machine's real
                # render clock — feed it back so the next shot's early fit
                # starts from the truth instead of the 75 Hz lab value.
                if len(clock.steps) >= 25:
                    update_clock_prior(clock.period_raw, clock.step_raw)
            else:
                log.info("[shot] RELEASE (prior rate) hold %.1f ms | "
                         "fill %.0f/%.0f cross %.1f lead %.1f",
                         (t - press) * 1000, fill, target, cross, lead)
            try:
                settle(scr, crop, t)
            except Exception:
                log.exception("[settle] failed")   # never costs a shot
            return
        time.sleep(0.001)


def worker() -> None:
    scr = Screen()
    waiter = Waiter()
    while not _stop.is_set():
        if not _armed.wait(0.5):
            continue
        _armed.clear()
        try:
            track(scr, waiter)
        except Exception:
            log.exception("[shot] tracking failed")
            # A crash must cost a normal-length shot, never the 1200 ms
            # safety hold — that reads as a hard Late every time.
            try:
                while time.perf_counter() < _press_t + cfg["hold_ms"] / 1000.0:
                    time.sleep(0.001)
                inject_release()
            except Exception:
                log.exception("[shot] fallback failed")
        finally:
            _clear_cycle()


def _clear_cycle() -> None:
    global _cycle
    with _lock:
        _cycle = False


_cycle = False


def hook(ev) -> bool:
    global _held, _cycle, _press_t
    if _injecting:
        return True
    if ev.event_type == keyboard.KEY_DOWN:
        with _lock:
            if _held:
                return True
            _held = True
            if not _active or not game_focused():
                return True
            if not _cycle:
                _cycle = True
                _press_t = time.perf_counter()
                _armed.set()
                if cfg["debug"]:
                    log.info("[trigger] K down — armed")
        return True
    with _lock:
        _held = False
    return False        # swallow the physical release; we inject our own


def _watch_cfg() -> None:
    """Re-read k2.json when it changes, so tuning needs no restart."""
    last = None
    while not _stop.is_set():
        try:
            m = os.path.getmtime(CFG_PATH)
            if last is not None and m != last:
                before = dict(cfg)
                load_cfg()
                diff = {k: cfg[k] for k in cfg if cfg[k] != before.get(k)}
                if diff:
                    log.info("[cfg] reloaded: %s",
                             ", ".join(f"{k}={v}" for k, v in diff.items()))
            last = m
        except OSError:
            pass
        time.sleep(0.5)


def main() -> None:
    global _active
    load_cfg()
    threading.Thread(target=_watch_cfg, daemon=True).start()
    _active = True
    threading.Thread(target=worker, daemon=True).start()
    keyboard.hook_key(cfg["action_key"], hook, suppress=True)
    log.info("k2 ready — firing %.0f ms before the crossing frame. "
             "Ctrl+Alt+F9 quits.", cfg["latency_ms"])
    keyboard.add_hotkey("ctrl+alt+f9", lambda: _stop.set())
    try:
        while not _stop.is_set():
            time.sleep(0.25)
    finally:
        _stop.set()
        log.info("k2 stopped")


if __name__ == "__main__":
    main()
