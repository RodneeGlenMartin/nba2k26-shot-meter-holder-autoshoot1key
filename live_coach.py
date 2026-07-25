"""LiveCoach — realtime closed-loop tempo tuning for NBA 2K26.

Run this WHILE you play (alongside precision_timer_pad.py). It captures
the game window a few times a second, reads the shot feedback banners
("Rushed", "Slightly Rushed", "Great Tempo", "Slightly Late", ...) with
Windows OCR, detects the green-release paint splash, and after every
shot nudges `hold_ms` in precision_timer_pad.json. The running timer
hot-reloads the file, closing the loop:

    you tap X -> timer shoots -> game grades it -> coach reads the grade
              -> hold_ms nudged -> next shot is closer to Great

Adjustment policy (per shot, tempo verdict first):
    Rushed          +5 ms        Slow            -5 ms
    Slightly Rushed +2 ms        Slightly Slow   -2 ms
    Great            0           (timing verdict used if tempo missing:
                                  Early = hold longer, Late = shorter)

Usage:
    python live_coach.py               # closed loop (adjusts hold_ms)
    python live_coach.py --observe     # watch and report only
    python live_coach.py --window "NBA 2K26"   # custom window title

Stop with Ctrl-C — prints a session summary. Requires: mss, opencv,
winrt-* (Windows OCR), and the game in windowed/borderless mode.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import ctypes.wintypes as wt
import datetime as dt
import json
import os
import sys
import time

import cv2
import numpy as np
from mss import MSS

from winrt.windows.graphics.imaging import SoftwareBitmap, BitmapPixelFormat
from winrt.windows.media.ocr import OcrEngine
from winrt.windows.security.cryptography import CryptographicBuffer

from tempo_learn import classify   # strict verdict-phrase classifier

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "precision_timer_pad.json")

GREEN_LO, GREEN_HI = (48, 85, 110), (75, 255, 255)
SPLASH_PX = 1200          # green px in the banner crop = green release
POLL_HZ = 4.0             # capture+OCR rate
EVENT_GAP_S = 1.4         # banner silent this long = shot finished
STEP = {"Rushed": +5, "Slightly Rushed": +2, "Great": 0,
        "Slightly Slow": -2, "Slow": -5}
STEP_TIMING = {"Early": +5, "Slightly Early": +2, "Great": 0,
               "Slightly Late": -2, "Late": -5}
HOLD_MIN, HOLD_MAX = 300, 1200


def game_rect(title: str):
    """Client-area rect of the game window in screen coordinates."""
    u32 = ctypes.windll.user32
    hwnd = u32.FindWindowW(None, title)
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


def banner_crop(rect):
    """Where feedback banners live: central band of the game window."""
    x, y, w, h = rect
    return {"left": x + int(w * 0.22), "top": y + int(h * 0.22),
            "width": int(w * 0.56), "height": int(h * 0.50)}


def to_sb(img):
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    h, w = bgra.shape[:2]
    ibuf = CryptographicBuffer.create_from_byte_array(bgra.tobytes())
    return SoftwareBitmap.create_copy_from_buffer(
        ibuf, BitmapPixelFormat.BGRA8, w, h)


def load_cfg():
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def save_cfg(cfg):
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=4)
    os.replace(tmp, CONFIG_PATH)


class Coach:
    def __init__(self, adjust: bool):
        self.adjust = adjust
        self.event = None          # active shot being observed
        self.shots = 0
        self.greens = 0
        self.changes = 0
        self.tempo_counts = {}

    def feed(self, now, verdicts, splash):
        if verdicts or (self.event and splash):
            if self.event is None:
                self.event = {"tempo": {}, "timing": {}, "splash": False,
                              "last": now}
            for kind, verdict in verdicts:
                self.event[kind][verdict] = \
                    self.event[kind].get(verdict, 0) + 1
            self.event["splash"] = self.event["splash"] or splash
            if verdicts:
                self.event["last"] = now
        elif self.event and now - self.event["last"] > EVENT_GAP_S:
            self.finish()

    def finish(self):
        e, self.event = self.event, None
        tempo = max(e["tempo"], key=e["tempo"].get) if e["tempo"] else None
        timing = max(e["timing"], key=e["timing"].get) if e["timing"] \
            else None
        if tempo is None and timing is None:
            return
        self.shots += 1
        self.greens += int(e["splash"])
        if tempo:
            self.tempo_counts[tempo] = self.tempo_counts.get(tempo, 0) + 1
        step = STEP.get(tempo) if tempo else STEP_TIMING.get(timing)
        stamp = dt.datetime.now().strftime("%H:%M:%S")
        desc = (f"tempo={tempo or '?'} timing={timing or '?'}"
                f"{'  GREEN' if e['splash'] else ''}")
        if self.adjust and step:
            cfg = load_cfg()
            if cfg:
                old = int(cfg.get("hold_ms", 0))
                new = max(HOLD_MIN, min(HOLD_MAX, old + step))
                if new != old:
                    cfg["hold_ms"] = new
                    save_cfg(cfg)
                    self.changes += 1
                    print(f"[{stamp}] shot #{self.shots}: {desc}  ->  "
                          f"hold {old} -> {new} ms", flush=True)
                    return
        print(f"[{stamp}] shot #{self.shots}: {desc}  (hold unchanged)",
              flush=True)

    def summary(self):
        print("\n" + "=" * 56)
        print(f"  LiveCoach session: {self.shots} shots, "
              f"{self.greens} green, {self.changes} adjustments")
        for k in ("Rushed", "Slightly Rushed", "Great",
                  "Slightly Slow", "Slow"):
            if self.tempo_counts.get(k):
                print(f"    {k:16s} {self.tempo_counts[k]}")
        cfg = load_cfg()
        if cfg:
            print(f"  Final hold_ms: {cfg.get('hold_ms')}")
        print("=" * 56)


async def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--observe", action="store_true",
                    help="report shots but do not change hold_ms")
    ap.add_argument("--window", default="NBA 2K26")
    args = ap.parse_args()

    engine = OcrEngine.try_create_from_user_profile_languages()
    if engine is None:
        sys.exit("No Windows OCR engine available")

    coach = Coach(adjust=not args.observe)
    mode = "OBSERVE (no changes)" if args.observe else "CLOSED LOOP"
    print(f"LiveCoach running — {mode}. Window: '{args.window}'. "
          f"Ctrl-C to stop.", flush=True)

    rect = None
    rect_checked = 0.0
    period = 1.0 / POLL_HZ
    with MSS() as sct:
        try:
            while True:
                loop_t = time.monotonic()
                if rect is None or loop_t - rect_checked > 5.0:
                    rect_checked = loop_t
                    r = game_rect(args.window)
                    if r != rect:
                        rect = r
                        if rect:
                            print(f"[coach] game window {rect}", flush=True)
                        else:
                            print("[coach] game window not found — "
                                  "waiting", flush=True)
                if rect is None:
                    await asyncio.sleep(1.0)
                    continue
                shot = sct.grab(banner_crop(rect))
                img = np.frombuffer(shot.bgra, np.uint8).reshape(
                    shot.height, shot.width, 4)[:, :, :3]
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                splash = int(np.count_nonzero(
                    cv2.inRange(hsv, GREEN_LO, GREEN_HI))) >= SPLASH_PX
                res = await engine.recognize_async(to_sb(img))
                verdicts = []
                for line in res.lines:
                    c = classify(line.text)
                    if c:
                        verdicts.append(c)
                coach.feed(time.monotonic(), verdicts, splash)
                el = time.monotonic() - loop_t
                if el < period:
                    await asyncio.sleep(period - el)
        except KeyboardInterrupt:
            pass
    coach.summary()


if __name__ == "__main__":
    asyncio.run(main())
