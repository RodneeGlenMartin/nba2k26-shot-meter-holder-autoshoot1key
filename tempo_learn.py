"""TempoLearn — watch a gameplay recording and learn your shot tempo.

Scans an OBS recording of NBA 2K26 with Windows OCR, finds every shot
feedback banner (Rushed / Slightly Rushed / Great Tempo / Slow ... and
Early / Late / Great Timing), groups them into shot events, and produces:

  * a per-shot log (video time, wall time, tempo verdict, timing verdict)
  * verdict distributions and a time trend (fatigue drift shows up here)
  * a learned recommendation: how far and in which direction to move
    `hold_ms` in precision_timer_pad.json

If pad_log.txt (written by precision_timer_pad.py) covers the recording's
time span, each shot is also matched to the exact press and hold value
that produced it, giving per-hold accuracy tables — the strongest signal.

Usage:
    python tempo_learn.py "D:\\path\\to\\recording.mp4" [--stride 20]

The wall-clock start time is parsed from OBS-style filenames
("YYYY-MM-DD HH-MM-SS.mp4"); pass --start "HH:MM:SS" to override.

Requires: opencv-python, winrt-* (Windows OCR). Windows 10/11 only.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import os
import re
import sys
import time

import cv2

from winrt.windows.graphics.imaging import SoftwareBitmap, BitmapPixelFormat
from winrt.windows.media.ocr import OcrEngine
from winrt.windows.security.cryptography import CryptographicBuffer

# --------------------------------------------------------------------------- #
# Capture layout (1920x1080 desktop recording, game windowed)
# --------------------------------------------------------------------------- #
GAME_REGION = (200, 100, 1700, 900)   # x0, y0, x1, y1 — excludes HUD/ticker

EVENT_GAP_S = 2.5      # banner samples farther apart than this = new shot
BANNER_MIN_HITS = 2    # require at least N samples to accept a shot event

TEMPO_VALUE = {        # ms correction implied by each tempo verdict
    "Rushed": +20, "Slightly Rushed": +8,
    "Great": 0,
    "Slightly Slow": -8, "Slow": -20,
}
TIMING_VALUE = {       # informational; same knob (hold) drives it
    "Early": +20, "Slightly Early": +8,
    "Great": 0,
    "Slightly Late": -8, "Late": -20,
}


PHRASES = {
    "RUSHED": ("tempo", "Rushed"),
    "SLIGHTLYRUSHED": ("tempo", "Slightly Rushed"),
    "GREATTEMPO": ("tempo", "Great"),
    "SLIGHTLYSLOW": ("tempo", "Slightly Slow"),
    "SLOW": ("tempo", "Slow"),
    "EARLY": ("timing", "Early"),
    "SLIGHTLYEARLY": ("timing", "Slightly Early"),
    "GREATTIMING": ("timing", "Great"),
    "SLIGHTLYLATE": ("timing", "Slightly Late"),
    "LATE": ("timing", "Late"),
}


def _edit(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def classify(text: str):
    """Map one OCR line to ('tempo'|'timing', verdict) or None.
    Strict: the line must BE one of the verdict phrases (allowing OCR
    garble via small edit distance, or a clean truncation) — substring
    keywords alone false-positive on HUD text like 'Offensive Artist'."""
    if any(ch.isdigit() for ch in text) or "[" in text or "]" in text:
        return None
    s = re.sub(r"[^A-Z]", "", text.upper())
    if not 4 <= len(s) <= 20:
        return None
    best = None
    for phrase, res in PHRASES.items():
        if len(s) >= 6 and phrase.startswith(s):
            d = 0                      # clean truncation, e.g. GREATTIM
        else:
            cap = 1 if len(phrase) <= 6 else 2
            d = _edit(s, phrase)
            if d > cap:
                continue
        if best is None or d < best[0]:
            best = (d, res)
    return best[1] if best else None


def to_sb(img):
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    h, w = bgra.shape[:2]
    ibuf = CryptographicBuffer.create_from_byte_array(bgra.tobytes())
    return SoftwareBitmap.create_copy_from_buffer(
        ibuf, BitmapPixelFormat.BGRA8, w, h)


# --------------------------------------------------------------------------- #
# pad_log.txt correlation
# --------------------------------------------------------------------------- #
_LOG_RE = re.compile(
    r"^(?:(\d{4}-\d{2}-\d{2}) )?(\d{2}:\d{2}:\d{2})\.(\d{3})\s+"
    r"\[trigger\] press -> gather (\d+)")


def load_presses(log_path: str, day: dt.date):
    presses = []
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = _LOG_RE.match(line.strip())
                if not m:
                    continue
                d = (dt.date.fromisoformat(m.group(1))
                     if m.group(1) else day)
                t = dt.time.fromisoformat(m.group(2))
                when = dt.datetime.combine(d, t) + dt.timedelta(
                    milliseconds=int(m.group(3)))
                presses.append((when, int(m.group(4))))
    except OSError:
        pass
    return presses


def match_press(presses, banner_wall: dt.datetime):
    """Find the press that produced a banner first seen at banner_wall.
    The press precedes the banner by roughly hold+flight (~0.5-4 s)."""
    best = None
    for when, hold in presses:
        lead = (banner_wall - when).total_seconds()
        if 0.2 <= lead <= 4.5:
            if best is None or lead < best[0]:
                best = (lead, when, hold)
    return best


# --------------------------------------------------------------------------- #
# Main sweep
# --------------------------------------------------------------------------- #
async def sweep(video: str, stride: int, out_dir: str):
    engine = OcrEngine.try_create_from_user_profile_languages()
    if engine is None:
        sys.exit("No Windows OCR engine available")
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"Cannot open {video}")
    total_s = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(
        cap.get(cv2.CAP_PROP_FPS), 1)
    x0, y0, x1, y1 = GAME_REGION
    raw = []          # (t, kind, verdict)
    fidx = 0
    t = 0.0
    wall = time.time()
    last_prog = 0.0
    while True:
        if not cap.grab():
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        fidx += 1
        if fidx % stride:
            continue
        ok, fr = cap.retrieve()
        if not ok:
            continue
        res = await engine.recognize_async(to_sb(fr[y0:y1, x0:x1]))
        for line in res.lines:
            c = classify(line.text)
            if c:
                raw.append((t, c[0], c[1]))
        if t - last_prog >= 60:
            last_prog = t
            el = time.time() - wall
            print(f"  ... {t:6.0f}/{total_s:.0f}s scanned "
                  f"({el:.0f}s elapsed, {len(raw)} raw hits)", flush=True)
    cap.release()
    print(f"sweep done: {len(raw)} raw verdict lines "
          f"in {time.time()-wall:.0f}s", flush=True)
    return raw


def group_events(raw):
    """Cluster raw (t, kind, verdict) samples into shot events."""
    events = []
    cur = None
    for t, kind, verdict in sorted(raw):
        if cur is None or t - cur["last"] > EVENT_GAP_S:
            cur = {"start": t, "last": t, "tempo": {}, "timing": {}, "n": 0}
            events.append(cur)
        cur["last"] = t
        cur["n"] += 1
        cur[kind][verdict] = cur[kind].get(verdict, 0) + 1
    out = []
    for e in events:
        if e["n"] < BANNER_MIN_HITS:
            continue
        tempo = max(e["tempo"], key=e["tempo"].get) if e["tempo"] else None
        timing = max(e["timing"], key=e["timing"].get) if e["timing"] else None
        out.append({"t": e["start"], "tempo": tempo, "timing": timing})
    return out


def fmt_dist(counter, order):
    total = sum(counter.values()) or 1
    return "\n".join(
        f"      {k:16s} {counter.get(k, 0):3d}  "
        f"({100.0 * counter.get(k, 0) / total:4.0f}%)"
        for k in order if counter.get(k, 0) or k == "Great")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--stride", type=int, default=20,
                    help="analyze every Nth frame (20 = 3 Hz at 60fps)")
    ap.add_argument("--start", default=None,
                    help="wall-clock start HH:MM:SS (default: from filename)")
    ap.add_argument("--log", default=None,
                    help="pad_log.txt path (default: next to this script)")
    args = ap.parse_args()

    base = os.path.splitext(os.path.basename(args.video))[0]
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "tempo_learn_out")
    os.makedirs(out_dir, exist_ok=True)

    # Wall-clock anchor from OBS filename "YYYY-MM-DD HH-MM-SS"
    m = re.match(r"(\d{4}-\d{2}-\d{2})[ _](\d{2})-(\d{2})-(\d{2})", base)
    start_wall = None
    if args.start:
        d = dt.date.fromisoformat(m.group(1)) if m else dt.date.today()
        start_wall = dt.datetime.combine(
            d, dt.time.fromisoformat(args.start))
    elif m:
        start_wall = dt.datetime.fromisoformat(
            f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}")

    raw = asyncio.run(sweep(args.video, args.stride, out_dir))
    events = group_events(raw)
    print(f"{len(events)} shot events detected")

    presses = []
    if start_wall:
        log_path = args.log or os.path.join(here, "pad_log.txt")
        presses = load_presses(log_path, start_wall.date())
        if presses:
            print(f"{len(presses)} presses loaded from {log_path}")

    # Attach wall time + press/hold where possible
    tempo_counts, timing_counts = {}, {}
    by_hold = {}
    rows = []
    for ev in events:
        wall_t = (start_wall + dt.timedelta(seconds=ev["t"])
                  if start_wall else None)
        hold = None
        if wall_t and presses:
            mm = match_press(presses, wall_t)
            if mm:
                hold = mm[2]
        if ev["tempo"]:
            tempo_counts[ev["tempo"]] = tempo_counts.get(ev["tempo"], 0) + 1
            if hold is not None:
                by_hold.setdefault(hold, {})[ev["tempo"]] = \
                    by_hold.setdefault(hold, {}).get(ev["tempo"], 0) + 1
        if ev["timing"]:
            timing_counts[ev["timing"]] = \
                timing_counts.get(ev["timing"], 0) + 1
        rows.append({
            "video_s": f"{ev['t']:.1f}",
            "wall": wall_t.strftime("%H:%M:%S") if wall_t else "",
            "tempo": ev["tempo"] or "",
            "timing": ev["timing"] or "",
            "hold_ms": hold if hold is not None else "",
        })

    csv_path = os.path.join(out_dir, f"{base}.shots.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows
                           else ["video_s"])
        w.writeheader()
        w.writerows(rows)

    # ---- report ---------------------------------------------------------- #
    rep = []
    rep.append("=" * 62)
    rep.append(f"  TempoLearn report — {base}")
    rep.append("=" * 62)
    rep.append(f"  Shots detected : {len(events)}")
    rep.append("  Tempo verdicts:")
    rep.append(fmt_dist(tempo_counts,
                        ["Rushed", "Slightly Rushed", "Great",
                         "Slightly Slow", "Slow"]))
    rep.append("  Timing verdicts:")
    rep.append(fmt_dist(timing_counts,
                        ["Early", "Slightly Early", "Great",
                         "Slightly Late", "Late"]))

    n_tempo = sum(tempo_counts.values())
    if n_tempo:
        delta = sum(TEMPO_VALUE[k] * v for k, v in tempo_counts.items()
                    ) / n_tempo
        great = tempo_counts.get("Great", 0)
        rep.append("-" * 62)
        rep.append(f"  Great-tempo rate : {100.0*great/n_tempo:.0f}%")
        rep.append(f"  Learned correction: {delta:+.1f} ms "
                   f"({'raise' if delta > 0 else 'lower'} hold_ms)")
        rep.append("  (positive = shots graded rushed on average -> "
                   "hold longer)")

    # trend over time (fatigue drift)
    if events:
        rep.append("-" * 62)
        rep.append("  Trend (5-min buckets, avg implied correction):")
        buckets = {}
        for ev in events:
            if not ev["tempo"]:
                continue
            b = int(ev["t"] // 300)
            buckets.setdefault(b, []).append(TEMPO_VALUE[ev["tempo"]])
        for b in sorted(buckets):
            vals = buckets[b]
            bias = sum(vals) / len(vals)
            bar = "#" * min(20, int(abs(bias) / 2 + 0.5))
            side = "rushed" if bias > 0 else ("slow" if bias < 0 else "even")
            rep.append(f"    {b*5:3d}-{b*5+5:<3d} min  {len(vals):3d} shots"
                       f"  {bias:+5.1f} ms  {side:6s} {bar}")

    if by_hold:
        rep.append("-" * 62)
        rep.append("  Per-hold results (from pad_log.txt):")
        for hold in sorted(by_hold):
            c = by_hold[hold]
            n = sum(c.values())
            g = c.get("Great", 0)
            det = ", ".join(f"{k}:{v}" for k, v in sorted(c.items()))
            rep.append(f"    hold {hold:4d} ms  {n:3d} shots  "
                       f"Great {100.0*g/n:3.0f}%   [{det}]")
        best = max(by_hold, key=lambda h: (
            by_hold[h].get("Great", 0) / sum(by_hold[h].values()),
            sum(by_hold[h].values())))
        rep.append(f"  Best observed hold: {best} ms")

    rep.append("-" * 62)
    rep.append(f"  Shot list: {csv_path}")
    rep.append("=" * 62)
    text = "\n".join(rep)
    print(text)
    with open(os.path.join(out_dir, f"{base}.report.txt"), "w",
              encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
