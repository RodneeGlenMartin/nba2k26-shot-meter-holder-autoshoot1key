"""MeterWatch — watch full gameplay video, track the 2K26 shot meter.

Processes EVERY frame of the recording (no sampling around known events,
no fixed meter position — the meter follows the player and changes size
with the camera). For each frame it searches for the rhythm-shot meter:
a small saturated GREEN target window attached to a bright low-saturation
arc (white = filled, gray = remaining). A state machine turns the
per-frame detections into shot events:

    t_start   meter appears           (gather begins)
    t_full    gray runs out           (fill front reaches the top)
    splash    big green particle burst = perfect ("green") release
    t_end     meter disappears

Each shot is then joined against pad_log.txt (press time + hold value)
when the log covers the recording, producing per-hold outcome tables and
a learned hold recommendation.

Usage:
    python meter_watch.py "video.mp4" [--start HH:MM:SS] [--out name]

Wall-clock start is parsed from OBS-style filenames. Writes
meter_watch_out/<video>.shots.csv and .report.txt next to this script.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import sys
import time

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Detection parameters
# --------------------------------------------------------------------------- #
GREEN_LO, GREEN_HI = (48, 85, 110), (75, 255, 255)
UI_SAT_MAX = 32          # arc stroke is near-colorless
WHITE_V_MIN = 195        # filled part
GRAY_V_MIN = 105         # unfilled part (up to WHITE_V_MIN)
NEIGH_X, NEIGH_UP, NEIGH_DN = 95, 45, 175   # arc search box around window

MIN_ARC_STROKE = 150     # stroke px (arc components only) to accept a meter
MIN_SHOT_S = 0.30
MAX_SHOT_S = 4.5
MERGE_GAP_S = 0.7        # merge events closer than this (meter fade blinks)
DROPOUT_S = 0.35         # tolerate occlusion/detector misses this long
GRAY_EMPTY_PX = 25       # arc gray below this = fill complete
SPLASH_PX = 1200         # green px around meter = perfect-release splash
IDLE_STRIDE = 3          # every Nth frame while idle; every frame in a shot


def measure_arc(hsv, cx, cy):
    """Count white/gray stroke pixels belonging to arc components near the
    anchor (cx, cy) — component-limited so floor/court pixels don't count.
    Returns (gray, white) or None if no arc-like stroke is present."""
    H, W = hsv.shape[:2]
    xx0, xx1 = max(0, cx - NEIGH_X), min(W, cx + NEIGH_X)
    yy0, yy1 = max(0, cy - NEIGH_UP), min(H, cy + NEIGH_DN)
    nb = hsv[yy0:yy1, xx0:xx1]
    s = nb[:, :, 1]; v = nb[:, :, 2]
    ui = (s < UI_SAT_MAX) & (v >= GRAY_V_MIN)
    stroke = ui.astype(np.uint8)
    ncc, lab, st, ce = cv2.connectedComponentsWithStats(stroke)
    ax, ay = cx - xx0, cy - yy0            # anchor in neighborhood coords
    white = gray = 0
    vmask = v >= WHITE_V_MIN
    for i in range(1, ncc):
        x, y, w, h, a = st[i]
        if a < 40 or a > 7000 or w > 160 or h > 220:
            continue
        # component must lie near the anchor point
        dx = max(x - ax, ax - (x + w), 0)
        dy = max(y - ay, ay - (y + h), 0)
        if dx * dx + dy * dy > 55 * 55:
            continue
        comp = lab == i
        white += int(np.count_nonzero(comp & vmask))
        gray += int(np.count_nonzero(comp & ~vmask))
    if white + gray < MIN_ARC_STROKE:
        return None
    return gray, white


def splash_near(hsv, cx, cy):
    """Green particle burst around the shooter = perfect release."""
    H, W = hsv.shape[:2]
    x0, x1 = max(0, cx - 220), min(W, cx + 220)
    y0, y1 = max(0, cy - 220), min(H, cy + 200)
    g = cv2.inRange(hsv[y0:y1, x0:x1], GREEN_LO, GREEN_HI)
    return int(np.count_nonzero(g))


def find_meter(hsv, prev_xy=None):
    """Locate the meter. Primary: green target window blob with an arc
    stroke beside it. Fallback (mid-shot, window covered by the fill):
    re-measure the arc at the previous location. Returns
    (cx, cy, gray, white) or None."""
    g = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    if prev_xy is not None:
        x0 = max(0, prev_xy[0] - 220); x1 = min(g.shape[1], prev_xy[0] + 220)
        y0 = max(0, prev_xy[1] - 180); y1 = min(g.shape[0], prev_xy[1] + 180)
        sub = np.zeros_like(g); sub[y0:y1, x0:x1] = g[y0:y1, x0:x1]; g = sub
    ncc, lab, st, ce = cv2.connectedComponentsWithStats(g)
    best = None
    H, W = hsv.shape[:2]
    for i in range(1, ncc):
        x, y, w, h, a = st[i]
        if not (8 <= a <= 600 and w <= 55 and h <= 55):
            continue
        cx, cy = int(ce[i][0]), int(ce[i][1])
        if cy < 120 or cy > H - 120 or cx < 250 or cx > W - 250:
            continue
        arc = measure_arc(hsv, cx, cy)
        if arc is None:
            continue
        score = arc[0] + arc[1]
        if best is None or score > best[0]:
            best = (score, cx, cy, arc[0], arc[1])
    if best is not None:
        return best[1:] + (True,)
    if prev_xy is not None:                 # window covered by fill
        arc = measure_arc(hsv, prev_xy[0], prev_xy[1])
        if arc is not None and arc[0] + arc[1] >= 400:
            return prev_xy[0], prev_xy[1], arc[0], arc[1], False
    return None


# --------------------------------------------------------------------------- #
# pad_log correlation
# --------------------------------------------------------------------------- #
_LOG_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})\.(\d{3})\s+"
    r"\[trigger\] press -> gather (\d+)")


def load_presses(path):
    out = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = _LOG_RE.match(line.strip())
                if m:
                    when = dt.datetime.fromisoformat(
                        f"{m.group(1)} {m.group(2)}") + dt.timedelta(
                        milliseconds=int(m.group(3)))
                    out.append((when, int(m.group(4))))
    except OSError:
        pass
    return out


# --------------------------------------------------------------------------- #
# Main watch loop
# --------------------------------------------------------------------------- #
def _close(shots, state):
    dur = state["last_seen"] - state["start"]
    if dur >= MIN_SHOT_S and state["frames"] >= 6 \
            and state["green_frames"] >= 4:
        shots.append({
            "t_start": state["start"],
            "t_full": state["t_full"],
            "t_end": state["last_seen"],
            "splash": state["splash"],
        })


def _open(t, cx, cy, gray):
    return {"start": t, "xy": (cx, cy), "last_seen": t, "last_green": t,
            "t_full": None, "splash": False, "max_gray": gray,
            "frames": 1, "green_frames": 1}


def watch(video, t0=None, t1=None):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"cannot open {video}")
    if t0:
        cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    shots = []
    state = None      # active shot dict
    fidx = 0
    wall = time.time()
    next_prog = 0.0
    while True:
        if state is None and IDLE_STRIDE > 1:
            # cheap skip while idle
            for _ in range(IDLE_STRIDE - 1):
                cap.grab()
                fidx += 1
        ok, fr = cap.read()
        if not ok:
            break
        fidx += 1
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t1 and t > t1:
            break
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        det = find_meter(hsv, state["xy"] if state else None)
        if state is None:
            if det and det[4]:               # only a green anchor opens
                state = _open(t, det[0], det[1], det[2])
        else:
            if det:
                cx, cy, gray, white, greened = det
                # Fallback (no green window) allowed only briefly,
                # mid-gather, while the fill covers the window.
                ok_det = greened or (
                    t - state["start"] <= 1.5
                    and t - state["last_green"] <= 0.6)
                if ok_det:
                    # A fresh gray refill after fill-complete, or a big
                    # position jump, is the NEXT shot starting.
                    jumped = greened and (
                        abs(cx - state["xy"][0]) > 150
                        or abs(cy - state["xy"][1]) > 150)
                    refill = (state["t_full"] is not None
                              and gray >= 80
                              and t - state["t_full"] > 0.4)
                    if jumped or refill:
                        _close(shots, state)
                        state = _open(t, cx, cy, gray)
                    else:
                        state["xy"] = (cx, cy)
                        state["last_seen"] = t
                        state["frames"] += 1
                        if greened:
                            state["last_green"] = t
                            state["green_frames"] += 1
                        state["max_gray"] = max(state["max_gray"], gray)
                        if (state["t_full"] is None
                                and gray <= GRAY_EMPTY_PX
                                and state["max_gray"] > 80):
                            state["t_full"] = t
                        if (not state["splash"]
                                and splash_near(hsv, cx, cy) >= SPLASH_PX):
                            state["splash"] = True
            gone = t - state["last_seen"] > DROPOUT_S
            too_long = t - state["start"] > MAX_SHOT_S
            if gone or too_long:
                _close(shots, state)
                state = None
        if t >= next_prog:
            next_prog = t + 120
            el = time.time() - wall
            print(f"  ... {t:6.0f}s  ({fidx}/{total} frames, "
                  f"{len(shots)} shots, {el:.0f}s elapsed)", flush=True)
    cap.release()
    # merge events split by the meter's fade blink
    merged = []
    for s in shots:
        if merged and s["t_start"] - merged[-1]["t_end"] < MERGE_GAP_S:
            m = merged[-1]
            m["t_end"] = s["t_end"]
            m["splash"] = m["splash"] or s["splash"]
            if m["t_full"] is None:
                m["t_full"] = s["t_full"]
        else:
            merged.append(dict(s))
    print(f"watch done: {len(merged)} shots "
          f"({len(shots)} raw), {time.time()-wall:.0f}s", flush=True)
    return merged


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", default=None)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    base = os.path.splitext(os.path.basename(args.video))[0]
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "meter_watch_out")
    os.makedirs(out_dir, exist_ok=True)

    m = re.match(r"(\d{4}-\d{2}-\d{2})[ _](\d{2})-(\d{2})-(\d{2})", base)
    start_wall = None
    if args.start:
        d = dt.date.fromisoformat(m.group(1)) if m else dt.date.today()
        start_wall = dt.datetime.combine(d, dt.time.fromisoformat(args.start))
    elif m:
        start_wall = dt.datetime.fromisoformat(
            f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}")

    presses = load_presses(args.log or os.path.join(here, "pad_log.txt"))

    shots = watch(args.video)

    rows = []
    by_hold = {}
    for s in shots:
        wall_t = (start_wall + dt.timedelta(seconds=s["t_start"])
                  if start_wall else None)
        hold = None
        if wall_t and presses:
            best = None
            for when, h in presses:
                lead = (wall_t - when).total_seconds()
                if -0.5 <= lead <= 1.0:      # meter appears ~with the press
                    if best is None or abs(lead) < abs(best[0]):
                        best = (lead, h)
            if best:
                hold = best[1]
        fill_ms = (None if s["t_full"] is None
                   else round(1000 * (s["t_full"] - s["t_start"])))
        rows.append({
            "video_s": f"{s['t_start']:.2f}",
            "wall": wall_t.strftime("%H:%M:%S") if wall_t else "",
            "dur_ms": round(1000 * (s["t_end"] - s["t_start"])),
            "fill_ms": fill_ms if fill_ms is not None else "",
            "green_splash": int(s["splash"]),
            "hold_ms": hold if hold is not None else "",
        })
        if hold is not None:
            e = by_hold.setdefault(hold, {"n": 0, "green": 0, "fills": []})
            e["n"] += 1
            e["green"] += int(s["splash"])
            if fill_ms:
                e["fills"].append(fill_ms)

    csv_path = os.path.join(out_dir, f"{base}.shots.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows
                           else ["video_s"])
        w.writeheader()
        w.writerows(rows)

    rep = ["=" * 62,
           f"  MeterWatch report — {base}",
           "=" * 62,
           f"  Shots (meter events) : {len(shots)}",
           f"  Green-splash makes   : {sum(r['green_splash'] for r in rows)}"]
    fills = [r["fill_ms"] for r in rows if r["fill_ms"] != ""]
    if fills:
        fills.sort()
        med = fills[len(fills) // 2]
        rep.append(f"  Meter fill time      : median {med} ms "
                   f"(p25 {fills[len(fills)//4]}, "
                   f"p75 {fills[3*len(fills)//4]})")
    if by_hold:
        rep.append("-" * 62)
        rep.append("  Per-hold outcomes (joined via pad_log.txt):")
        for hold in sorted(by_hold):
            e = by_hold[hold]
            gr = 100.0 * e["green"] / e["n"]
            fm = (sorted(e["fills"])[len(e["fills"]) // 2]
                  if e["fills"] else "-")
            rep.append(f"    hold {hold:4d} ms  {e['n']:3d} shots  "
                       f"green {gr:3.0f}%   median fill {fm} ms")
        best = max(by_hold,
                   key=lambda h: (by_hold[h]["green"] / by_hold[h]["n"],
                                  by_hold[h]["n"]))
        rep.append(f"  Best observed hold: {best} ms "
                   f"({100.0*by_hold[best]['green']/by_hold[best]['n']:.0f}%"
                   f" green on {by_hold[best]['n']} shots)")
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
