"""vision_probe — player detection for NBA 2K26, offline evaluation.

Detects on-court players with a COCO-pretrained YOLOv4 run through
cv2.dnn (no torch, no onnxruntime — OpenCV 5.0 is enough), then tags each
box with a team guess from the jersey colour.

This is a MEASUREMENT tool, not part of the live loop. It exists to answer
"is perception good enough to act on?" before anything touches the game.
Run it over frames of a gameplay recording and look at the output images.

    python vision_probe.py --video "D:\\path\\clip.mp4" --frames 8

Known limits (measured, see NOTES at the bottom):
  * ~8 of 10 on-court players per frame at conf 0.35
  * crowd/bench behind the baseline produces false positives
  * team ID is unreliable: Pacers gold and the wood court share a hue
  * 458 ms/frame on CPU — offline only, nowhere near live speed
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

# YOLOv4 (COCO) decode constants. The onnx-zoo export applies sigmoid to
# objectness and class scores INSIDE the graph but leaves box xy/wh as raw
# logits — applying sigmoid to the scores again squashes everything to
# ~0.3 and every cell passes threshold. That mistake produced 950 boxes of
# noise before this was pinned down; don't reintroduce it.
ANCHORS = [[[12, 16], [19, 36], [40, 28]],
           [[36, 75], [76, 55], [72, 146]],
           [[142, 110], [192, 243], [459, 401]]]
STRIDES = [8, 16, 32]
XYSCALE = [1.2, 1.1, 1.05]
PERSON = 0                      # COCO class index

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "models", "yolov4.onnx")


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def decode(outs, thr):
    """Raw YOLOv4 heads -> (boxes xywh, scores) for the person class."""
    boxes, scores = [], []
    # heads come back unordered; largest grid is stride 8
    order = sorted(range(len(outs)), key=lambda i: -outs[i].shape[1])
    for k, i in enumerate(order):
        o = outs[i][0]
        g = o.shape[0]
        gy, gx = np.mgrid[0:g, 0:g]
        for a in range(3):
            t = o[:, :, a, :]
            sc = t[..., 4] * t[..., 5 + PERSON]      # already probabilities
            m = sc > thr
            if not m.any():
                continue
            xy = (_sigmoid(t[..., 0:2]) * XYSCALE[k]
                  - 0.5 * (XYSCALE[k] - 1)
                  + np.stack([gx, gy], -1)) * STRIDES[k]
            wh = np.exp(np.clip(t[..., 2:4], -9, 9)) * np.array(ANCHORS[k][a])
            boxes.append(np.concatenate([xy[m] - wh[m] / 2, wh[m]], -1))
            scores.append(sc[m])
    if not boxes:
        return np.empty((0, 4)), np.empty(0)
    return np.concatenate(boxes), np.concatenate(scores)


def team_of(hsv, box, w_img, h_img):
    """Guess the team from the torso patch.

    Deliberately conservative: returns "?" rather than a wrong label.
    Gold jerseys and the hardwood occupy nearly the same hue band, so hue
    alone is useless here — saturation is the only clean separator, and
    even that fails when the box carries a lot of floor.
    """
    x, y, w, h = box.astype(int)
    tx0, tx1 = max(0, x + w // 4), min(w_img, x + 3 * w // 4)
    ty0, ty1 = max(0, y + h // 4), min(h_img, y + h // 2)
    if tx1 <= tx0 or ty1 <= ty0:
        return "?", (0, 0, 0)
    med = np.median(hsv[ty0:ty1, tx0:tx1].reshape(-1, 3), 0).astype(int)
    hh, ss, vv = med
    if 12 <= hh <= 30 and ss > 140 and vv > 110:
        return "GOLD", tuple(med)
    if ss < 60 and vv > 135:
        return "WHITE", tuple(med)
    return "?", tuple(med)


def game_rect(frame):
    """Game content area inside a full-desktop capture.

    The recording is 1920x1080 desktop with NBA 2K26 windowed at 1600x900.
    Live, k_meter finds this via the Win32 window handle; offline we only
    have pixels, so this is the measured offset for that recording.
    """
    h, w = frame.shape[:2]
    if (w, h) == (1920, 1080):
        return 162, 92, 1762, 992
    return 0, 0, w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--out", default="probe_out")
    a = ap.parse_args()

    if not os.path.exists(MODEL):
        sys.exit(f"missing {MODEL} — see README note on fetching yolov4.onnx")
    os.makedirs(a.out, exist_ok=True)
    net = cv2.dnn.readNetFromONNX(MODEL)
    names = net.getUnconnectedOutLayersNames()

    cap = cv2.VideoCapture(a.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    picks = [int(total * (i + 0.5) / a.frames) for i in range(a.frames)]

    tally = []
    for n, fno in enumerate(picks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ok, frame = cap.read()
        if not ok:
            continue
        gx0, gy0, gx1, gy1 = game_rect(frame)
        sub = frame[gy0:gy1, gx0:gx1]
        H, W = sub.shape[:2]
        S = 416

        x = (cv2.resize(cv2.cvtColor(sub, cv2.COLOR_BGR2RGB), (S, S))
             .astype(np.float32) / 255.0)[None]
        net.setInput(x)
        t0 = time.perf_counter()
        outs = net.forward(names)
        ms = (time.perf_counter() - t0) * 1000

        b, s = decode(outs, a.conf)
        if len(b):
            b[:, [0, 2]] *= W / S
            b[:, [1, 3]] *= H / S
            keep = np.array(cv2.dnn.NMSBoxes(
                b.tolist(), s.tolist(), a.conf, 0.45)).ravel()
        else:
            keep = np.array([], dtype=int)

        hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        vis = sub.copy()
        teams = {"GOLD": 0, "WHITE": 0, "?": 0}
        for i in keep:
            lab, _ = team_of(hsv, b[i], W, H)
            teams[lab] += 1
            col = {"GOLD": (0, 215, 255), "WHITE": (255, 255, 255),
                   "?": (120, 120, 120)}[lab]
            x0, y0, w, h = b[i].astype(int)
            cv2.rectangle(vis, (x0, y0), (x0 + w, y0 + h), col, 2)
            cv2.putText(vis, f"{lab} {s[i]:.2f}", (x0, max(12, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        dst = os.path.join(a.out, f"probe_{n:02d}.png")
        cv2.imwrite(dst, vis)
        tally.append((fno, len(keep), ms, dict(teams)))
        print(f"  frame {fno:7d}  {len(keep):2d} people  {ms:5.0f} ms  "
              f"gold {teams['GOLD']} white {teams['WHITE']} "
              f"unknown {teams['?']}  -> {dst}")
    cap.release()

    if tally:
        det = [t[1] for t in tally]
        lat = [t[2] for t in tally]
        print(f"\n  median {sorted(det)[len(det) // 2]} detections/frame, "
              f"{sorted(lat)[len(lat) // 2]:.0f} ms/frame")
        print("  live defense needs <16 ms (60 Hz) — CPU inference is "
              f"{sorted(lat)[len(lat) // 2] / 16:.0f}x too slow")


if __name__ == "__main__":
    main()

# NOTES / measured on 2026-07-20, "2026-07-20 22-55-48.mp4"
#   detection    : ~8 of 10 on-court players at conf 0.35
#   false pos    : crowd + bench behind the baseline get boxed
#   misses       : players mid-air (dunk/jump) — pose is off-distribution
#   team ID      : unreliable. gold jersey H~20 S~170, wood court H~15 S~90.
#                  a box with much floor in it reads gold. needs jersey
#                  segmentation, not a median over the torso rectangle.
#   speed        : 458 ms/frame CPU @416 — ~29x too slow for 60 Hz.
#                  GPU path: pip install onnxruntime-directml + yolov*-tiny
