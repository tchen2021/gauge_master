#!/usr/bin/env python3
"""
make_test_video.py - create a simulated gauge video for testing the pipeline
before the real ZED camera is used.

The dial is drawn exactly like sample_gauge.png (0-100 PSI, same position and
size), so the existing gauge_config.json calibration works for it.

Default: holds 50 PSI for 20 s, rises to 65 PSI over 3 s, holds 20 s, falls
back over 3 s (46 s per cycle), with a small needle wobble. gauge_reader.py
plays video files in real time and loops them.

  python make_test_video.py
  python make_test_video.py --low 50 --high 65 --hold 20 --ramp 3 --out my_test.mp4
"""
from __future__ import annotations

import argparse
import math
import random

import cv2
import numpy as np

W, H = 800, 600
C, R = (400, 300), 200
START, END = 135.0, 405.0          # 270-degree clockwise scale
VMIN, VMAX = 0.0, 100.0


def _pt(deg: float, r: float = R) -> tuple[int, int]:
    a = math.radians(deg)
    return int(C[0] + r * math.cos(a)), int(C[1] + r * math.sin(a))


def _dial() -> np.ndarray:
    img = np.full((H, W, 3), 90, np.uint8)
    cv2.circle(img, C, R + 15, (230, 230, 230), -1)
    cv2.circle(img, C, R + 15, (40, 40, 40), 3)
    for i in range(51):
        a = START + (END - START) * i / 50
        r0 = R * (0.78 if i % 5 == 0 else 0.85)
        cv2.line(img, _pt(a, r0), _pt(a, R * 0.95), (30, 30, 30), 2 if i % 5 == 0 else 1)
    for v in range(0, 101, 10):
        x, y = _pt(START + (END - START) * v / 100, R * 0.62)
        cv2.putText(img, str(v), (x - 12, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    cv2.putText(img, "PSI", (C[0] - 20, C[1] + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.putText(img, "SIMULATED TEST GAUGE", (15, H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return img


def frame_for(value: float, dial: np.ndarray) -> np.ndarray:
    img = dial.copy()
    a = START + (END - START) * (value - VMIN) / (VMAX - VMIN)
    cv2.line(img, _pt(a + 180, R * 0.15), _pt(a, R * 0.9), (30, 30, 30), 4)
    cv2.circle(img, C, 12, (30, 30, 30), -1)
    return img


def profile(low: float, high: float, hold: float, ramp: float, fps: int) -> list[float]:
    """One loop: hold low, ramp up, hold high, ramp down (smooth S-curves)."""
    vals: list[float] = []
    def add_hold(v):  vals.extend([v] * int(hold * fps))
    def add_ramp(a, b):
        n = max(1, int(ramp * fps))
        vals.extend(a + (b - a) * (1 - math.cos(math.pi * i / n)) / 2 for i in range(n))
    add_hold(low); add_ramp(low, high); add_hold(high); add_ramp(high, low)
    return vals


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a simulated gauge test video.")
    ap.add_argument("--low", type=float, default=50)
    ap.add_argument("--high", type=float, default=65)
    ap.add_argument("--hold", type=float, default=20, help="seconds at each level")
    ap.add_argument("--ramp", type=float, default=3, help="seconds to move between levels")
    ap.add_argument("--wobble", type=float, default=0.3, help="random needle jitter (PSI)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default="gauge_test_50_65.mp4")
    args = ap.parse_args()

    random.seed(1)
    dial = _dial()
    values = profile(args.low, args.high, args.hold, args.ramp, args.fps)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    if not writer.isOpened():
        raise SystemExit("Could not create the video file (codec not available).")
    for v in values:
        writer.write(frame_for(v + random.uniform(-args.wobble, args.wobble), dial))
    writer.release()
    print(f"Wrote {args.out}: {len(values) / args.fps:.0f} s loop at {args.fps} fps "
          f"({args.low:g} -> {args.high:g} PSI, {args.hold:g} s holds, {args.ramp:g} s ramps)")


if __name__ == "__main__":
    main()
