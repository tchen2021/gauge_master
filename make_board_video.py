#!/usr/bin/env python3
"""
make_board_video.py - simulated gauge board for testing before the real board:
two 0-100 PSI gauges and four red indicator lights, as one camera would see it.

Writes the video AND a matching calibration file, so no clicking is needed:

  python make_board_video.py
  python gauge_reader.py --source board_test.mp4 --config board_test_config.json --log readings.jsonl

Timeline (46 s, loops forever in gauge_reader.py):
  gauge-1 : 50 PSI for 20 s -> 65 PSI for 20 s   (crosses the 60 limit)
  gauge-2 : 40 PSI for 10 s -> 70 PSI for 20 s -> 40 PSI (crosses at a different time)
  light-1 : ON while gauge-1 is above the 60 PSI limit
  light-2 : ON once gauge-1 has been above the limit for more than 10 s
  light-3 : ON while gauge-2 is above the limit
  light-4 : ON once gauge-2 has been above the limit for more than 10 s
"""
from __future__ import annotations

import argparse
import json
import math
import random

import cv2
import numpy as np

W, H, FPS, LOOP_S = 1280, 720, 10, 46
R = 170
GAUGES = {"gauge-1": (330, 300), "gauge-2": (950, 300)}
LIGHTS = {"light-1": (340, 625), "light-2": (540, 625), "light-3": (740, 625), "light-4": (940, 625)}
LIGHT_R = 28
START, END = 135.0, 405.0          # 270-degree clockwise scale, 0-100 PSI


def _pt(c, deg, r):
    a = math.radians(deg)
    return int(c[0] + r * math.cos(a)), int(c[1] + r * math.sin(a))


def draw_dial(img, c):
    cv2.circle(img, c, R + 13, (230, 230, 230), -1)
    cv2.circle(img, c, R + 13, (40, 40, 40), 3)
    for i in range(51):
        a = START + (END - START) * i / 50
        cv2.line(img, _pt(c, a, R * (0.78 if i % 5 == 0 else 0.85)), _pt(c, a, R * 0.95),
                 (30, 30, 30), 2 if i % 5 == 0 else 1)
    for v in range(0, 101, 10):
        x, y = _pt(c, START + (END - START) * v / 100, R * 0.62)
        cv2.putText(img, str(v), (x - 11, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 2)
    cv2.putText(img, "PSI", (c[0] - 18, c[1] + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)


def draw_needle(img, c, value):
    a = START + (END - START) * value / 100
    cv2.line(img, _pt(c, a + 180, R * 0.15), _pt(c, a, R * 0.9), (30, 30, 30), 4)
    cv2.circle(img, c, 11, (30, 30, 30), -1)


def draw_light(img, c, on):
    cv2.circle(img, c, LIGHT_R + 8, (190, 190, 195), -1)          # chrome bezel
    cv2.circle(img, c, LIGHT_R + 8, (120, 120, 125), 2)
    if on:
        cv2.circle(img, c, LIGHT_R, (60, 60, 255), -1)            # glowing red
        cv2.circle(img, c, int(LIGHT_R * 0.7), (200, 200, 255), -1)
    else:
        cv2.circle(img, c, LIGHT_R, (30, 30, 120), -1)            # dark red lens
        cv2.circle(img, (c[0] - 8, c[1] - 8), 4, (220, 220, 230), -1)   # small reflection


def ramp(t, t0, t1, a, b):
    if t <= t0:
        return a
    if t >= t1:
        return b
    x = (t - t0) / (t1 - t0)
    return a + (b - a) * (1 - math.cos(math.pi * x)) / 2


def g1(t):  # 50 -> 65 at 20-23 s, back 43-46 s
    return ramp(t, 20, 23, 50, 65) if t < 43 else ramp(t, 43, 46, 65, 50)


def g2(t):  # 40 -> 70 at 10-13 s, back 33-36 s
    return ramp(t, 10, 13, 40, 70) if t < 33 else ramp(t, 33, 36, 70, 40)


LIMIT, LATCH_S = 60.0, 10.0


def above_since(f, t: float) -> float:
    """Seconds the gauge function f has been continuously above LIMIT at time t."""
    if f(t) <= LIMIT:
        return 0.0
    s = t
    while s > 0 and f(s - 0.1) > LIMIT:
        s -= 0.1
    return t - s


def calibration() -> dict:
    gauges = [{"id": gid, "center": list(c), "start": list(_pt(c, START, R)),
               "mid": list(_pt(c, 270, R)), "end": list(_pt(c, END, R)),
               "min_value": 0.0, "max_value": 100.0, "unit": "PSI"}
              for gid, c in GAUGES.items()]
    lights = [{"id": lid, "center": list(c), "radius": float(LIGHT_R), "threshold": 0.35}
              for lid, c in LIGHTS.items()]
    return {"gauges": gauges, "lights": lights}


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a simulated 2-gauge, 4-light board video.")
    ap.add_argument("--out", default="board_test.mp4")
    ap.add_argument("--config-out", default="board_test_config.json")
    args = ap.parse_args()

    random.seed(2)
    base = np.full((H, W, 3), 95, np.uint8)
    cv2.rectangle(base, (40, 40), (W - 40, H - 30), (120, 120, 125), -1)   # panel
    for c in GAUGES.values():
        draw_dial(base, c)
    cv2.putText(base, "SIMULATED TEST BOARD", (60, H - 45), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (230, 230, 230), 1)

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise SystemExit("Could not create the video file.")
    for i in range(LOOP_S * FPS):
        t = i / FPS
        v1, v2 = g1(t) + random.uniform(-0.3, 0.3), g2(t) + random.uniform(-0.3, 0.3)
        img = base.copy()
        draw_needle(img, GAUGES["gauge-1"], v1)
        draw_needle(img, GAUGES["gauge-2"], v2)
        states = {"light-1": g1(t) > LIMIT,
                  "light-2": above_since(g1, t) > LATCH_S,
                  "light-3": g2(t) > LIMIT,
                  "light-4": above_since(g2, t) > LATCH_S}
        for lid, c in LIGHTS.items():
            draw_light(img, c, states[lid])
        writer.write(img)
    writer.release()
    with open(args.config_out, "w") as f:
        json.dump(calibration(), f, indent=2)
    print(f"Wrote {args.out} ({LOOP_S} s loop) and {args.config_out}")


if __name__ == "__main__":
    main()
