#!/usr/bin/env python3
"""
gauge_reader.py - Step 1 of the gauge monitoring system.

Reads an analog gauge from a fixed camera (the ZED works as a normal webcam on
macOS) and outputs one JSON reading per interval.

How it works (following the keypoint idea from the KEPCO paper):
  1. Calibrate once: click the needle pivot (center), the scale START (min),
     any point MID-scale, and the scale END (max). Saved to gauge_config.json.
  2. Every frame: cast rays outward from the center and find the angle where
     the needle is (the ray that is "needle-colored" along most of its length).
  3. Convert the angle to a value by linear interpolation between start and end.
  4. Validity checks: if the needle isn't clearly found, or there is glare or
     darkness, report status "unavailable" instead of a wrong number.

A board can have several gauges and indicator lights: calibrate them all once
(--gauges N --lights M); each gauge gets its own reading (gauge-1, gauge-2, ...)
and each light reports ON/OFF (light-1, light-2, ...).

Usage:
  python gauge_reader.py --list-cameras
  python gauge_reader.py --calibrate --min 0 --max 60 --unit PSI
  python gauge_reader.py --source 1 --calibrate --gauges 2 --lights 4 --min 0 --max 60
  python gauge_reader.py --source 1 --log readings.jsonl
  python gauge_reader.py --needle-color red --log readings.jsonl

Keys in the live window:  q = quit,  c = recalibrate,  s = save snapshot
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

WINDOW = "Gauge reader"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ANGLE_STEP_DEG = 0.5          # resolution of the needle search
RAY_INNER, RAY_OUTER = 0.20, 0.75   # ray span as a fraction of gauge radius
SEPARATION_DEG = 15           # how far the 2nd-best candidate must be from the best

Point = tuple[float, float]


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def _angle_deg(origin: Point, p: Point) -> float:
    """Angle of p around origin in image coordinates (0 = right, grows clockwise)."""
    return math.degrees(math.atan2(p[1] - origin[1], p[0] - origin[0])) % 360.0


@dataclass
class GaugeCalibration:
    center: Point
    start: Point
    mid: Point
    end: Point
    min_value: float
    max_value: float
    unit: str = "PSI"

    @property
    def radius(self) -> float:
        return (math.dist(self.center, self.start) + math.dist(self.center, self.end)) / 2

    @property
    def clockwise(self) -> bool:
        """True if values increase clockwise (determined from the mid-scale click)."""
        a_s = _angle_deg(self.center, self.start)
        to_mid = (_angle_deg(self.center, self.mid) - a_s) % 360
        to_end = (_angle_deg(self.center, self.end) - a_s) % 360
        return to_mid < to_end

    def _travel(self, angle: float) -> float:
        """Degrees travelled from the start mark in the direction of increasing value."""
        a_s = _angle_deg(self.center, self.start)
        return (angle - a_s) % 360 if self.clockwise else (a_s - angle) % 360

    @property
    def span_deg(self) -> float:
        return self._travel(_angle_deg(self.center, self.end))

    def angle_to_value(self, needle_angle: float) -> tuple[float, str]:
        """Linear interpolation; returns (value, status)."""
        travel, span = self._travel(needle_angle), self.span_deg
        if travel <= span:
            frac = travel / span
            return self.min_value + frac * (self.max_value - self.min_value), "ok"
        # Needle is in the dead zone between END and START: decide which side.
        past_end = travel - span
        before_start = 360 - travel
        if before_start < past_end:
            return self.min_value, "below_scale"
        return self.max_value, "above_scale"

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "GaugeCalibration":
        data = json.loads(path.read_text())
        for key in ("center", "start", "mid", "end"):
            data[key] = tuple(data[key])
        return cls(**data)


# --------------------------------------------------------------------------- #
# Needle detection
# --------------------------------------------------------------------------- #
@dataclass
class NeedleResult:
    angle: Optional[float]
    confidence: float
    reason: str = ""


def _needle_likeness(frame: np.ndarray, color: str) -> np.ndarray:
    """Image where needle pixels are bright (high values)."""
    if color == "red":
        b, g, r = cv2.split(frame.astype(np.int16))
        return np.clip((r - np.maximum(g, b)) * 2, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return 255 - gray if color == "dark" else gray


def _ray_maps(cal: GaugeCalibration) -> tuple[np.ndarray, np.ndarray]:
    angles = np.deg2rad(np.arange(0, 360, ANGLE_STEP_DEG))
    radii = np.linspace(RAY_INNER * cal.radius, RAY_OUTER * cal.radius, 48)
    map_x = (cal.center[0] + np.outer(np.cos(angles), radii)).astype(np.float32)
    map_y = (cal.center[1] + np.outer(np.sin(angles), radii)).astype(np.float32)
    return map_x, map_y


def detect_needle(
    frame: np.ndarray,
    cal: GaugeCalibration,
    color: str = "dark",
    min_fill: float = 0.5,
    min_margin: float = 0.2,
) -> NeedleResult:
    map_x, map_y = _ray_maps(cal)

    # Lighting checks on the gauge face (fail safe instead of guessing).
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR)
    if face.mean() < 25:
        return NeedleResult(None, 0.0, "too dark")
    if (face > 250).mean() > 0.25:
        return NeedleResult(None, 0.0, "glare on gauge glass")

    likeness = cv2.GaussianBlur(_needle_likeness(frame, color), (3, 3), 0)
    samples = cv2.remap(likeness, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    otsu, _ = cv2.threshold(samples, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(otsu, 60)

    # Fraction of each ray that is needle-colored. The needle fills a whole ray;
    # tick marks and numbers only fill small parts of one.
    fill = (samples > threshold).mean(axis=1)
    fill = np.convolve(np.concatenate([fill[-1:], fill, fill[:1]]), np.ones(3) / 3, "valid")

    best = int(np.argmax(fill))
    peak = float(fill[best])
    n = len(fill)
    sep = int(SEPARATION_DEG / ANGLE_STEP_DEG)
    idx = np.arange(n)
    circ_dist = np.minimum(np.abs(idx - best), n - np.abs(idx - best))
    second = float(fill[circ_dist > sep].max())
    confidence = peak - second

    if peak < min_fill:
        return NeedleResult(None, confidence, "needle not found")
    if confidence < min_margin:
        return NeedleResult(None, confidence, "needle ambiguous")

    # Sub-step refinement with a parabola through the peak and its neighbours.
    y0, y1, y2 = fill[(best - 1) % n], fill[best], fill[(best + 1) % n]
    denom = y0 - 2 * y1 + y2
    offset = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    angle = ((best + offset) * ANGLE_STEP_DEG) % 360
    return NeedleResult(angle, confidence)


# --------------------------------------------------------------------------- #
# Reading = needle + calibration + smoothing
# --------------------------------------------------------------------------- #
@dataclass
class Reading:
    timestamp: str
    gauge_id: str
    value: Optional[float]       # smoothed value (what dashboards/alerts should use)
    raw_value: Optional[float]   # this frame only
    unit: str
    status: str                  # ok | below_scale | above_scale | unavailable
    confidence: float
    reason: str = ""


class GaugeReader:
    def __init__(self, cal: GaugeCalibration, gauge_id: str, color: str, smooth: int = 5):
        self.cal, self.gauge_id, self.color = cal, gauge_id, color
        self.history: deque[float] = deque(maxlen=smooth)
        self.last_angle: Optional[float] = None

    def read(self, frame: np.ndarray) -> Reading:
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        needle = detect_needle(frame, self.cal, self.color)
        self.last_angle = needle.angle
        if needle.angle is None:
            return Reading(now, self.gauge_id, None, None, self.cal.unit,
                           "unavailable", round(needle.confidence, 3), needle.reason)
        raw, status = self.cal.angle_to_value(needle.angle)
        self.history.append(raw)
        smoothed = statistics.median(self.history)
        return Reading(now, self.gauge_id, round(smoothed, 2), round(raw, 2),
                       self.cal.unit, status, round(needle.confidence, 3))


# --------------------------------------------------------------------------- #
# Camera / image input
# --------------------------------------------------------------------------- #
class FrameSource:
    """Camera index, video file, or still image. Crops the ZED's left image."""

    def __init__(self, source: str, stereo: str = "auto"):
        self.stereo = stereo
        self.still: Optional[np.ndarray] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.image_path: Optional[Path] = None
        self.image_mtime = 0.0
        self.video_file = False
        self.frame_period = 0.0
        self.next_frame_time = 0.0
        self.camera_index: Optional[int] = None
        self.next_retry = 0.0
        if Path(source).suffix.lower() in IMAGE_EXTENSIONS:
            self.image_path = Path(source)
            self.still = cv2.imread(source)
            if self.still is None:
                sys.exit(f"Could not read image: {source}")
            self.image_mtime = self.image_path.stat().st_mtime
        elif source.isdigit():
            # Live camera: never give up. If it is unplugged or not ready, keep
            # retrying while reporting "camera disconnected" readings.
            self.camera_index = int(source)
            if not self._open_camera():
                print(f"Camera {source} not available yet - will keep retrying every 2 s. "
                      "(Check the USB cable, --list-cameras, and System Settings > "
                      "Privacy & Security > Camera.)", file=sys.stderr)
        else:
            # Video file (e.g. a test recording): play at real speed and loop forever.
            self.cap = cv2.VideoCapture(source)
            self.video_file = True
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_period = 1.0 / fps if fps and fps > 0 else 0.1
        if self.video_file and not self.cap.isOpened():
            sys.exit(f"Could not open video '{source}'.")

    @property
    def is_camera(self) -> bool:
        return self.camera_index is not None

    def _open_camera(self) -> bool:
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.camera_index, backend)
        if not cap.isOpened():
            cap.release()
            self.cap = None
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)   # ZED HD720 side-by-side
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap = cap
        return True

    def read(self) -> Optional[np.ndarray]:
        if self.still is not None:
            time.sleep(0.03)
            self._reload_if_changed()
            frame = self.still.copy()
        elif self.is_camera:
            if self.cap is None:                       # disconnected: retry every 2 s
                if time.monotonic() < self.next_retry:
                    time.sleep(0.1)
                    return None
                self.next_retry = time.monotonic() + 2.0
                if not self._open_camera():
                    return None
                print("Camera reconnected.", file=sys.stderr)
            ok, frame = self.cap.read()
            if not ok:
                print("Camera stopped sending frames - reconnecting...", file=sys.stderr)
                self.cap.release()
                self.cap = None
                self.next_retry = time.monotonic() + 2.0
                return None
        else:
            if self.video_file:
                # Pace frames to the video's own frame rate (real time).
                wait = self.next_frame_time - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                self.next_frame_time = max(self.next_frame_time, time.monotonic()) + self.frame_period
            ok, frame = self.cap.read()
            if not ok and self.video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # end of video: start over
                ok, frame = self.cap.read()
            if not ok:
                return None
        h, w = frame.shape[:2]
        if self.stereo == "left" or (self.stereo == "auto" and w / h >= 3.0):
            frame = frame[:, : w // 2]   # ZED side-by-side: keep the left lens
        return frame

    def _reload_if_changed(self) -> None:
        """Test mode: if the image file is replaced, use the new picture.
        Lets you simulate a moving needle by copying a different image over it."""
        try:
            mtime = self.image_path.stat().st_mtime
        except OSError:
            return                      # file briefly missing while being replaced
        if mtime == self.image_mtime:
            return
        img = cv2.imread(str(self.image_path))
        if img is not None:             # ignore a half-written file; retry next frame
            self.still, self.image_mtime = img, mtime
            print(f"Image changed: {self.image_path.name} reloaded", file=sys.stderr)

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()


def list_cameras(max_index: int = 5) -> None:
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ok, frame = cap.read()
            if ok:
                h, w = frame.shape[:2]
                hint = "  <- looks like the ZED (side-by-side)" if w / h >= 3.0 else ""
                print(f"Camera {i}: {w}x{h}{hint}")
        cap.release()


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Board = several gauges + indicator lights seen by one camera
# --------------------------------------------------------------------------- #
@dataclass
class LightSpec:
    id: str
    center: Point
    radius: float
    threshold: float = 0.35   # fraction of bright pixels inside the lens that means "ON"


@dataclass
class GaugeSpec:
    id: str
    cal: GaugeCalibration


def _cal_from_dict(d: dict) -> GaugeCalibration:
    d = dict(d)
    for key in ("center", "start", "mid", "end"):
        d[key] = tuple(d[key])
    return GaugeCalibration(**{k: d[k] for k in ("center", "start", "mid", "end",
                                                 "min_value", "max_value", "unit")})


def save_board(path: Path, gauges: list[GaugeSpec], lights: list[LightSpec]) -> None:
    data = {"gauges": [{"id": g.id, **asdict(g.cal)} for g in gauges],
            "lights": [asdict(l) for l in lights]}
    path.write_text(json.dumps(data, indent=2))


def load_board(path: Path, default_id: str) -> tuple[list[GaugeSpec], list[LightSpec]]:
    data = json.loads(path.read_text())
    if "gauges" not in data:                      # old single-gauge file
        return [GaugeSpec(default_id, _cal_from_dict(data))], []
    gauges = [GaugeSpec(g["id"], _cal_from_dict(g)) for g in data["gauges"]]
    lights = [LightSpec(l["id"], tuple(l["center"]), float(l["radius"]),
                        float(l.get("threshold", 0.35))) for l in data.get("lights", [])]
    return gauges, lights


def light_state(frame: np.ndarray, spec: LightSpec) -> tuple[bool, float]:
    """ON if enough of the lens is bright. Returns (on, score 0-1)."""
    h, w = frame.shape[:2]
    cx, cy, r = int(spec.center[0]), int(spec.center[1]), max(2, int(spec.radius * 0.8))
    x0, x1, y0, y1 = max(0, cx - r), min(w, cx + r + 1), max(0, cy - r), min(h, cy + r + 1)
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return False, 0.0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    v = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 2]
    score = float((v[mask] >= 200).mean())
    return score >= spec.threshold, score


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def light_reading(spec: LightSpec, on: Optional[bool], score: float, status: str = "ok",
                  reason: str = "") -> dict:
    return {"timestamp": _now_iso(), "type": "light", "light_id": spec.id,
            "on": on, "score": round(score, 3), "status": status, "reason": reason}


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def _banner(view: np.ndarray, msg: str) -> None:
    cv2.putText(view, msg, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(view, msg, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def _click_points(frame: np.ndarray, prompts: list[str], labels: str,
                  done: Optional[list] = None) -> Optional[list[Point]]:
    """Generic click collector. ENTER = accept, R = redo, ESC = cancel."""
    points: list[Point] = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(prompts):
            points.append((float(x), float(y)))

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    try:
        while True:
            view = frame.copy()
            for p in done or []:                       # earlier calibrations, for reference
                cv2.circle(view, (int(p[0]), int(p[1])), 4, (160, 160, 160), -1)
            for i, p in enumerate(points):
                cv2.circle(view, (int(p[0]), int(p[1])), 5, (0, 200, 255), -1)
                cv2.putText(view, labels[i], (int(p[0]) + 8, int(p[1]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            _banner(view, prompts[len(points)] if len(points) < len(prompts)
                    else "ENTER = save,  R = redo,  ESC = cancel")
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10) and len(points) == len(prompts):
                return points
            if key in (ord("r"), ord("R")):
                points.clear()
            if key == 27:
                return None
    finally:
        cv2.setMouseCallback(WINDOW, lambda *a: None)


def run_calibration(source: FrameSource, min_value: float, max_value: float, unit: str,
                    label: str = "", frame: Optional[np.ndarray] = None,
                    done: Optional[list] = None) -> Optional[GaugeCalibration]:
    if frame is None:
        frame = source.read()
        if frame is None:
            sys.exit("No frame from camera.")
    tag = f"[{label}] " if label else ""
    prompts = [f"{tag}Click the needle PIVOT (center)",
               f"{tag}Click the scale START ({min_value:g} {unit})",
               f"{tag}Click any point MID-scale",
               f"{tag}Click the scale END ({max_value:g} {unit})"]
    pts = _click_points(frame, prompts, "CSME", done)
    if pts is None:
        return None
    return GaugeCalibration(*pts, min_value=min_value, max_value=max_value, unit=unit)


def run_board_calibration(source: FrameSource, n_gauges: int, n_lights: int,
                          ranges: list[tuple[float, float, str]]
                          ) -> Optional[tuple[list[GaugeSpec], list[LightSpec]]]:
    frame = source.read()
    if frame is None:
        sys.exit("No frame from camera.")
    gauges, done = [], []
    for i in range(n_gauges):
        gid = f"gauge-{i + 1}"
        lo, hi, unit = ranges[min(i, len(ranges) - 1)]
        cal = run_calibration(source, lo, hi, unit, label=gid, frame=frame, done=done)
        if cal is None:
            return None
        gauges.append(GaugeSpec(gid, cal))
        done += [cal.center, cal.start, cal.mid, cal.end]
    lights = []
    for i in range(n_lights):
        lid = f"light-{i + 1}"
        pts = _click_points(frame, [f"[{lid}] Click the CENTER of the light",
                                    f"[{lid}] Click the EDGE of the light's lens"], "CE", done)
        if pts is None:
            return None
        lights.append(LightSpec(lid, pts[0], math.dist(pts[0], pts[1])))
        done += [pts[0]]
    return gauges, lights


def draw_overlay(frame: np.ndarray, gauges: list[GaugeSpec], readers: dict,
                 readings: dict, lights: list[LightSpec], light_states: dict) -> np.ndarray:
    view = frame.copy()
    for g in gauges:
        cal, reader, reading = g.cal, readers[g.id], readings[g.id]
        c = (int(cal.center[0]), int(cal.center[1]))
        for p in (cal.start, cal.end):
            cv2.line(view, c, (int(p[0]), int(p[1])), (255, 160, 0), 1)
        if reader.last_angle is not None:
            a = math.radians(reader.last_angle)
            tip = (int(c[0] + cal.radius * math.cos(a)), int(c[1] + cal.radius * math.sin(a)))
            cv2.line(view, c, tip, (0, 220, 0), 2)
        if reading.status == "unavailable":
            text, col = f"{g.id}: UNAVAILABLE ({reading.reason})", (0, 0, 255)
        else:
            flag = "" if reading.status == "ok" else f" ({reading.status})"
            text, col = f"{g.id}: {reading.value:.1f} {reading.unit}{flag}", (0, 200, 0)
        org = (max(5, int(cal.center[0] - cal.radius)), max(25, int(cal.center[1] - cal.radius - 15)))
        cv2.putText(view, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(view, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    for l in lights:
        on, score = light_states[l.id]
        c, r = (int(l.center[0]), int(l.center[1])), int(l.radius)
        cv2.circle(view, c, r + 4, (0, 255, 255) if on else (160, 160, 160), 2)
        label = f"{l.id}: {'ON' if on else 'OFF'}"
        org = (c[0] - r - 10, c[1] + r + 22)
        cv2.putText(view, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(view, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255) if on else (220, 220, 220), 1)
    return view


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Read analog gauges and indicator lights from a camera.")
    ap.add_argument("--source", default="0", help="camera index, video file, or image file")
    ap.add_argument("--stereo", choices=["auto", "left", "none"], default="auto",
                    help="crop the ZED's left image (auto-detects side-by-side frames)")
    ap.add_argument("--config", type=Path, default=Path("gauge_config.json"))
    ap.add_argument("--gauge-id", default="gauge-1", help="id for an old single-gauge config")
    ap.add_argument("--needle-color", choices=["dark", "light", "red"], default="dark")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between outputs")
    ap.add_argument("--log", type=Path, help="append readings to this .jsonl file")
    ap.add_argument("--headless", action="store_true", help="no window, output only")
    ap.add_argument("--list-cameras", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--gauges", type=int, default=1, help="number of gauges to calibrate")
    ap.add_argument("--lights", type=int, default=0, help="number of indicator lights to calibrate")
    ap.add_argument("--min", type=float, help="gauge minimum (for --calibrate)")
    ap.add_argument("--max", type=float, help="gauge maximum (for --calibrate)")
    ap.add_argument("--unit", default="PSI")
    args = ap.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    source = FrameSource(args.source, args.stereo)
    try:
        if args.calibrate:
            if args.min is None or args.max is None:
                sys.exit("--calibrate needs --min and --max, e.g. --min 0 --max 60")
            board = run_board_calibration(source, args.gauges, args.lights,
                                          [(args.min, args.max, args.unit)])
            if board is None:
                sys.exit("Calibration cancelled.")
            gauges, lights = board
            save_board(args.config, gauges, lights)
            print(f"Saved calibration of {len(gauges)} gauge(s) and {len(lights)} "
                  f"light(s) to {args.config}", file=sys.stderr)
        elif not args.config.exists():
            sys.exit(f"No {args.config} yet. Run first, e.g.:\n"
                     "  python gauge_reader.py --calibrate --gauges 2 --lights 4 --min 0 --max 60")
        else:
            gauges, lights = load_board(args.config, args.gauge_id)

        readers = {g.id: GaugeReader(g.cal, g.id, args.needle_color) for g in gauges}
        last_output = 0.0

        def emit(lines: list[dict]) -> None:
            nonlocal last_output
            if time.monotonic() - last_output < args.interval:
                return
            last_output = time.monotonic()
            out = "\n".join(json.dumps(d) for d in lines)
            print(out, flush=True)
            if args.log:
                with args.log.open("a") as f:
                    f.write(out + "\n")

        while True:
            frame = source.read()
            if frame is None:
                if not source.is_camera:
                    print("Video/image source ended.", file=sys.stderr)
                    break
                # Camera lost: keep reporting so the alert monitor can raise a camera ticket.
                lines = []
                for g in gauges:
                    readers[g.id].last_angle = None
                    lines.append(asdict(Reading(_now_iso(), g.id, None, None, g.cal.unit,
                                                "unavailable", 0.0, "camera disconnected")))
                lines += [light_reading(l, None, 0.0, "unavailable", "camera disconnected")
                          for l in lights]
                emit(lines)
                if not args.headless:
                    blank = np.zeros((360, 640, 3), np.uint8)
                    cv2.putText(blank, "CAMERA DISCONNECTED - reconnecting...", (20, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow(WINDOW, blank)
                    if (cv2.waitKey(100) & 0xFF) == ord("q"):
                        break
                continue

            readings = {g.id: readers[g.id].read(frame) for g in gauges}
            light_states = {l.id: light_state(frame, l) for l in lights}
            emit([asdict(r) for r in readings.values()] +
                 [light_reading(l, *light_states[l.id]) for l in lights])

            if args.headless:
                continue
            cv2.imshow(WINDOW, draw_overlay(frame, gauges, readers, readings, lights, light_states))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                ranges = [(g.cal.min_value, g.cal.max_value, g.cal.unit) for g in gauges]
                board = run_board_calibration(source, len(gauges), len(lights), ranges)
                if board is not None:
                    new_g, new_l = board
                    # keep the existing ids / light thresholds
                    gauges = [GaugeSpec(o.id, n.cal) for o, n in zip(gauges, new_g)]
                    lights = [LightSpec(o.id, n.center, n.radius, o.threshold)
                              for o, n in zip(lights, new_l)]
                    save_board(args.config, gauges, lights)
                    readers = {g.id: GaugeReader(g.cal, g.id, args.needle_color) for g in gauges}
            if key == ord("s"):
                name = f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.png"
                cv2.imwrite(name, frame)
                print(f"Saved {name}", file=sys.stderr)
    except KeyboardInterrupt:
        pass
    finally:
        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
