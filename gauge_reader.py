import cv2
import numpy as np

# -----------------------------------
# 1. Settings
# -----------------------------------

# Only search the top part of the rectified panel (gauges sit in the top half)
# 55% instead of 50% because the bottom edge of each gauge face lands right on the halfway line
SEARCH_FRACTION = 0.55

# A gauge face is a big, round, white blob
MIN_FACE_AREA = 20000  # each face is ~90,000 px at 1000 x 733, LEDs and specks are far smaller
MIN_FACE_CIRCULARITY = 0.8  # faces measure ~0.89

# Needle search ring, as a fraction of the face radius
# Inner edge skips the hub and the short counterweight tail, outer edge stops before the numbers and tick marks
NEEDLE_RING_INNER = 0.40
NEEDLE_RING_OUTER = 0.70

# Dial calibration (angles in degrees, measured counterclockwise from the +x axis with y pointing up)
ZERO_ANGLE = 225.0  # 0 psi mark, lower left
SWEEP_ANGLE = 270.0  # clockwise sweep from 0 psi to full scale
MAX_PRESSURE = 60.0  # psi at the end of the sweep

# -----------------------------------
# 2. Find the gauge faces
# -----------------------------------

def find_gauges(frame):
    """
    Find the white gauge faces in the top of the rectified panel.

    Returns:
        gauges (list of (contour, hierarchy_index, (cx, cy, r))), sorted left to right, plus the hierarchy
    """
    search_height = int(frame.shape[0] * SEARCH_FRACTION)
    gray = cv2.cvtColor(frame[:search_height], cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Otsu picks the white/dark split each frame, so lighting changes are handled

    # RETR_CCOMP gives a two-level tree: white outer shapes (faces) and the dark holes inside them (needle, text, ticks)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return [], contours, None
    hierarchy = hierarchy[0]  # shape (N, 4): [next, previous, first_child, parent]

    gauges = []
    for i, contour in enumerate(contours):
        if hierarchy[i][3] != -1:  # holes have a parent, faces do not
            continue
        area = cv2.contourArea(contour)
        if area < MIN_FACE_AREA:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / perimeter**2
        if circularity < MIN_FACE_CIRCULARITY:
            continue

        M = cv2.moments(contour)
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]  # centroid of the face, close to the needle pivot
        r = np.sqrt(area / np.pi)  # radius of a circle with the same area
        gauges.append((i, (cx, cy, r)))

    gauges.sort(key=lambda item: item[1][0])  # left to right
    return gauges, contours, hierarchy

# -----------------------------------
# 3. Find the needle inside one face
# -----------------------------------

def find_needle(contours, hierarchy, gauge, frame_shape):
    """
    Pick the needle among the dark holes (children) of one gauge face.

    Returns:
        angle (degrees, counterclockwise from +x with y up) and tip (x, y), or (None, None)
    """
    face_index, (cx, cy, r) = gauge

    # Ring mask: keeps only the stretch where the long pointer is the only dark thing crossing
    yy, xx = np.ogrid[:frame_shape[0], :frame_shape[1]]
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    ring = (dist > NEEDLE_RING_INNER * r) & (dist < NEEDLE_RING_OUTER * r)

    best_pixels = None
    child = hierarchy[face_index][2]  # first hole inside this face
    while child != -1:
        hole = contours[child]
        hole_pts = hole[:, 0, :].astype(np.float32)
        nearest = np.min(np.hypot(hole_pts[:, 0] - cx, hole_pts[:, 1] - cy))
        if nearest < 0.25 * r:  # the needle is attached to the hub, text and ticks are not
            hole_mask = np.zeros(frame_shape, dtype=np.uint8)
            cv2.drawContours(hole_mask, [hole], -1, 255, -1)  # fill the hole
            ys, xs = np.nonzero((hole_mask > 0) & ring)
            if len(xs) > 0 and (best_pixels is None or len(xs) > len(best_pixels[0])):
                best_pixels = (xs, ys)  # keep the hole with the most pixels in the ring
        child = hierarchy[child][0]  # next hole at the same level

    if best_pixels is None:
        return None, None

    xs, ys = best_pixels
    mx, my = np.mean(xs), np.mean(ys)  # middle of the pointer segment inside the ring
    angle = np.degrees(np.arctan2(cy - my, mx - cx)) % 360  # flip y so angles read like a normal math plot
    tip = (cx + 0.85 * r * np.cos(np.radians(angle)), cy - 0.85 * r * np.sin(np.radians(angle)))
    return angle, tip

# -----------------------------------
# 4. Angle to pressure
# -----------------------------------

def angle_to_pressure(angle):
    """Map a needle angle to psi using the linear dial scale."""
    clockwise = (ZERO_ANGLE - angle) % 360  # degrees travelled clockwise from the 0 psi mark
    if clockwise > SWEEP_ANGLE + (360 - SWEEP_ANGLE) / 2:  # inside the dead zone, closer to 0 than to full scale
        clockwise -= 360
    pressure = clockwise / SWEEP_ANGLE * MAX_PRESSURE
    return float(np.clip(pressure, 0, MAX_PRESSURE))

# -----------------------------------
# 5. Read all gauges
# -----------------------------------

def read_all_gauges(frame):
    """
    Find both gauges and read their pressures.

    Returns:
        results (dict): "Gauge_1" -> {"pressure", "angle", "center", "radius", "tip"}, left to right
    """
    gauges, contours, hierarchy = find_gauges(frame)
    search_shape = (int(frame.shape[0] * SEARCH_FRACTION), frame.shape[1])

    results = {}
    for i, gauge in enumerate(gauges):
        angle, tip = find_needle(contours, hierarchy, gauge, search_shape)
        cx, cy, r = gauge[1]
        results[f"Gauge_{i+1}"] = {
            "pressure": None if angle is None else angle_to_pressure(angle),
            "angle": angle,
            "center": (cx, cy),
            "radius": r,
            "tip": tip
        }
    return results

def draw_gauges(frame, results):
    """Draw the face circle, needle line, and reading on a copy of the frame."""
    out = frame.copy()
    for name, g in results.items():
        cx, cy = map(int, g["center"])
        cv2.circle(out, (cx, cy), int(g["radius"]), (0, 255, 0), 2)
        if g["tip"] is not None:
            tx, ty = map(int, g["tip"])
            cv2.line(out, (cx, cy), (tx, ty), (0, 0, 255), 2)
            label = f"{g['pressure']:.1f} psi"
        else:
            label = "no needle"
        cv2.putText(out, label, (cx - 60, cy + int(g["radius"]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return out

# -----------------------------------
# 6. Standalone test loop
# -----------------------------------

def main():
    from panel_ID_perspective_transformation import CAMERA_INDEX, get_panel_aoi

    camera = cv2.VideoCapture(CAMERA_INDEX)
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Could not open camera {CAMERA_INDEX}")
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the camera")
            panel = get_panel_aoi(frame)
            if panel is None:
                cv2.imshow("Gauges", frame)
            else:
                results = read_all_gauges(panel)
                print({name: g["pressure"] for name, g in results.items()})
                cv2.imshow("Gauges", draw_gauges(panel, results))
            if (cv2.waitKey(1) & 0xFF) == 27:  # esc
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
