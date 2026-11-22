from __future__ import annotations

import cv2
import numpy as np

# Settings

CAMERA_INDEX = 1

# Rectified ROI size based on the 15 in by 11 in marker-center rectangle.
OUTPUT_WIDTH = 1000
OUTPUT_HEIGHT = 733

# Marker IDs assigned to each corner
TOP_LEFT_ID = 0
TOP_RIGHT_ID = 1
BOTTOM_RIGHT_ID = 3
BOTTOM_LEFT_ID = 2


# ArUco setup

aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

detector_params = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(
    aruco_dict,
    detector_params
)


def get_marker_centers(corners, ids):
    """
    Returns dictionary:
    marker ID -> marker center (x, y)
    """

    centers = {}

    if ids is None:
        return centers

    ids = ids.flatten()

    for marker_corners, marker_id in zip(corners, ids):

        # Shape from ArUco is usually (1, 4, 2)
        pts = marker_corners[0]

        center = np.mean(pts, axis=0)

        centers[int(marker_id)] = center

    return centers


def get_roi_corners(marker_centers):
    """
    Return ROI corners in order:
    top-left, top-right, bottom-right, bottom-left
    """

    required_ids = [
        TOP_LEFT_ID,
        TOP_RIGHT_ID,
        BOTTOM_RIGHT_ID,
        BOTTOM_LEFT_ID
    ]

    for marker_id in required_ids:
        if marker_id not in marker_centers:
            return None

    roi = np.array([
        marker_centers[TOP_LEFT_ID],
        marker_centers[TOP_RIGHT_ID],
        marker_centers[BOTTOM_RIGHT_ID],
        marker_centers[BOTTOM_LEFT_ID]
    ], dtype=np.float32)

    return roi


def perspective_transform(image, roi_corners):

    destination = np.array([
        [0, 0],
        [OUTPUT_WIDTH - 1, 0],
        [OUTPUT_WIDTH - 1, OUTPUT_HEIGHT - 1],
        [0, OUTPUT_HEIGHT - 1]
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(
        roi_corners,
        destination
    )

    warped = cv2.warpPerspective(
        image,
        H,
        (OUTPUT_WIDTH, OUTPUT_HEIGHT)
    )

    return warped


def get_panel_aoi(frame: np.ndarray) -> np.ndarray | None:
    """Return the rectified panel image at the configured output size.

    The output preserves the input's color format. Marker IDs 0, 1, 2,
    and 3 define the top-left, top-right, bottom-right, and bottom-left
    corners, respectively. Return None if any required marker is missing.
    """
    if frame is None or frame.size == 0:
        raise ValueError("Expected a non-empty camera frame.")

    corners, ids, _ = detector.detectMarkers(frame)
    roi_corners = get_roi_corners(get_marker_centers(corners, ids))
    if roi_corners is None:
        return None

    return perspective_transform(frame, roi_corners)


def main():
    camera = cv2.VideoCapture(CAMERA_INDEX)
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Could not open camera {CAMERA_INDEX}.")

        missing_panel = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
        cv2.putText(
            missing_panel,
            "Waiting for ArUco markers 0, 1, 2, and 3",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError(f"Could not read a frame from camera {CAMERA_INDEX}.")

            aoi = get_panel_aoi(frame)
            cv2.imshow("Camera Feed", frame)
            cv2.imshow("Rectified AOI", missing_panel if aoi is None else aoi)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
