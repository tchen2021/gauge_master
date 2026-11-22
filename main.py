import cv2

from Indicator import detect_all_indicators, find_indicator_rois
from panel_ID_perspective_transformation import CAMERA_INDEX, get_panel_aoi


def main():
    camera = cv2.VideoCapture(CAMERA_INDEX)
    indicator_rois = None
    previous_states = None

    try:
        if not camera.isOpened():
            raise RuntimeError(f"Could not open camera {CAMERA_INDEX}")

        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the camera")

            # Straighten the panel using the four ArUco markers.
            panel = get_panel_aoi(frame)

            if panel is None:
                cv2.imshow("Gauge Master", frame)
                previous_states = None
            else:
                cv2.imshow("Gauge Master", panel)

                # Find the four light boxes once and reuse them.
                if indicator_rois is None:
                    boxes = find_indicator_rois(panel)
                    if len(boxes) == 4:
                        indicator_rois = boxes
                        print("Found four indicator lights.")

                if indicator_rois is not None:
                    readings = detect_all_indicators(panel, indicator_rois)
                    states = {
                        name: reading["status"]
                        for name, reading in readings.items()
                    }
                    if states != previous_states:
                        print(states)
                        previous_states = states

                # Your teammate's gauge reader will take `panel` here.

            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()