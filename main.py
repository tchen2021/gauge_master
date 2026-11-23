import cv2
# from networkx import display

from Indicator import detect_all_indicators, find_indicator_rois
from panel_ID_perspective_transformation import CAMERA_INDEX, get_panel_aoi
from gauge_reader import draw_gauges, read_all_gauges


def main():
    camera = cv2.VideoCapture(CAMERA_INDEX)
    indicator_rois = None
    # previous_states = None
    previous_pressures = None

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
                # previous_states = None
                previous_pressures = None
            else:
                gauges = read_all_gauges(panel.copy())
                pressures = {
                    name: round(gauge["pressure"], 1)
                    for name, gauge in gauges.items()
                    if gauge["pressure"] is not None
                }
                

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
                    # if states != previous_states:
                    #     print(states)
                    #     previous_states = states

                # Create a separate image for visualization
                display = draw_gauges(panel.copy(), gauges)

                # Draw indicator ROIs
                if indicator_rois is not None:
                    for i, (roi, indicator_box) in enumerate(indicator_rois):

                        x1, y1, rw, rh = roi
                        x, y, w, h = indicator_box

                        # Draw expanded ROI (blue)
                        cv2.rectangle(display, (x1, y1),
                                    (x1+rw, y1+rh), (255, 0, 0), 2)

                        # Draw indicator rectangle (green)
                        cv2.rectangle(display, (x, y),
                                    (x+w, y+h), (0, 255, 0), 2)

                        # Draw indicator name
                        cv2.putText(display, f"Indicator_{i+1}",
                                    (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 255), 1)

                # Display the annotated image
                cv2.imshow("Gauge Master", display)



                if pressures != previous_pressures:
                    print(pressures)
                    previous_pressures = pressures

                # gauge reader will take `panel` here.

            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()