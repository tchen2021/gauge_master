import cv2
import numpy as np

# -----------------------------------
# 1. Define thresholds for HSV
# -----------------------------------

# Threshold of the brightness difference shall be at least 70
VALUE_THRESHOLD = 30                                        

# -----------------------------------
# 2. Find indicator ROIs including surrounding background
# -----------------------------------

# Create a function to find the regions of interest (ROIs) for the indicators in the frame
def find_indicator_rois(frame):
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)            

    # Define red color range in HSV
    # The first red range is [0, 70, 40] to [10, 255, 255]
    lower_red1 = np.array([0, 70, 40])
    upper_red1 = np.array([10, 255, 255])

    # The second red range is [170, 70, 40] to [179, 255, 255]
    lower_red2 = np.array([170, 70, 40])
    upper_red2 = np.array([179, 255, 255])

    # Create masks to detect red regions in the image
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine the two masks to get the final red mask
    red_mask = mask1 | mask2

    # Remove noise and fill gaps in the red mask using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours of the red regions in the mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    # Examine each contour to determine if it is a valid indicator candidate
    for contour in contours:
        
        # Calculate area and perimeter of the contour
        area = cv2.contourArea(contour)
        
        # If the area is too small, skip this contour
        if area < 50:
            continue
        
        # If the perimeter is zero, skip this contour
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        
        # Calculate circularity of the contour to filter out non-circular shapes
        circularity = 4 * np.pi * area / perimeter**2
        
        # If the circularity is less than 0.65, skip this contour
        if circularity < 0.65:
            continue
        
        # Calculate the smallest bounding rectangle for the contour
        # x, y are the top-left coordinates, w is width, h is height
        x, y, w, h = cv2.boundingRect(contour)

        # Add padding equal to 40% of the larger bounding-box dimension on each side
        padding = int(0.4 * max(w, h))
        x1 = max(0, x-padding)
        y1 = max(0, y-padding)
        x2 = min(frame.shape[1], x+w+padding)
        y2 = min(frame.shape[0], y+h+padding)

        # Original indicator bounding rectangle
        indicator_box = (x, y, w, h)

        # Expanded ROI including surrounding background
        roi = (x1, y1, x2-x1, y2-y1)

        # Store area, expanded ROI, and original indicator rectangle
        candidates.append((area, roi, indicator_box))

    # Select the four largest candidates
    candidates.sort(key=lambda item: item[0], reverse=True)

    # Store expanded ROI and original indicator rectangle together
    rois = [(item[1], item[2]) for item in candidates[:4]]

    # Sort indicators from left to right
    rois.sort(key=lambda item: item[0][0])

    return rois

# -----------------------------------
# 3. Detect a single indicator
# -----------------------------------

# Create a function to detect the status of a single indicator based on its ROI
def detect_indicator(frame, indicator_roi):

    # Unpack original indicator bounding rectangle and expanded ROI
    roi, indicator_box = indicator_roi

    # Convert the expanded ROI coordinates to integers
    x1, y1, rw, rh = map(int, roi)

    # Convert original indicator rectangle coordinates to integers
    x, y, w, h = map(int, indicator_box)

    # Crop the expanded ROI
    indicator = frame[y1:y1+rh, x1:x1+rw]

    # Convert BGR to HSV
    hsv = cv2.cvtColor(indicator, cv2.COLOR_BGR2HSV)

    # Extract brightness channel (V) only
    V = hsv[:, :, 2]

    # Find original indicator coordinates relative to the expanded ROI
    bx = x - x1
    by = y - y1

    # Create rectangular mask around the indicator (green rectangle)
    # Initialize a black mask for the entire ROI then fill the indicator area with white
    indicator_mask = np.zeros((rh, rw), dtype=np.uint8)
    indicator_mask[by:by+h, bx:bx+w] = 255

    # Create background mask (blue rectangle excluding green rectangle)
    # Initialize a white mask for the entire ROI then fill the indicator area with black
    background_mask = np.ones((rh, rw), dtype=np.uint8) * 255
    background_mask[by:by+h, bx:bx+w] = 0

    # Calculate mean brightness for the indicator and background regions and find the difference
    indicator_v = cv2.mean(V, mask=indicator_mask)[0]
    background_v = cv2.mean(V, mask=background_mask)[0]
    difference_v = indicator_v - background_v

    # Determine ON/OFF status
    is_on = difference_v > VALUE_THRESHOLD
    status = "ON" if is_on else "OFF"

    return status, float(difference_v)

# -----------------------------------
# 4. Detect all indicators
# -----------------------------------

# Store previous indicator readings
previous_results = {}

# Create a function to detect all indicators
def detect_all_indicators(frame, indicator_rois):

    global previous_results

    # Create a dictionary to store current results
    results = {}

    # Loop through each indicator ROI
    for i, indicator_roi in enumerate(indicator_rois):

        # Detect indicator status and brightness difference
        status, difference_v = detect_indicator(frame, indicator_roi)

        # Get indicator name
        indicator_name = f"Indicator_{i+1}"

        # Store current readings
        results[indicator_name] = {
            "status": status,
            "difference_v": difference_v
        }

        # Get previous status
        previous_status = previous_results.get(
            indicator_name, {}
        ).get("status")

        # Print only when indicator status changes
        if status != previous_status:
            print(
                f"{indicator_name}: "
                f"Status = {status}, "
                f"Difference V = {difference_v:.1f}"
            )

    # Save current readings for comparison with next frame
    previous_results = results

    # Return current readings
    return results