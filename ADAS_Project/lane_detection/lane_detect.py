import cv2
import numpy as np

def detect_lanes(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # White color mask
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Yellow color mask
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mask = cv2.bitwise_or(white_mask, yellow_mask)
    filtered = cv2.bitwise_and(frame, frame, mask=mask)

    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = edges.shape
    roi = np.zeros_like(edges)

    polygon = np.array([[
        (100, height),
        (width - 100, height),
        (width // 2, height // 2)
    ]], np.int32)

    cv2.fillPoly(roi, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, roi)

    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=80,
        maxLineGap=50
    )

    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)

            if slope < -0.5:
                left_lines.append(line[0])
            elif slope > 0.5:
                right_lines.append(line[0])

    return left_lines, right_lines
