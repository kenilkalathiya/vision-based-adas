import cv2
import numpy as np

from lane_detection.lane_detect import detect_lanes
from vehicle_detection.vehicle_detect import detect_vehicles
from fusion.collision_warning import estimate_distance, collision_warning


def draw_average_line(frame, lines):
    if len(lines) == 0:
        return

    x_coords = []
    y_coords = []

    for x1, y1, x2, y2 in lines:
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])

    poly = np.polyfit(y_coords, x_coords, 1)

    y_bottom = frame.shape[0]
    y_top = int(frame.shape[0] * 0.6)

    x_bottom = int(poly[0] * y_bottom + poly[1])
    x_top = int(poly[0] * y_top + poly[1])

    cv2.line(frame, (x_bottom, y_bottom), (x_top, y_top), (0, 255, 0), 6)


cap = cv2.VideoCapture("data/input_videos/test.mp4")

if not cap.isOpened():
    print("Error: Video not found or cannot be opened.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ===== Lane Detection =====
    left_lines, right_lines = detect_lanes(frame)
    draw_average_line(frame, left_lines)
    draw_average_line(frame, right_lines)

    # ===== Vehicle Detection =====
    vehicles, fps = detect_vehicles(frame)

    for (x1, y1, x2, y2, conf) in vehicles:
        box_width = x2 - x1
        distance = estimate_distance(box_width)
        warning, ttc = collision_warning(distance)

        color = (0, 255, 0)
        label = "Vehicle"

        if warning:
            color = (0, 0, 255)
            label = "COLLISION WARNING!"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if distance is not None and ttc is not None:
            cv2.putText(
                frame,
                f"{label} {distance}m TTC:{ttc}s",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    # ===== FPS Display =====
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow("ADAS - Lane, Vehicle & Collision Warning", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
