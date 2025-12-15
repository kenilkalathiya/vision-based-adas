REAL_VEHICLE_WIDTH = 1.8  # meters
FOCAL_LENGTH = 700        # pixels (approx)

def estimate_distance(box_width_pixels):
    if box_width_pixels == 0:
        return None
    distance = (REAL_VEHICLE_WIDTH * FOCAL_LENGTH) / box_width_pixels
    return round(distance, 2)

def collision_warning(distance, speed_kmph=50):
    speed_mps = speed_kmph / 3.6
    if distance is None:
        return False, None

    ttc = distance / speed_mps if speed_mps > 0 else None

    if ttc is not None and ttc < 2.0:
        return True, round(ttc, 2)

    return False, round(ttc, 2)
