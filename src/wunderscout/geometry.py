import cv2
import numpy as np

PITCH_CONFIG = {
    # --- LEFT GOAL LINE ---
    0: (0.000, 0.000),  # Top-Left Corner
    1: (0.000, 0.204),  # Top Edge of Penalty Box
    2: (0.000, 0.365),  # Top Edge of Goal Area
    3: (0.000, 0.635),  # Bottom Edge of Goal Area
    4: (0.000, 0.796),  # Bottom Edge of Penalty Box
    5: (0.000, 1.000),  # Bottom-Left Corner
    # --- LEFT PENALTY AREA ---
    6: (0.052, 0.365),
    7: (0.052, 0.635),
    8: (0.105, 0.500),  # Penalty Spot (Left)
    9: (0.157, 0.204),
    10: (0.157, 0.392),
    11: (0.157, 0.608),
    12: (0.157, 0.796),
    # --- MIDFIELD ---
    13: (0.413, 0.500),
    14: (0.500, 0.000),
    15: (0.500, 0.365),
    16: (0.500, 0.635),
    17: (0.500, 1.000),
    18: (0.587, 0.500),
    # --- RIGHT PENALTY AREA ---
    19: (0.843, 0.204),
    20: (0.843, 0.392),
    21: (0.843, 0.608),
    22: (0.843, 0.796),
    23: (0.895, 0.500),  # Penalty Spot (Right)
    24: (0.948, 0.365),
    25: (0.948, 0.635),
    # --- RIGHT GOAL LINE ---
    26: (1.000, 0.000),
    27: (1.000, 0.204),
    28: (1.000, 0.365),
    29: (1.000, 0.635),
    30: (1.000, 0.796),
    31: (1.000, 1.000),
}


class PitchMapper:
    def __init__(self, pitch_config=PITCH_CONFIG):
        self.pitch_config = pitch_config
        self.last_h = None

    def get_matrix(self, keypoints_xy, keypoints_conf):
        src_points = []
        dst_points = []

        for i, (xy, conf) in enumerate(zip(keypoints_xy, keypoints_conf)):
            if conf > 0.5 and i in self.pitch_config:
                src_points.append(xy)
                dst_points.append(self.pitch_config[i])

        if len(src_points) >= 4:
            H, _ = cv2.findHomography(
                np.array(src_points), np.array(dst_points), cv2.RANSAC
            )
            self.last_h = H

        return self.last_h

    def transform(self, points, H=None):
        target_h = H if H is not None else self.last_h
        if target_h is None or len(points) == 0:
            return []

        points_reshaped = np.array(points).reshape(-1, 1, 2).astype(np.float32)
        projected = cv2.perspectiveTransform(points_reshaped, target_h)
        return projected.reshape(-1, 2)
