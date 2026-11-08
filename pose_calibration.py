"""pose_calibration.py
Module for scanning, capturing, saving, and matching custom hand poses in real-time.
Allows the user to redefine gestures (Click, Scroll, Pause, etc.) by showing custom hand poses to the camera.
"""

import os
import json
import math
from typing import List, Tuple, Dict, Optional
import numpy as np

from kinematics import euclidean_distance, finger_is_extended

WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9

class PoseManager:
    """Manages custom scanned hand poses."""

    def __init__(self, filepath: Optional[str] = None):
        if filepath is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(base_dir, "custom_poses.json")
        self.filepath = filepath
        self.poses: Dict[str, dict] = {}
        self.load_poses()

    @staticmethod
    def extract_feature_vector(landmarks: List[Tuple[int, int, float]]) -> Optional[List[float]]:
        """Extract a 42-D scale-invariant and translation-invariant normalized feature vector.
        Wrist (0) is origin (0, 0), normalized by palm size (distance 0 to 9).
        """
        if not landmarks or len(landmarks) < 21:
            return None

        wrist_x, wrist_y = landmarks[WRIST][:2]
        palm_size = euclidean_distance(landmarks[WRIST][:2], landmarks[MIDDLE_MCP][:2])
        if palm_size < 10:
            palm_size = 100.0

        vector = []
        for x, y, _ in landmarks:
            norm_x = (x - wrist_x) / palm_size
            norm_y = (y - wrist_y) / palm_size
            vector.extend([round(norm_x, 4), round(norm_y, 4)])

        return vector

    @staticmethod
    def get_finger_states(landmarks: List[Tuple[int, int, float]]) -> List[bool]:
        """Return [Thumb, Index, Middle, Ring, Little] extension states."""
        if not landmarks or len(landmarks) < 21:
            return [False] * 5
        # Compare distance from wrist (0)
        thumb_ext = finger_is_extended(landmarks, 4, 2, wrist_idx=WRIST)
        index_ext = finger_is_extended(landmarks, 8, 6, wrist_idx=WRIST)
        middle_ext = finger_is_extended(landmarks, 12, 10, wrist_idx=WRIST)
        ring_ext = finger_is_extended(landmarks, 16, 14, wrist_idx=WRIST)
        little_ext = finger_is_extended(landmarks, 20, 18, wrist_idx=WRIST)
        return [thumb_ext, index_ext, middle_ext, ring_ext, little_ext]

    def capture_pose(self, gesture_name: str, landmarks: List[Tuple[int, int, float]]) -> bool:
        """Capture and record the current hand pose for a specific gesture."""
        vec = self.extract_feature_vector(landmarks)
        if vec is None:
            return False

        finger_states = self.get_finger_states(landmarks)

        # Compute pinch distances
        diag = 800.0
        palm_size = euclidean_distance(landmarks[WRIST][:2], landmarks[MIDDLE_MCP][:2])
        scale = palm_size if palm_size > 15 else diag
        thumb_idx_dist = round((euclidean_distance(landmarks[4][:2], landmarks[8][:2]) / scale) * 0.25, 4)
        thumb_mid_dist = round((euclidean_distance(landmarks[4][:2], landmarks[12][:2]) / scale) * 0.25, 4)

        self.poses[gesture_name] = {
            "name": gesture_name,
            "feature_vector": vec,
            "finger_states": finger_states,
            "thumb_index_pinch": thumb_idx_dist,
            "thumb_middle_pinch": thumb_mid_dist,
        }
        return True

    def match_pose(self, landmarks: List[Tuple[int, int, float]], gesture_name: str, threshold: float = 0.45) -> Tuple[bool, float]:
        """Compare current live hand against a captured pose profile.
        Returns (is_match, similarity_distance).
        Lower similarity distance = closer match.
        """
        if gesture_name not in self.poses:
            return False, 1.0

        target = self.poses[gesture_name]
        live_vec = self.extract_feature_vector(landmarks)
        if live_vec is None:
            return False, 1.0

        target_vec = target["feature_vector"]
        # Mean Euclidean distance across normalized landmarks
        total_dist = 0.0
        num_points = len(live_vec) // 2
        for i in range(num_points):
            dx = live_vec[2 * i] - target_vec[2 * i]
            dy = live_vec[2 * i + 1] - target_vec[2 * i + 1]
            total_dist += math.hypot(dx, dy)

        avg_dist = total_dist / num_points
        is_match = (avg_dist < threshold)
        return is_match, avg_dist

    def save_poses(self) -> bool:
        """Save captured poses to JSON file."""
        try:
            with open(self.filepath, "w") as f:
                json.dump(self.poses, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving poses: {e}")
            return False

    def load_poses(self) -> bool:
        """Load captured poses from JSON file if available."""
        if not os.path.exists(self.filepath):
            return False
        try:
            with open(self.filepath, "r") as f:
                self.poses = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading poses: {e}")
            return False

