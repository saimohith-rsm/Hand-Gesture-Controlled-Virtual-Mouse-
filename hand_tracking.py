"""hand_tracking.py
Module for hand detection and landmark tracking using MediaPipe.
Supports both legacy MediaPipe solutions and the modern MediaPipe Tasks API.
"""

import os
import urllib.request
import cv2
import numpy as np
from typing import List, Tuple, Optional
import mediapipe as mp

# MediaPipe 21 hand landmarks connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17) # Little & Palm
]

DEFAULT_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

class HandTracker:
    def __init__(self,
                 detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.7,
                 max_num_hands: int = 1,
                 model_path: Optional[str] = None):
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.max_num_hands = max_num_hands

        # Check if legacy mp.solutions is available
        self.use_tasks_api = not (hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"))

        if not self.use_tasks_api:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
            )
        else:
            # Modern MediaPipe Tasks API
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            if model_path is None:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(base_dir, "hand_landmarker.task")

            if not os.path.exists(model_path):
                print(f"[HandTracker] Downloading model to {model_path}...")
                try:
                    urllib.request.urlretrieve(DEFAULT_MODEL_URL, model_path)
                    print("[HandTracker] Model downloaded successfully.")
                except Exception as e:
                    print(f"[HandTracker] Error downloading model: {e}")
                    print(f"Please download it manually from: {DEFAULT_MODEL_URL}")

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=max_num_hands,
                min_hand_detection_confidence=detection_confidence,
                min_hand_presence_confidence=tracking_confidence,
                running_mode=vision.RunningMode.IMAGE,
            )
            self.detector = vision.HandLandmarker.create_from_options(options)

    def draw_skeleton(self, frame: np.ndarray, landmarks: List[Tuple[int, int, float]]) -> np.ndarray:
        """Draw 21 landmarks and skeletal connections on the frame."""
        # Draw connections (green lines)
        for p1, p2 in HAND_CONNECTIONS:
            if p1 < len(landmarks) and p2 < len(landmarks):
                pt1 = (landmarks[p1][0], landmarks[p1][1])
                pt2 = (landmarks[p2][0], landmarks[p2][1])
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Draw joints and fingertips
        for i, (x, y, _) in enumerate(landmarks):
            # Highlight fingertips (thumb=4, index=8, middle=12, ring=16, little=20)
            if i in (4, 8, 12, 16, 20):
                cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)  # Red filled
                cv2.circle(frame, (x, y), 8, (255, 255, 255), 1)  # White outline
            else:
                cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)  # Blue filled
                cv2.circle(frame, (x, y), 5, (255, 255, 255), 1)

        return frame

    def process(self, frame: np.ndarray) -> Tuple[Optional[List[Tuple[int, int, float]]], np.ndarray]:
        """Detect hands in a BGR frame.

        Returns:
            landmarks: List of (x, y, z) tuples in pixel coordinates for the first hand, or None.
            annotated_frame: Frame with skeleton drawing.
        """
        h, w = frame.shape[:2]
        annotated = frame.copy()
        landmarks = None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not self.use_tasks_api:
            results = self.hands.process(rgb)
            if results.multi_hand_landmarks:
                hand_lm = results.multi_hand_landmarks[0]
                landmarks = [(int(lm.x * w), int(lm.y * h), lm.z) for lm in hand_lm.landmark]
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.detector.detect(mp_image)
            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                first_hand = result.hand_landmarks[0]
                landmarks = [(int(lm.x * w), int(lm.y * h), lm.z) for lm in first_hand]

        if landmarks:
            annotated = self.draw_skeleton(annotated, landmarks)

        return landmarks, annotated

    def close(self):
        if not self.use_tasks_api and hasattr(self, "hands"):
            self.hands.close()
        elif hasattr(self, "detector"):
            self.detector.close()
