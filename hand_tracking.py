"""hand_tracking.py
Module for hand detection and landmark tracking using MediaPipe.
Supports both legacy MediaPipe solutions and the modern MediaPipe Tasks API.
"""

import os
import time
import urllib.request
import cv2
import numpy as np
from typing import List, Tuple, Optional
import mediapipe as mp
import config

# MediaPipe 21 hand landmarks connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17) # Little & Palm
]

DEFAULT_MODEL_URL = getattr(config, "MODEL_URL", "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")


def resolve_or_download_model(model_path: Optional[str] = None) -> str:
    """Ensure the hand landmarker model file exists.
    Checks the local project folder first. Only downloads if missing.
    If download fails, raises a clear RuntimeError with manual curl instructions.
    """
    if model_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_name = getattr(config, "MODEL_PATH", "hand_landmarker.task")
        model_path = os.path.join(base_dir, model_name)

    # 1. Check local project folder first
    if os.path.exists(model_path) and os.path.getsize(model_path) > 1000:
        return model_path

    # 2. Only download if missing
    url = getattr(config, "MODEL_URL", DEFAULT_MODEL_URL)
    print(f"[HandTracker] Model not found locally. Attempting download from:\n  {url}\n  to: {model_path}...")
    try:
        urllib.request.urlretrieve(url, model_path)
        if os.path.exists(model_path) and os.path.getsize(model_path) > 1000:
            print("[HandTracker] Model downloaded successfully.")
            return model_path
    except Exception as e:
        print(f"[HandTracker] Automatic download failed: {e}")

    instructions = (
        f"\n{'='*70}\n"
        f"  [ERROR] MediaPipe Hand Landmarker model file could not be found or downloaded.\n"
        f"  Expected path: {model_path}\n"
        f"\n"
        f"  MANUAL INSTALLATION INSTRUCTIONS:\n"
        f"  Run the following command in PowerShell / Command Prompt:\n"
        f"    curl.exe -o \"{model_path}\" \"{url}\"\n"
        f"\n"
        f"  Or download manually in your browser from:\n"
        f"    {url}\n"
        f"  and place it directly into the project directory:\n"
        f"    {os.path.dirname(model_path)}\n"
        f"{'='*70}\n"
    )
    raise RuntimeError(instructions)


class HandTracker:
    def __init__(self,
                 detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.7,
                 max_num_hands: int = 1,
                 model_path: Optional[str] = None):
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.max_num_hands = max_num_hands
        self.last_handedness: Optional[str] = None

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

            model_path = resolve_or_download_model(model_path)

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=max_num_hands,
                min_hand_detection_confidence=detection_confidence,
                min_hand_presence_confidence=tracking_confidence,
                running_mode=vision.RunningMode.VIDEO,
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            self._last_timestamp_ms = 0

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
        self.last_handedness = None

        if not self.use_tasks_api:
            results = self.hands.process(rgb)
            if results.multi_hand_landmarks:
                hand_lm = results.multi_hand_landmarks[0]
                landmarks = [(int(lm.x * w), int(lm.y * h), lm.z) for lm in hand_lm.landmark]
                if results.multi_handedness and len(results.multi_handedness) > 0:
                    self.last_handedness = results.multi_handedness[0].classification[0].label
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(time.time() * 1000)
            if ts_ms <= self._last_timestamp_ms:
                ts_ms = self._last_timestamp_ms + 1
            self._last_timestamp_ms = ts_ms

            result = self.detector.detect_for_video(mp_image, ts_ms)
            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                first_hand = result.hand_landmarks[0]
                landmarks = [(int(lm.x * w), int(lm.y * h), lm.z) for lm in first_hand]
                if result.handedness and len(result.handedness) > 0 and len(result.handedness[0]) > 0:
                    self.last_handedness = result.handedness[0][0].category_name

        if landmarks:
            annotated = self.draw_skeleton(annotated, landmarks)

        return landmarks, annotated

    def close(self):
        if not self.use_tasks_api and hasattr(self, "hands"):
            self.hands.close()
        elif hasattr(self, "detector"):
            self.detector.close()
