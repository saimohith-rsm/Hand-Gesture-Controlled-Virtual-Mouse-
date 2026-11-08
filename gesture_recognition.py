"""gesture_recognition.py
Combine MediaPipe landmarks and finger kinematics to output high-level gestures.
Implements the 5 exact gesture mappings:
1. OPEN PALM (all 5 fingers extended) -> Move mouse cursor.
2. CUSTOM LEFT CLICK POSE (four fingers extended together, thumb folded inward) -> Left click.
3. CUSTOM RIGHT CLICK POSE (four fingers extended together, thumb extended outward) -> Right click.
4. FOUR-FINGER VERTICAL MOVEMENT (four fingers moving up/down) -> Scroll up / Scroll down.
5. CLOSED FIST -> Pause tracking. Open palm -> Resume tracking.
"""

from typing import List, Tuple, Optional
import numpy as np

from kinematics import euclidean_distance, finger_is_extended
from pose_calibration import PoseManager

# MediaPipe landmark indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

LITTLE_MCP = 17
LITTLE_PIP = 18
LITTLE_DIP = 19
LITTLE_TIP = 20

# Gesture constants
CURSOR_MOVE = "cursor_move"
LEFT_CLICK = "left_click"
RIGHT_CLICK = "right_click"
DOUBLE_CLICK = "double_click"
SCROLL = "scroll"
DRAG = "drag"
PAUSE = "pause"
NONE = "none"

class GestureRecognizer:
    """Stateful gesture recognizer implementing the 5 core virtual mouse gestures."""

    def __init__(self,
                 pinch_activate: float = 0.055,
                 pinch_release: float = 0.085,
                 confirmation_frames: int = 2,
                 double_click_interval: int = 14,
                 scroll_velocity_threshold: float = 6.0,
                 pose_manager: Optional[PoseManager] = None):
        self.pinch_activate = pinch_activate
        self.pinch_release = pinch_release
        self.confirmation_frames = confirmation_frames
        self.double_click_interval = double_click_interval
        self.scroll_velocity_threshold = scroll_velocity_threshold
        self.pose_manager = pose_manager if pose_manager is not None else PoseManager()

        # Runtime state
        self._paused = False
        self._left_click_triggered = False
        self._right_click_triggered = False
        self._prev_hand_y: Optional[float] = None
        self.last_pinch_ratio = 1.0

    def _get_palm_size(self, lm: List[Tuple[int, int, float]]) -> float:
        """Reference scale: distance from wrist (0) to middle MCP (9)."""
        if len(lm) > 9:
            d = euclidean_distance(lm[WRIST][:2], lm[MIDDLE_MCP][:2])
            if d > 10.0:
                return d
        return 100.0

    def _is_thumb_folded_inward(self, lm: List[Tuple[int, int, float]], palm_size: float) -> bool:
        """Check if thumb is folded inward across the palm.
        When folded, distance from thumb tip (4) to index MCP (5) or middle MCP (9) is small.
        """
        d_index_mcp = euclidean_distance(lm[THUMB_TIP][:2], lm[INDEX_MCP][:2]) / palm_size
        d_middle_mcp = euclidean_distance(lm[THUMB_TIP][:2], lm[MIDDLE_MCP][:2]) / palm_size
        return (d_index_mcp < 0.62) or (d_middle_mcp < 0.65)

    def _is_thumb_extended_outward(self, lm: List[Tuple[int, int, float]], palm_size: float) -> bool:
        """Check if thumb is extended outward away from the palm."""
        d_index_mcp = euclidean_distance(lm[THUMB_TIP][:2], lm[INDEX_MCP][:2]) / palm_size
        return d_index_mcp >= 0.65

    def _are_four_fingers_together(self, lm: List[Tuple[int, int, float]], palm_size: float) -> bool:
        """Check if index, middle, ring, and little fingers are touching / close together."""
        gap1 = euclidean_distance(lm[INDEX_TIP][:2], lm[MIDDLE_TIP][:2])
        gap2 = euclidean_distance(lm[MIDDLE_TIP][:2], lm[RING_TIP][:2])
        gap3 = euclidean_distance(lm[RING_TIP][:2], lm[LITTLE_TIP][:2])
        avg_gap = (gap1 + gap2 + gap3) / (3.0 * palm_size)
        return avg_gap < 0.38

    def _all_fingers_folded(self, lm: List[Tuple[int, int, float]]) -> bool:
        """Return True if index, middle, ring, and little fingers are folded into a fist."""
        return all(
            not finger_is_extended(lm, tip, tip - 2)
            for tip in (INDEX_TIP, MIDDLE_TIP, RING_TIP, LITTLE_TIP)
        )

    def recognize(self,
                  landmarks: List[Tuple[int, int, float]],
                  frame_shape: Tuple[int, int],
                  previous_gesture: Optional[str] = NONE) -> str:
        """Analyse the current frame and return one of the 5 exact gestures."""
        if not landmarks or len(landmarks) < 21:
            return NONE

        palm_size = self._get_palm_size(landmarks)

        # 1. Finger extension states
        index_ext = finger_is_extended(landmarks, INDEX_TIP, INDEX_PIP)
        middle_ext = finger_is_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP)
        ring_ext = finger_is_extended(landmarks, RING_TIP, RING_PIP)
        little_ext = finger_is_extended(landmarks, LITTLE_TIP, LITTLE_PIP)
        four_fingers_ext = index_ext and middle_ext and ring_ext and little_ext

        thumb_inward = self._is_thumb_folded_inward(landmarks, palm_size)
        thumb_outward = self._is_thumb_extended_outward(landmarks, palm_size)
        fingers_together = self._are_four_fingers_together(landmarks, palm_size)

        # Track hand vertical velocity for 4-finger scrolling
        hand_y = (landmarks[INDEX_TIP][1] + landmarks[MIDDLE_TIP][1] +
                  landmarks[RING_TIP][1] + landmarks[LITTLE_TIP][1]) / 4.0
        if self._prev_hand_y is not None:
            dy = self._prev_hand_y - hand_y
        else:
            dy = 0.0
        self._prev_hand_y = hand_y

        # Update live pinch ratio for HUD meter
        d_thumb_idx = euclidean_distance(landmarks[THUMB_TIP][:2], landmarks[INDEX_TIP][:2])
        self.last_pinch_ratio = (d_thumb_idx / palm_size) * 0.25

        # -----------------------------------------------------------------
        # GESTURE 5: CLOSED FIST -> PAUSE / OPEN PALM -> RESUME
        # -----------------------------------------------------------------
        if self._all_fingers_folded(landmarks):
            self._paused = True
            return PAUSE
        elif self._paused:
            # Resume only upon deliberate OPEN PALM (all 5 fingers extended)
            if four_fingers_ext and thumb_outward:
                self._paused = False
                return CURSOR_MOVE
            return PAUSE

        # -----------------------------------------------------------------
        # GESTURE 2: CUSTOM LEFT CLICK POSE
        # (Four fingers extended together, thumb folded inward)
        # -----------------------------------------------------------------
        is_left_click_pose = four_fingers_ext and fingers_together and thumb_inward
        if is_left_click_pose:
            # Reset other gesture states
            self._right_click_triggered = False
            # Click debounce: execute exactly one click per intentional gesture
            if not self._left_click_triggered:
                self._left_click_triggered = True
                return LEFT_CLICK
            else:
                return NONE
        else:
            self._left_click_triggered = False

        # -----------------------------------------------------------------
        # GESTURE 3: CUSTOM RIGHT CLICK POSE
        # (Four fingers extended together, thumb extended outward)
        # -----------------------------------------------------------------
        is_right_click_pose = four_fingers_ext and fingers_together and thumb_outward
        if is_right_click_pose and abs(dy) < self.scroll_velocity_threshold:
            self._left_click_triggered = False
            # Click debounce: execute exactly one right click per intentional gesture
            if not self._right_click_triggered:
                self._right_click_triggered = True
                return RIGHT_CLICK
            else:
                return NONE
        else:
            self._right_click_triggered = False

        # -----------------------------------------------------------------
        # GESTURE 4: FOUR-FINGER VERTICAL MOVEMENT (SCROLL UP / DOWN)
        # -----------------------------------------------------------------
        if four_fingers_ext and abs(dy) >= self.scroll_velocity_threshold:
            # Active vertical movement scrolls
            return SCROLL

        # -----------------------------------------------------------------
        # GESTURE 1: OPEN PALM (All five fingers extended) -> CURSOR MOVEMENT
        # -----------------------------------------------------------------
        if four_fingers_ext and thumb_outward:
            return CURSOR_MOVE

        # Default fallback
        if index_ext:
            return CURSOR_MOVE

        return NONE
