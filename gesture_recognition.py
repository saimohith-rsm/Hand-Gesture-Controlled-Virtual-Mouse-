"""gesture_recognition.py
Combine MediaPipe landmarks and finger kinematics to output high-level gestures.
Implements the 5 core gesture mappings:
1. OPEN PALM (all 5 fingers extended) -> Move mouse cursor.
2. CUSTOM LEFT CLICK POSE (four fingers extended together, thumb folded inward) -> Left click.
3. CUSTOM RIGHT CLICK POSE (four fingers extended together, thumb extended outward) -> Right click.
4. SCROLL: Two fingers extended (index + middle, ring + little folded) or custom scroll pose.
5. CLOSED FIST -> Pause tracking. Open palm -> Resume tracking.
Also supports DRAG (pinch held) and DOUBLE_CLICK (two clicks within interval), plus custom scanned poses.
"""

from typing import List, Tuple, Optional
import numpy as np
import config

from kinematics import (
    euclidean_distance,
    finger_is_extended,
    is_thumb_folded_inward,
    is_thumb_extended_outward,
)
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

# Public gesture constants
CURSOR_MOVE = "cursor_move"
LEFT_CLICK = "left_click"
RIGHT_CLICK = "right_click"
DOUBLE_CLICK = "double_click"
DRAG = "drag"
MIDDLE_CLICK = "middle_click"
SCROLL_V = "scroll_v"
SCROLL_H = "scroll_h"
ZOOM = "zoom"
BACK = "back"
FORWARD = "forward"
PAUSE = "pause"
NONE = "none"

# Backward compatibility alias
SCROLL = SCROLL_V

class GestureRecognizer:
    """Stateful gesture recognizer with hysteresis, debouncing, custom pose, handedness, and stability checks."""

    def __init__(self,
                 pinch_activate: Optional[float] = None,
                 pinch_release: Optional[float] = None,
                 confirmation_frames: Optional[int] = None,
                 double_click_interval: Optional[int] = None,
                 scroll_velocity_threshold: float = 6.0,
                 click_stability_frames: Optional[int] = None,
                 max_click_speed: Optional[float] = None,
                 swipe_speed_threshold: Optional[float] = None,
                 swipe_cooldown_frames: Optional[int] = None,
                 pose_manager: Optional[PoseManager] = None):
        self.pinch_activate = pinch_activate if pinch_activate is not None else getattr(config, "PINCH_ACTIVATION_THRESHOLD", 0.055)
        self.pinch_release = pinch_release if pinch_release is not None else getattr(config, "PINCH_RELEASE_THRESHOLD", 0.085)
        self.confirmation_frames = confirmation_frames if confirmation_frames is not None else getattr(config, "GESTURE_CONFIRMATION_FRAMES", 2)
        self.double_click_interval = double_click_interval if double_click_interval is not None else getattr(config, "DOUBLE_CLICK_MAX_INTERVAL", 14)
        self.scroll_velocity_threshold = scroll_velocity_threshold
        self.click_stability_frames = click_stability_frames if click_stability_frames is not None else 1
        self.max_click_speed = max_click_speed if max_click_speed is not None else getattr(config, "MAX_CLICK_SPEED_THRESHOLD", 32.0)
        self.swipe_speed_threshold = swipe_speed_threshold if swipe_speed_threshold is not None else getattr(config, "SWIPE_SPEED_THRESHOLD", 18.0)
        self.swipe_cooldown_frames = swipe_cooldown_frames if swipe_cooldown_frames is not None else getattr(config, "SWIPE_COOLDOWN_FRAMES", 15)
        self.pose_manager = pose_manager if pose_manager is not None else PoseManager()

        # Runtime state
        self._paused = False
        self._left_click_triggered = False
        self._right_click_triggered = False
        self._left_candidate_frames = 0
        self._right_candidate_frames = 0
        self._prev_hand_y: Optional[float] = None
        self._custom_pose_frames = 0
        self._last_matched_custom_pose: Optional[str] = None
        self.last_handedness: Optional[str] = None

        # Drag state
        self._is_dragging = False

        # Pinch click states
        self._double_click_candidate_frames = 0
        self._double_click_pinch_triggered = False
        self._middle_click_candidate_frames = 0
        self._middle_click_pinch_triggered = False

        # Motion tracking states
        self._prev_scroll_pos: Optional[Tuple[float, float]] = None
        self._prev_shaka_x: Optional[float] = None
        self._swipe_cooldown = 0

        # Double click & legacy pinch state
        self._pinch_frames = 0
        self._frames_since_last_left = 1000
        self._double_click_pending = False
        self._pinch_released_after_click = False
        self.is_pinched = False
        self.last_pinch_ratio = 1.0

    def _get_palm_size(self, lm: List[Tuple[int, int, float]]) -> float:
        """Reference scale: distance from wrist (0) to middle MCP (9)."""
        if len(lm) > 9:
            d = euclidean_distance(lm[WRIST][:2], lm[MIDDLE_MCP][:2])
            if d > 10.0:
                return d
        return 100.0

    def _is_thumb_folded_inward(self, lm: List[Tuple[int, int, float]], palm_size: float, handedness: Optional[str] = None) -> bool:
        """Check if thumb is folded inward across the palm for either Left or Right hand."""
        return is_thumb_folded_inward(lm, palm_size, handedness or self.last_handedness)

    def _is_thumb_extended_outward(self, lm: List[Tuple[int, int, float]], palm_size: float, handedness: Optional[str] = None) -> bool:
        """Check if thumb is extended outward away from the palm for either Left or Right hand."""
        return is_thumb_extended_outward(lm, palm_size, handedness or self.last_handedness)

    def _are_four_fingers_together(self, lm: List[Tuple[int, int, float]], palm_size: float) -> bool:
        """Check if index, middle, ring, and little fingers are touching / close together."""
        gap1 = euclidean_distance(lm[INDEX_TIP][:2], lm[MIDDLE_TIP][:2])
        gap2 = euclidean_distance(lm[MIDDLE_TIP][:2], lm[RING_TIP][:2])
        gap3 = euclidean_distance(lm[RING_TIP][:2], lm[LITTLE_TIP][:2])
        avg_gap = (gap1 + gap2 + gap3) / (3.0 * palm_size)
        return avg_gap < 0.38

    def _all_fingers_folded(self, lm: List[Tuple[int, int, float]]) -> bool:
        """Return True if all 4 fingers (index, middle, ring, little) are folded into a fist."""
        margin = getattr(config, "FINGER_EXTENSION_MARGIN", 1.1)
        return all(
            not finger_is_extended(lm, tip, tip - 2, wrist_idx=WRIST, margin=margin)
            for tip in (INDEX_TIP, MIDDLE_TIP, RING_TIP, LITTLE_TIP)
        )

    def recognize(self,
                  landmarks: List[Tuple[int, int, float]],
                  frame_shape: Tuple[int, int],
                  previous_gesture: Optional[str] = NONE,
                  handedness: Optional[str] = None,
                  hand_speed: float = 0.0) -> str:
        """Analyse the current frame and return a gesture constant."""
        if not landmarks or len(landmarks) < 21:
            return NONE

        if handedness is not None:
            self.last_handedness = handedness

        palm_size = self._get_palm_size(landmarks)
        margin = getattr(config, "FINGER_EXTENSION_MARGIN", 1.1)
        self._frames_since_last_left += 1

        # Finger extension states (rotation-invariant relative to wrist)
        index_ext = finger_is_extended(landmarks, INDEX_TIP, INDEX_PIP, wrist_idx=WRIST, margin=margin)
        middle_ext = finger_is_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP, wrist_idx=WRIST, margin=margin)
        ring_ext = finger_is_extended(landmarks, RING_TIP, RING_PIP, wrist_idx=WRIST, margin=margin)
        little_ext = finger_is_extended(landmarks, LITTLE_TIP, LITTLE_PIP, wrist_idx=WRIST, margin=margin)
        four_fingers_ext = index_ext and middle_ext and ring_ext and little_ext

        thumb_inward = self._is_thumb_folded_inward(landmarks, palm_size, handedness=handedness)
        thumb_outward = self._is_thumb_extended_outward(landmarks, palm_size, handedness=handedness)
        fingers_together = self._are_four_fingers_together(landmarks, palm_size)

        # Compute pinch distances normalized by palm size
        d_thumb_idx = euclidean_distance(landmarks[THUMB_TIP][:2], landmarks[INDEX_TIP][:2]) / palm_size
        d_thumb_mid = euclidean_distance(landmarks[THUMB_TIP][:2], landmarks[MIDDLE_TIP][:2]) / palm_size
        d_thumb_ring = euclidean_distance(landmarks[THUMB_TIP][:2], landmarks[RING_TIP][:2]) / palm_size

        # Update HUD pinch meter to reflect nearest finger pinch
        min_pinch = min(d_thumb_idx, d_thumb_mid, d_thumb_ring)
        self.last_pinch_ratio = min_pinch * 0.25

        # -----------------------------------------------------------------
        # 0. Check Custom Scanned Poses (User Overrides)
        # -----------------------------------------------------------------
        if self.pose_manager and hasattr(self.pose_manager, "poses") and self.pose_manager.poses:
            for custom_name in (PAUSE, DRAG, DOUBLE_CLICK, MIDDLE_CLICK, ZOOM, SCROLL_V, SCROLL_H, BACK, FORWARD, LEFT_CLICK, RIGHT_CLICK):
                if custom_name in self.pose_manager.poses:
                    is_match, _ = self.pose_manager.match_pose(landmarks, custom_name)
                    if is_match:
                        if self._last_matched_custom_pose == custom_name:
                            self._custom_pose_frames += 1
                        else:
                            self._custom_pose_frames = 1
                            self._last_matched_custom_pose = custom_name

                        if self._custom_pose_frames >= self.confirmation_frames:
                            if custom_name == PAUSE:
                                self._paused = True
                                return PAUSE
                            elif custom_name == LEFT_CLICK:
                                if hand_speed <= self.max_click_speed and not self._left_click_triggered:
                                    self._left_click_triggered = True
                                    return LEFT_CLICK
                                return NONE
                            elif custom_name == RIGHT_CLICK:
                                if hand_speed <= self.max_click_speed and not self._right_click_triggered:
                                    self._right_click_triggered = True
                                    return RIGHT_CLICK
                                return NONE
                            return custom_name
                    elif self._last_matched_custom_pose == custom_name:
                        self._custom_pose_frames = 0
                        self._last_matched_custom_pose = None

        # -----------------------------------------------------------------
        # PRIORITY 1: CLOSED FIST -> PAUSE / OPEN PALM -> RESUME
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
        # PRIORITY 2: PINCH GESTURES (DRAG / DOUBLE_CLICK / MIDDLE_CLICK)
        # -----------------------------------------------------------------
        # 2a. DRAG: Thumb tip touches INDEX tip, middle/ring/little extended
        is_drag_pinch = (d_thumb_idx < self.pinch_activate) and (middle_ext and ring_ext and little_ext)
        if is_drag_pinch or (self._is_dragging and d_thumb_idx < self.pinch_release):
            self._is_dragging = True
            self._double_click_candidate_frames = 0
            self._middle_click_candidate_frames = 0
            return DRAG
        else:
            self._is_dragging = False

        # 2b. DOUBLE_CLICK: Thumb tip touches MIDDLE tip, index/ring/little extended
        is_double_pinch = (d_thumb_mid < self.pinch_activate) and (index_ext and ring_ext and little_ext)
        if is_double_pinch:
            self._double_click_candidate_frames += 1
            if self._double_click_candidate_frames >= self.confirmation_frames:
                if not self._double_click_pinch_triggered:
                    self._double_click_pinch_triggered = True
                    return DOUBLE_CLICK
                return NONE
        else:
            self._double_click_candidate_frames = 0
            if d_thumb_mid > self.pinch_release:
                self._double_click_pinch_triggered = False

        # 2c. MIDDLE_CLICK: Thumb tip touches RING tip, index/middle/little extended
        is_middle_pinch = (d_thumb_ring < self.pinch_activate) and (index_ext and middle_ext and little_ext)
        if is_middle_pinch:
            self._middle_click_candidate_frames += 1
            if self._middle_click_candidate_frames >= self.confirmation_frames:
                if not self._middle_click_pinch_triggered:
                    self._middle_click_pinch_triggered = True
                    return MIDDLE_CLICK
                return NONE
        else:
            self._middle_click_candidate_frames = 0
            if d_thumb_ring > self.pinch_release:
                self._middle_click_pinch_triggered = False

        # -----------------------------------------------------------------
        # PRIORITY 3: ZOOM (Index + Middle + Ring extended, Little folded)
        # -----------------------------------------------------------------
        three_fingers_zoom = index_ext and middle_ext and ring_ext and not little_ext
        if three_fingers_zoom:
            self._prev_scroll_pos = None
            self._prev_shaka_x = None
            return ZOOM

        # -----------------------------------------------------------------
        # PRIORITY 4: SCROLL (Index + Middle extended, Ring + Little folded)
        # -----------------------------------------------------------------
        two_fingers_scroll = index_ext and middle_ext and not ring_ext and not little_ext
        if two_fingers_scroll:
            current_pos = landmarks[INDEX_TIP][:2]
            scroll_dir = SCROLL_V
            if self._prev_scroll_pos is not None:
                dx = current_pos[0] - self._prev_scroll_pos[0]
                dy = current_pos[1] - self._prev_scroll_pos[1]
                if abs(dx) > abs(dy) and abs(dx) >= 3.0:
                    scroll_dir = SCROLL_H
            self._prev_scroll_pos = current_pos
            self._prev_shaka_x = None
            return scroll_dir
        else:
            self._prev_scroll_pos = None

        # -----------------------------------------------------------------
        # PRIORITY 5: BACK / FORWARD (Shaka Pose Swipe)
        # Thumb & Little extended, Index / Middle / Ring folded
        # -----------------------------------------------------------------
        shaka_pose = thumb_outward and little_ext and not index_ext and not middle_ext and not ring_ext
        if shaka_pose:
            wrist_x = landmarks[WRIST][0]
            action = NONE
            if self._prev_shaka_x is not None:
                dx = wrist_x - self._prev_shaka_x
                speed = abs(dx)
                if self._swipe_cooldown == 0 and speed >= self.swipe_speed_threshold:
                    self._swipe_cooldown = self.swipe_cooldown_frames
                    # Account for horizontally mirrored webcam:
                    # User moving to their left translates to dx < 0 on mirrored screen -> BACK
                    # User moving to their right translates to dx > 0 on mirrored screen -> FORWARD
                    action = BACK if dx < 0 else FORWARD
            self._prev_shaka_x = wrist_x
            if self._swipe_cooldown > 0:
                self._swipe_cooldown -= 1
            if action != NONE:
                return action
            return NONE
        else:
            self._prev_shaka_x = None
            if self._swipe_cooldown > 0:
                self._swipe_cooldown -= 1

        # -----------------------------------------------------------------
        # PRIORITY 6: LEFT CLICK / RIGHT CLICK (Four Fingers Extended)
        # -----------------------------------------------------------------
        # 6a. LEFT CLICK: Four fingers together, thumb folded inward
        is_left_click_pose = four_fingers_ext and fingers_together and thumb_inward
        if is_left_click_pose:
            self._right_candidate_frames = 0
            self._right_click_triggered = False
            self._left_candidate_frames += 1
            if self._left_candidate_frames >= self.click_stability_frames and hand_speed <= self.max_click_speed:
                if not self._left_click_triggered:
                    self._left_click_triggered = True
                    # Double click interval backwards compatibility
                    if self._double_click_pending and self._frames_since_last_left <= self.double_click_interval:
                        self._double_click_pending = False
                        self._frames_since_last_left = 0
                        return DOUBLE_CLICK
                    else:
                        self._double_click_pending = True
                        self._frames_since_last_left = 0
                        return LEFT_CLICK
                else:
                    return NONE
            else:
                return NONE
        else:
            self._left_candidate_frames = 0
            self._left_click_triggered = False

        # 6b. RIGHT CLICK: Four fingers together, thumb extended outward
        is_right_click_pose = four_fingers_ext and fingers_together and thumb_outward
        if is_right_click_pose:
            self._left_candidate_frames = 0
            self._left_click_triggered = False
            self._right_candidate_frames += 1
            if self._right_candidate_frames >= self.click_stability_frames and hand_speed <= self.max_click_speed:
                if not self._right_click_triggered:
                    self._right_click_triggered = True
                    return RIGHT_CLICK
                else:
                    return NONE
            else:
                return NONE
        else:
            self._right_candidate_frames = 0
            self._right_click_triggered = False

        # -----------------------------------------------------------------
        # PRIORITY 7: CURSOR_MOVE (Open Palm or Single Index Pointing)
        # -----------------------------------------------------------------
        if four_fingers_ext and thumb_outward:
            return CURSOR_MOVE

        if index_ext:
            return CURSOR_MOVE

        return NONE
