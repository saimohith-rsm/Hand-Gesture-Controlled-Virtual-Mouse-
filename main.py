"""main.py
Main entry point for High-Accuracy Hand Gesture Controlled Virtual Mouse.
Real-time webcam tracking, kinematics, convex hull, gesture recognition,
operating system mouse control, and live Custom Pose Calibration.
"""

import cv2
import numpy as np
import screeninfo
import time
import math
import os
from typing import Optional, Tuple, List

import config
import pyautogui
from hand_tracking import HandTracker
from gesture_recognition import (
    GestureRecognizer,
    CURSOR_MOVE,
    LEFT_CLICK,
    RIGHT_CLICK,
    DOUBLE_CLICK,
    DRAG,
    MIDDLE_CLICK,
    SCROLL_V,
    SCROLL_H,
    ZOOM,
    BACK,
    FORWARD,
    PAUSE,
    NONE,
    SCROLL,
)
from mouse_control import MouseController
from filters import OneEuroFilter
from utils import (
    map_to_screen,
    FPSCounter,
    ThreadedCamera,
    PositionHistory,
    check_lighting_and_enhance,
    load_calibration_region,
    save_calibration_region,
)
from convex_hull import get_hand_mask, find_contour, convex_hull_and_defects, draw_hull_and_defects
from pose_calibration import PoseManager

def draw_active_region_box(frame: np.ndarray, active_region: tuple) -> np.ndarray:
    """Draw a professional bounding box representing the active screen control area."""
    h, w = frame.shape[:2]
    fx1, fy1, fx2, fy2 = active_region
    x1, y1 = int(fx1 * w), int(fy1 * h)
    x2, y2 = int(fx2 * w), int(fy2 * h)

    color = (0, 255, 128)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Corner brackets for HUD aesthetic
    corner_len = 22
    thickness = 4
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

    cv2.putText(frame, "CONTROL REGION (Press 'C' for Pose Scanner)", (x1 + 6, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return frame

def draw_cheat_sheet(frame: np.ndarray) -> np.ndarray:
    """Render a clean, modern semi-transparent cheat sheet overlay of gestures and shortcuts."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    box_x1 = max(15, int(w * 0.04))
    box_y1 = max(15, int(h * 0.04))
    box_x2 = min(w - 15, int(w * 0.96))
    box_y2 = min(h - 15, int(h * 0.96))

    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (18, 18, 24), -1)
    frame = cv2.addWeighted(overlay, 0.88, frame, 0.12, 0)
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 200), 2)

    # Header
    cv2.putText(frame, "=== GESTURE & SHORTCUT CHEAT SHEET ===", (box_x1 + 20, box_y1 + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Press [H] or [ESC] to close", (box_x2 - 240, box_y1 + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

    col1_x = box_x1 + 20
    col2_x = box_x1 + int((box_x2 - box_x1) * 0.52)
    start_y = box_y1 + 68
    line_h = 24 if h < 600 else 28

    gestures_col = [
        ("GESTURE ACTIONS", (0, 255, 255), 2),
        ("Cursor Move : Open Palm (5 fingers extended)", (0, 255, 128), 1),
        ("Left Click  : 4 fingers together, thumb inward", (255, 255, 255), 1),
        ("Right Click : 4 fingers together, thumb outward", (255, 255, 255), 1),
        ("Double Click: Pinch Thumb + Middle tip", (255, 255, 255), 1),
        ("Middle Click: Pinch Thumb + Ring tip", (255, 255, 255), 1),
        ("Drag & Drop : Pinch Thumb + Index (hold/drop)", (255, 255, 255), 1),
        ("Scroll (V/H): Index + Middle (move up/down, L/R)", (255, 255, 255), 1),
        ("Zoom In/Out : Index + Middle + Ring (hand up/down)", (255, 255, 255), 1),
        ("Back/Forward: Shaka (Thumb+Pinky) swipe Left/Right", (255, 255, 255), 1),
        ("Pause/Resume: Fist to pause | Open palm to resume", (0, 200, 255), 1),
    ]

    shortcuts_col = [
        ("KEYBOARD SHORTCUTS", (0, 255, 255), 2),
        ("[M]     : Toggle Mode (Absolute vs Relative)", (255, 255, 255), 1),
        ("[P]     : Toggle Precision Mode (x0.4 speed)", (255, 255, 255), 1),
        ("[N]     : Cycle Active Monitor (Multi-screen)", (255, 255, 255), 1),
        ("[K]     : Calibrate Comfortable Reach Region", (255, 255, 255), 1),
        ("[C]     : Live Pose Scanner Studio (1-9 map)", (255, 255, 255), 1),
        ("[H]     : Toggle this Cheat Sheet overlay", (255, 255, 255), 1),
        ("[ESC/Q] : Quit Hand Gesture Virtual Mouse", (255, 255, 255), 1),
    ]

    y = start_y
    for text, color, thick in gestures_col:
        cv2.putText(frame, text, (col1_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, thick, cv2.LINE_AA)
        y += line_h

    y = start_y
    for text, color, thick in shortcuts_col:
        cv2.putText(frame, text, (col2_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, thick, cv2.LINE_AA)
        y += line_h

    return frame

def draw_hud(frame: np.ndarray,
             fps: float,
             gesture: str,
             pinch_ratio: float,
             pinch_threshold: float,
             is_paused: bool,
             control_mode: str,
             precision_mode: bool,
             is_low_light: bool,
             is_hand_lost: bool,
             active_monitor_str: str = "",
             calib_status: str = "") -> np.ndarray:
    """Render a clean, modern HUD dashboard overlay showing modes, status, and shortcuts."""
    h, w = frame.shape[:2]

    # Semi-transparent dashboard bar at the top via localized ROI blending
    roi = frame[0:55, 0:w]
    frame[0:55, 0:w] = (roi * 0.25).astype(np.uint8) + 15

    # 1. FPS display
    cv2.putText(frame, f"FPS: {fps:.0f}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    # 2. Control Mode badge (Absolute vs Relative)
    mode_color = (255, 180, 0) if control_mode == "relative" else (0, 220, 255)
    cv2.putText(frame, f"[{control_mode.upper()}]", (105, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, mode_color, 2, cv2.LINE_AA)

    # 3. Precision badge
    badge_x = 225
    if precision_mode:
        cv2.putText(frame, "[PRECISION]", (badge_x, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 128), 2, cv2.LINE_AA)
        badge_x += 125

    # 4. Monitor badge
    if active_monitor_str:
        cv2.putText(frame, f"[{active_monitor_str}]", (badge_x, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 220, 255), 2, cv2.LINE_AA)
        badge_x += 105

    # 5. Gesture Mode badge
    gesture_colors = {
        CURSOR_MOVE: (0, 255, 0),       # Green
        LEFT_CLICK: (0, 255, 255),      # Yellow
        RIGHT_CLICK: (255, 128, 0),     # Orange
        DOUBLE_CLICK: (255, 0, 255),    # Magenta
        MIDDLE_CLICK: (0, 200, 255),    # Cyan
        SCROLL_V: (255, 200, 0),        # Gold / Blue
        SCROLL_H: (255, 200, 0),
        SCROLL: (255, 200, 0),
        ZOOM: (200, 100, 255),          # Purple
        BACK: (100, 255, 200),          # Mint
        FORWARD: (100, 255, 200),
        DRAG: (0, 165, 255),            # Amber
        PAUSE: (0, 0, 255),             # Red
        NONE: (180, 180, 180),
    }
    color = (0, 0, 255) if (is_paused or is_hand_lost) else gesture_colors.get(gesture, (200, 200, 200))
    display_text = "HAND LOST" if is_hand_lost else ("PAUSED (FIST)" if is_paused else gesture.upper().replace("_", " "))

    status_x = badge_x + 10
    cv2.putText(frame, display_text, (status_x, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)

    # 6. Warning indicators: LOW LIGHT and HAND LOST
    meter_x, meter_y, meter_w, meter_h = w - 170, 20, 150, 14
    if is_low_light:
        cv2.putText(frame, "[LOW LIGHT]", (meter_x - 110, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)

    # 7. Live Pinch Meter
    cv2.rectangle(frame, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (80, 80, 80), 1)
    clamped_ratio = max(0.0, min(pinch_ratio, 0.12))
    fill_w = int((clamped_ratio / 0.12) * meter_w)
    meter_color = (0, 255, 0) if pinch_ratio > pinch_threshold else (0, 0, 255)
    cv2.rectangle(frame, (meter_x, meter_y), (meter_x + fill_w, meter_y + meter_h), meter_color, -1)
    cv2.putText(frame, "PINCH METER", (meter_x, meter_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    # 8. Bottom Shortcuts Footer Bar
    bar_y = h - 26
    footer_roi = frame[bar_y - 4:h, 0:w]
    frame[bar_y - 4:h, 0:w] = (footer_roi * 0.20).astype(np.uint8) + 10
    shortcuts_str = "[M] Mode | [P] Precision | [N] Monitor | [H] Help | [K] Reach | [C] Studio | [ESC/Q] Quit"
    cv2.putText(frame, shortcuts_str, (15, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)

    # Status toast notification
    if calib_status:
        cv2.putText(frame, calib_status, (15, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)

    # 9. Large center banner when PAUSED or HAND LOST
    if is_hand_lost:
        cv2.rectangle(frame, (w // 2 - 140, h // 2 - 35), (w // 2 + 140, h // 2 + 35), (0, 0, 0), -1)
        cv2.rectangle(frame, (w // 2 - 140, h // 2 - 35), (w // 2 + 140, h // 2 + 35), (0, 0, 255), 2)
        cv2.putText(frame, "HAND LOST", (w // 2 - 105, h // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
    elif is_paused:
        cv2.rectangle(frame, (w // 2 - 140, h // 2 - 35), (w // 2 + 140, h // 2 + 35), (0, 0, 0), -1)
        cv2.rectangle(frame, (w // 2 - 140, h // 2 - 35), (w // 2 + 140, h // 2 + 35), (0, 0, 255), 2)
        cv2.putText(frame, "PAUSED", (w // 2 - 75, h // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

    return frame


def draw_reach_calibration(frame: np.ndarray, step: int, pt1: Optional[Tuple[int, int]]) -> np.ndarray:
    """Render the comfortable reach corner calibration screen."""
    h, w = frame.shape[:2]
    roi = frame[50:h - 50, 50:w - 50]
    frame[50:h - 50, 50:w - 50] = (roi * 0.15).astype(np.uint8) + 12

    cv2.putText(frame, "=== COMFORTABLE REACH CALIBRATION ===", (70, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

    if step == 1:
        cv2.putText(frame, "Step 1 of 2: Move index finger to comfortable TOP-LEFT reach", (70, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Press [SPACE] or [ENTER] to lock TOP-LEFT corner.", (70, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 128), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Step 2 of 2: Move index finger to comfortable BOTTOM-RIGHT reach", (70, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Press [SPACE] or [ENTER] to lock BOTTOM-RIGHT corner & save.", (70, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 128), 2, cv2.LINE_AA)
        if pt1:
            cv2.circle(frame, pt1, 9, (0, 255, 255), 2)
            cv2.putText(frame, "Top-Left", (pt1[0] + 12, pt1[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, "Press [K] to cancel and keep current reach region.", (70, h - 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
    return frame

def draw_calibration_menu(frame: np.ndarray, pose_manager: PoseManager, status_msg: str) -> np.ndarray:
    """Render the Live Pose Scanning Studio overlay."""
    h, w = frame.shape[:2]

    # Dark translucent backdrop on the menu area
    roi = frame[30:h - 30, 30:w - 30]
    frame[30:h - 30, 30:w - 30] = (roi * 0.15).astype(np.uint8) + 12

    # Title
    cv2.putText(frame, "=== REAL-TIME GESTURE SCANNING STUDIO ===", (60, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Show your custom hand pose to camera and press a key to scan:", (60, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)

    col1 = [
        ("[1] Scan pose: LEFT CLICK", LEFT_CLICK in pose_manager.poses),
        ("[2] Scan pose: RIGHT CLICK", RIGHT_CLICK in pose_manager.poses),
        ("[3] Scan pose: DOUBLE CLICK", DOUBLE_CLICK in pose_manager.poses),
        ("[4] Scan pose: MIDDLE CLICK", MIDDLE_CLICK in pose_manager.poses),
        ("[5] Scan pose: SCROLL", (SCROLL_V in pose_manager.poses or SCROLL in pose_manager.poses)),
    ]
    col2 = [
        ("[6] Scan pose: ZOOM", ZOOM in pose_manager.poses),
        ("[7] Scan pose: BACK", BACK in pose_manager.poses),
        ("[8] Scan pose: FORWARD", FORWARD in pose_manager.poses),
        ("[9] Scan pose: PAUSE", PAUSE in pose_manager.poses),
        ("[S] SAVE & APPLY UPDATED POSES", False),
        ("[R] RESET TO FACTORY DEFAULTS", False),
        ("[C] RETURN TO VIRTUAL MOUSE", False),
    ]

    col1_x = 60
    col2_x = 60 + int((w - 120) * 0.50)
    line_h = 28

    y = 135
    for text, is_custom in col1:
        color = (0, 255, 128) if is_custom else (255, 255, 255)
        badge = " (CUSTOM)" if is_custom else ""
        cv2.putText(frame, text + badge, (col1_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2 if is_custom else 1, cv2.LINE_AA)
        y += line_h

    y = 135
    for text, is_custom in col2:
        color = (0, 255, 128) if is_custom else (255, 255, 255)
        badge = " (CUSTOM)" if is_custom else ""
        cv2.putText(frame, text + badge, (col2_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2 if is_custom else 1, cv2.LINE_AA)
        y += line_h

    if status_msg:
        cv2.rectangle(frame, (50, h - 75), (w - 50, h - 45), (0, 100, 0), -1)
        cv2.putText(frame, status_msg, (60, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

    return frame

def main():
    print("=" * 60)
    print("  HIGH-ACCURACY HAND GESTURE VIRTUAL MOUSE")
    print("  Press 'C' to scan and customize your hand poses!")
    print("  Press 'ESC' or 'Q' to quit anytime.")
    print("=" * 60)

    # Open threaded webcam (requests 1280x720, falls back to 640x480 if unavailable)
    cap = ThreadedCamera(
        src=config.CAMERA_INDEX,
        width=config.CAMERA_WIDTH,
        height=config.CAMERA_HEIGHT,
        fallback_width=config.CAMERA_FALLBACK_WIDTH,
        fallback_height=config.CAMERA_FALLBACK_HEIGHT,
        fps=config.CAMERA_FPS,
        buffer_size=config.CAMERA_BUFFER_SIZE,
        fourcc=config.CAMERA_FOURCC,
        use_dshow=config.CAMERA_USE_DSHOW,
    )

    if not cap.isOpened():
        print(f"Error: Could not open camera at index {config.CAMERA_INDEX}.")
        return

    # Initialize components
    try:
        hand_tracker = HandTracker(
            detection_confidence=config.DETECTION_CONFIDENCE,
            tracking_confidence=config.TRACKING_CONFIDENCE,
        )
    except RuntimeError as e:
        print(e)
        cap.release()
        return

    pose_manager = PoseManager()

    gesture_recognizer = GestureRecognizer(
        pinch_activate=config.PINCH_ACTIVATION_THRESHOLD,
        pinch_release=config.PINCH_RELEASE_THRESHOLD,
        confirmation_frames=config.GESTURE_CONFIRMATION_FRAMES,
        double_click_interval=config.DOUBLE_CLICK_MAX_INTERVAL,
        click_stability_frames=config.CLICK_STABILITY_FRAMES,
        max_click_speed=config.MAX_CLICK_SPEED_THRESHOLD,
        pose_manager=pose_manager,
    )

    mouse = MouseController(
        test_mode=False,
        click_cooldown_frames=config.CLICK_COOLDOWN_FRAMES,
    )

    cursor_filter = OneEuroFilter(
        freq=config.SMOOTHING_FREQ,
        min_cutoff=config.SMOOTHING_MIN_CUTOFF,
        beta=config.SMOOTHING_BETA,
        deadzone=config.DEADZONE_PIXELS,
    )

    fps_counter = FPSCounter()
    position_history = PositionHistory(maxlen=config.CLICK_FREEZE_FRAMES)

    # Load active region from calibration.json if present
    active_region = load_calibration_region(config.CALIBRATION_FILE, config.ACTIVE_REGION)
    first_run_calibration = not os.path.exists(config.CALIBRATION_FILE)
    reach_calib_mode = first_run_calibration
    reach_step = 1
    reach_pt1 = None

    control_mode = getattr(config, "DEFAULT_CONTROL_MODE", "absolute")
    precision_mode = False

    # Query monitor dimensions
    try:
        monitors = screeninfo.get_monitors()
        if not monitors:
            raise ValueError("No monitors detected")
    except Exception:
        class DummyMonitor:
            x = 0
            y = 0
            width = 1920
            height = 1080
        monitors = [DummyMonitor()]

    active_monitor_idx = 0
    current_mon = monitors[active_monitor_idx]
    screen_size = (current_mon.width, current_mon.height)
    monitor_offset = (current_mon.x, current_mon.y)

    cheat_sheet_mode = False
    prev_scroll_y = None
    prev_scroll_x = None
    prev_zoom_y = None
    prev_relative_anchor = None
    prev_anchor_pt = None
    calibration_mode = False
    calib_status = ""
    frame_count = 0
    cached_hull_data = None
    last_time = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if not cap.isOpened():
                    break
                key = cv2.waitKey(5) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    break
                continue

            frame_count += 1
            now = time.time()
            dt = (now - last_time) if last_time is not None else (1.0 / config.CAMERA_FPS)
            last_time = now

            # Lighting check and CLAHE contrast enhancement for low-light environments
            frame, is_low_light, _ = check_lighting_and_enhance(
                frame,
                threshold=config.LOW_LIGHT_THRESHOLD,
                enable_clahe=config.ENABLE_CLAHE,
                clip_limit=config.CLAHE_CLIP_LIMIT,
                grid_size=config.CLAHE_GRID_SIZE,
            )

            # Mirror webcam horizontally
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Process hand landmarks
            landmarks, annotated = hand_tracker.process(frame)
            fps = fps_counter.tick()

            gesture = NONE
            is_paused = False
            is_hand_lost = False

            # Check edge margin safety: auto-pause if hand is outside safe bounds
            if landmarks:
                wrist = landmarks[0]
                idx_tip = landmarks[8]
                margin = config.FRAME_EDGE_MARGIN
                if (wrist[0] < margin or wrist[0] > w - margin or
                    wrist[1] < margin or wrist[1] > h - margin or
                    idx_tip[0] < margin or idx_tip[0] > w - margin or
                    idx_tip[1] < margin or idx_tip[1] > h - margin):
                    is_hand_lost = True
            else:
                is_hand_lost = True

            # Dual-landmark stabilized index anchor (75% tip, 25% DIP)
            if landmarks:
                tip = landmarks[8]
                dip = landmarks[7]
                anchor_x = tip[0] * 0.75 + dip[0] * 0.25
                anchor_y = tip[1] * 0.75 + dip[1] * 0.25
                hand_speed = math.hypot(anchor_x - prev_anchor_pt[0], anchor_y - prev_anchor_pt[1]) if prev_anchor_pt else 0.0
                prev_anchor_pt = (anchor_x, anchor_y)
            else:
                anchor_x, anchor_y = 0.0, 0.0
                hand_speed = 0.0
                prev_anchor_pt = None

            if reach_calib_mode:
                # ----------------- First-Run / Reach Calibration Mode -----------------
                annotated = draw_reach_calibration(annotated, reach_step, reach_pt1)
                if landmarks:
                    cv2.circle(annotated, (int(anchor_x), int(anchor_y)), 10, (0, 255, 255), 2)
            elif calibration_mode:
                # ----------------- Gesture Scanning Studio Mode -----------------
                annotated = draw_calibration_menu(annotated, pose_manager, calib_status)
            else:
                # ----------------- Normal Virtual Mouse Mode ------------------
                if config.SHOW_ACTIVE_BOX and control_mode == "absolute":
                    annotated = draw_active_region_box(annotated, active_region)

                if landmarks and not is_hand_lost:
                    # Convex Hull visualization (throttled to every 3rd frame to conserve CPU/GPU)
                    if config.SHOW_CONVEX_HULL:
                        if frame_count % 3 == 0 or cached_hull_data is None:
                            lm_2d = [(p[0], p[1]) for p in landmarks]
                            mask = get_hand_mask(annotated, lm_2d)
                            contour = find_contour(mask)
                            if contour is not None:
                                hull, defects = convex_hull_and_defects(contour)
                                if hull.size > 0:
                                    cached_hull_data = (contour, hull, defects)
                                else:
                                    cached_hull_data = None
                            else:
                                cached_hull_data = None

                        if cached_hull_data is not None:
                            c_contour, c_hull, c_defects = cached_hull_data
                            annotated = draw_hull_and_defects(annotated, c_contour, c_hull, c_defects)

                    # Update filter parameters for Precision Mode
                    if precision_mode:
                        cursor_filter.min_cutoff = config.PRECISION_MIN_CUTOFF
                        cursor_filter.beta = config.PRECISION_BETA
                    else:
                        cursor_filter.min_cutoff = config.SMOOTHING_MIN_CUTOFF
                        cursor_filter.beta = config.SMOOTHING_BETA

                    smooth_pt = cursor_filter((anchor_x, anchor_y), dt=dt)

                    # Gesture recognition with handedness and velocity suppression
                    gesture = gesture_recognizer.recognize(
                        landmarks,
                        frame.shape,
                        handedness=hand_tracker.last_handedness,
                        hand_speed=hand_speed,
                    )
                    is_paused = (gesture == PAUSE)

                    # Determine screen coordinates
                    if control_mode == "absolute":
                        screen_pt = map_to_screen(
                            smooth_pt,
                            (frame.shape[1], frame.shape[0]),
                            screen_size,
                            active_region,
                            monitor_offset=monitor_offset,
                        )
                        position_history.append(screen_pt)
                    else:
                        screen_pt = mouse.get_position()

                    # Reticle on tracking index finger
                    reticle_color = (0, 255, 255) if gesture == DRAG else (0, 255, 0)
                    cv2.circle(annotated, (int(smooth_pt[0]), int(smooth_pt[1])), 9, reticle_color, 2)
                    cv2.circle(annotated, (int(smooth_pt[0]), int(smooth_pt[1])), 3, (0, 0, 255), -1)

                    # Mouse Control
                    if not is_paused:
                        if gesture == CURSOR_MOVE:
                            if control_mode == "relative":
                                if prev_relative_anchor is not None:
                                    dx = (smooth_pt[0] - prev_relative_anchor[0]) * config.RELATIVE_SENSITIVITY
                                    dy = (smooth_pt[1] - prev_relative_anchor[1]) * config.RELATIVE_SENSITIVITY
                                    if precision_mode:
                                        dx *= config.PRECISION_SPEED_MULTIPLIER
                                        dy *= config.PRECISION_SPEED_MULTIPLIER
                                    screen_pt = mouse.move_rel(int(dx), int(dy), screen_size)
                                    position_history.append(screen_pt)
                                prev_relative_anchor = smooth_pt
                            else:
                                mouse.move_to(*screen_pt)

                            mouse.drag_end()
                            mouse.release_ctrl()
                            prev_scroll_y = None
                            prev_scroll_x = None
                            prev_zoom_y = None

                        elif gesture == LEFT_CLICK:
                            frozen_pt = position_history.get_frozen_point()
                            if frozen_pt is not None:
                                mouse.move_to(*frozen_pt)
                            mouse.left_click()
                            mouse.release_ctrl()
                            prev_scroll_y = None
                            prev_scroll_x = None
                            prev_zoom_y = None

                        elif gesture == RIGHT_CLICK:
                            frozen_pt = position_history.get_frozen_point()
                            if frozen_pt is not None:
                                mouse.move_to(*frozen_pt)
                            mouse.right_click()
                            mouse.release_ctrl()
                            prev_scroll_y = None
                            prev_scroll_x = None
                            prev_zoom_y = None

                        elif gesture == DOUBLE_CLICK:
                            frozen_pt = position_history.get_frozen_point()
                            if frozen_pt is not None:
                                mouse.move_to(*frozen_pt)
                            mouse.double_click()
                            mouse.release_ctrl()
                            prev_scroll_y = None
                            prev_scroll_x = None
                            prev_zoom_y = None

                        elif gesture == MIDDLE_CLICK:
                            frozen_pt = position_history.get_frozen_point()
                            if frozen_pt is not None:
                                mouse.move_to(*frozen_pt)
                            mouse.middle_click()
                            mouse.release_ctrl()
                            prev_scroll_y = None
                            prev_scroll_x = None
                            prev_zoom_y = None

                        elif gesture == DRAG:
                            if control_mode == "relative":
                                if prev_relative_anchor is not None:
                                    dx = (smooth_pt[0] - prev_relative_anchor[0]) * config.RELATIVE_SENSITIVITY
                                    dy = (smooth_pt[1] - prev_relative_anchor[1]) * config.RELATIVE_SENSITIVITY
                                    if precision_mode:
                                        dx *= config.PRECISION_SPEED_MULTIPLIER
                                        dy *= config.PRECISION_SPEED_MULTIPLIER
                                    screen_pt = mouse.move_rel(int(dx), int(dy), screen_size)
                                    position_history.append(screen_pt)
                                prev_relative_anchor = smooth_pt
                            else:
                                mouse.move_to(*screen_pt)

                            mouse.drag_start()
                            mouse.release_ctrl()
                            prev_scroll_y = None
                            prev_scroll_x = None
                            prev_zoom_y = None

                        elif gesture in (SCROLL_V, SCROLL):
                            mouse.drag_end()
                            mouse.release_ctrl()
                            current_scroll_y = landmarks[8][1]
                            if prev_scroll_y is not None:
                                dy_px = prev_scroll_y - current_scroll_y
                                if abs(dy_px) >= config.SCROLL_DEADZONE:
                                    scroll_amount = int(dy_px * config.SCROLL_UNITS_PER_PIXEL)
                                    if precision_mode:
                                        scroll_amount = int(scroll_amount * config.PRECISION_SPEED_MULTIPLIER)
                                    mouse.scroll(scroll_amount)
                            prev_scroll_y = current_scroll_y
                            prev_scroll_x = None
                            prev_zoom_y = None

                        elif gesture == SCROLL_H:
                            mouse.drag_end()
                            mouse.release_ctrl()
                            current_scroll_x = landmarks[8][0]
                            if prev_scroll_x is not None:
                                dx_px = current_scroll_x - prev_scroll_x
                                if abs(dx_px) >= config.SCROLL_H_DEADZONE:
                                    scroll_amount = int(dx_px * config.SCROLL_H_UNITS_PER_PIXEL)
                                    if precision_mode:
                                        scroll_amount = int(scroll_amount * config.PRECISION_SPEED_MULTIPLIER)
                                    mouse.hscroll(scroll_amount)
                            prev_scroll_x = current_scroll_x
                            prev_scroll_y = None
                            prev_zoom_y = None

                        elif gesture == ZOOM:
                            mouse.drag_end()
                            current_zoom_y = landmarks[8][1]
                            if prev_zoom_y is not None:
                                dy_px = prev_zoom_y - current_zoom_y
                                if abs(dy_px) >= config.ZOOM_DEADZONE:
                                    zoom_amount = int(dy_px * config.ZOOM_UNITS_PER_PIXEL)
                                    if precision_mode:
                                        zoom_amount = max(1, int(zoom_amount * config.PRECISION_SPEED_MULTIPLIER)) if zoom_amount > 0 else min(-1, int(zoom_amount * config.PRECISION_SPEED_MULTIPLIER))
                                    mouse.zoom(zoom_amount)
                            prev_zoom_y = current_zoom_y
                            prev_scroll_y = None
                            prev_scroll_x = None

                        elif gesture == BACK:
                            mouse.drag_end()
                            mouse.release_ctrl()
                            mouse.back()
                            prev_scroll_y = None
                            prev_scroll_x = None
                            prev_zoom_y = None

                        elif gesture == FORWARD:
                            mouse.drag_end()
                            mouse.release_ctrl()
                            mouse.forward()
                            prev_scroll_y = None
                            prev_scroll_x = None
                            prev_zoom_y = None
                    else:
                        # Fist / Paused: lifting hand allows repositioning without moving cursor
                        mouse.drag_end()
                        mouse.release_ctrl()
                        prev_relative_anchor = None
                        prev_scroll_y = None
                        prev_scroll_x = None
                        prev_zoom_y = None
                else:
                    # Hand lost or outside edge bounds: auto-pause actions
                    cursor_filter.reset()
                    position_history.clear()
                    cached_hull_data = None
                    prev_relative_anchor = None
                    mouse.drag_end()
                    mouse.release_ctrl()
                    prev_scroll_y = None
                    prev_scroll_x = None
                    prev_zoom_y = None

                # Render modern HUD
                active_monitor_str = f"MON {active_monitor_idx + 1}/{len(monitors)}"
                annotated = draw_hud(
                    annotated,
                    fps=fps,
                    gesture=gesture,
                    pinch_ratio=gesture_recognizer.last_pinch_ratio,
                    pinch_threshold=config.PINCH_ACTIVATION_THRESHOLD,
                    is_paused=is_paused,
                    control_mode=control_mode,
                    precision_mode=precision_mode,
                    is_low_light=is_low_light,
                    is_hand_lost=is_hand_lost,
                    active_monitor_str=active_monitor_str,
                    calib_status=calib_status,
                )

                if cheat_sheet_mode:
                    annotated = draw_cheat_sheet(annotated)

            cv2.imshow("Hand Gesture Virtual Mouse", annotated)
            key = cv2.waitKey(1) & 0xFF

            # Global hotkeys
            if key in (27, ord('q'), ord('Q')):
                if cheat_sheet_mode:
                    cheat_sheet_mode = False
                else:
                    break
            elif key in (ord('h'), ord('H')):
                cheat_sheet_mode = not cheat_sheet_mode
                calib_status = "Cheat Sheet: ON" if cheat_sheet_mode else "Cheat Sheet: OFF"
            elif key in (ord('n'), ord('N')):
                if monitors:
                    active_monitor_idx = (active_monitor_idx + 1) % len(monitors)
                    current_mon = monitors[active_monitor_idx]
                    screen_size = (current_mon.width, current_mon.height)
                    monitor_offset = (current_mon.x, current_mon.y)
                    calib_status = f"Active Screen: Monitor {active_monitor_idx + 1}/{len(monitors)} [{screen_size[0]}x{screen_size[1]}]"
            elif key in (ord('m'), ord('M')):
                control_mode = "relative" if control_mode == "absolute" else "absolute"
                calib_status = f"Mode changed to: {control_mode.upper()}"
                prev_relative_anchor = None
            elif key in (ord('p'), ord('P')):
                precision_mode = not precision_mode
                calib_status = f"Precision Mode: {'ON (x0.4 speed)' if precision_mode else 'OFF'}"
            elif key in (ord('k'), ord('K')):
                reach_calib_mode = not reach_calib_mode
                reach_step = 1
                reach_pt1 = None
                calib_status = "Entered Reach Calibration" if reach_calib_mode else "Back to Mouse Control"
            elif reach_calib_mode:
                if key in (32, 13) and landmarks:  # SPACE or ENTER
                    if reach_step == 1:
                        reach_pt1 = (int(anchor_x), int(anchor_y))
                        reach_step = 2
                    elif reach_step == 2 and reach_pt1:
                        reach_pt2 = (int(anchor_x), int(anchor_y))
                        x1 = max(0.02, min(reach_pt1[0], reach_pt2[0]) / float(w))
                        y1 = max(0.02, min(reach_pt1[1], reach_pt2[1]) / float(h))
                        x2 = min(0.98, max(reach_pt1[0], reach_pt2[0]) / float(w))
                        y2 = min(0.98, max(reach_pt1[1], reach_pt2[1]) / float(h))
                        if (x2 - x1) >= 0.08 and (y2 - y1) >= 0.08:
                            active_region = (round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3))
                            save_calibration_region(active_region, config.CALIBRATION_FILE)
                            calib_status = f"Saved reach region: {active_region}"
                        reach_calib_mode = False
                        reach_step = 1
                        reach_pt1 = None
            elif key in (ord('c'), ord('C')):
                calibration_mode = not calibration_mode
                calib_status = "Entered Gesture Studio" if calibration_mode else "Back to Mouse Control"
            elif calibration_mode:
                if key == ord('1') and landmarks:
                    pose_manager.capture_pose(LEFT_CLICK, landmarks)
                    calib_status = "Captured pose for [LEFT CLICK]! Press 'S' to save."
                elif key == ord('2') and landmarks:
                    pose_manager.capture_pose(RIGHT_CLICK, landmarks)
                    calib_status = "Captured pose for [RIGHT CLICK]! Press 'S' to save."
                elif key == ord('3') and landmarks:
                    pose_manager.capture_pose(DOUBLE_CLICK, landmarks)
                    calib_status = "Captured pose for [DOUBLE CLICK]! Press 'S' to save."
                elif key == ord('4') and landmarks:
                    pose_manager.capture_pose(MIDDLE_CLICK, landmarks)
                    calib_status = "Captured pose for [MIDDLE CLICK]! Press 'S' to save."
                elif key == ord('5') and landmarks:
                    pose_manager.capture_pose(SCROLL_V, landmarks)
                    calib_status = "Captured pose for [SCROLL]! Press 'S' to save."
                elif key == ord('6') and landmarks:
                    pose_manager.capture_pose(ZOOM, landmarks)
                    calib_status = "Captured pose for [ZOOM]! Press 'S' to save."
                elif key == ord('7') and landmarks:
                    pose_manager.capture_pose(BACK, landmarks)
                    calib_status = "Captured pose for [BACK]! Press 'S' to save."
                elif key == ord('8') and landmarks:
                    pose_manager.capture_pose(FORWARD, landmarks)
                    calib_status = "Captured pose for [FORWARD]! Press 'S' to save."
                elif key == ord('9') and landmarks:
                    pose_manager.capture_pose(PAUSE, landmarks)
                    calib_status = "Captured pose for [PAUSE]! Press 'S' to save."
                elif key in (ord('s'), ord('S')):
                    pose_manager.save_poses()
                    calib_status = "SUCCESS: Updated poses saved and active! Press 'C' to use."
                elif key in (ord('r'), ord('R')):
                    pose_manager.poses.clear()
                    pose_manager.save_poses()
                    calib_status = "RESET: Restored default factory gesture poses."

            mouse.update_frame_counter()

    except pyautogui.FailSafeException:
        print("\n[FAILSAFE] PyAutoGUI FailSafe triggered. Exiting cleanly.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_tracker.close()
        mouse.drag_end()
        mouse.release_ctrl()
        print("\nVirtual Mouse stopped safely.")
        print("\nVirtual Mouse stopped safely.")

if __name__ == "__main__":
    main()
