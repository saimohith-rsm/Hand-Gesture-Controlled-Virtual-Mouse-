"""main.py
Main entry point for High-Accuracy Hand Gesture Controlled Virtual Mouse.
Real-time webcam tracking, kinematics, convex hull, gesture recognition,
operating system mouse control, and live Custom Pose Calibration.
"""

import cv2
import numpy as np
import screeninfo

import config
from hand_tracking import HandTracker
from gesture_recognition import (
    GestureRecognizer,
    CURSOR_MOVE,
    LEFT_CLICK,
    RIGHT_CLICK,
    DOUBLE_CLICK,
    SCROLL,
    DRAG,
    PAUSE,
    NONE,
)
from mouse_control import MouseController
from filters import OneEuroFilter
from utils import map_to_screen, FPSCounter
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

def draw_hud(frame: np.ndarray,
             fps: float,
             gesture: str,
             pinch_ratio: float,
             pinch_threshold: float,
             is_paused: bool,
             calib_status: str = "") -> np.ndarray:
    """Render a clean, modern HUD dashboard overlay."""
    h, w = frame.shape[:2]

    # Semi-transparent dashboard bar at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

    # 1. FPS display
    cv2.putText(frame, f"FPS: {fps:.0f}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    # 2. Gesture Mode badge
    gesture_colors = {
        CURSOR_MOVE: (0, 255, 0),     # Green
        LEFT_CLICK: (0, 255, 255),    # Yellow
        RIGHT_CLICK: (255, 128, 0),   # Orange
        DOUBLE_CLICK: (255, 0, 255),  # Magenta
        SCROLL: (255, 200, 0),        # Cyan/Blue
        DRAG: (0, 165, 255),          # Amber
        PAUSE: (0, 0, 255),           # Red
        NONE: (180, 180, 180),
    }
    color = (0, 0, 255) if is_paused else gesture_colors.get(gesture, (200, 200, 200))
    display_text = "PAUSED (FIST)" if is_paused else gesture.upper().replace("_", " ")

    cv2.putText(frame, "MODE:", (125, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, display_text, (190, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    # 3. Live Pinch Meter
    meter_x, meter_y, meter_w, meter_h = w - 180, 20, 150, 14
    cv2.rectangle(frame, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (80, 80, 80), 1)

    clamped_ratio = max(0.0, min(pinch_ratio, 0.12))
    fill_w = int((clamped_ratio / 0.12) * meter_w)
    meter_color = (0, 255, 0) if pinch_ratio > pinch_threshold else (0, 0, 255)
    cv2.rectangle(frame, (meter_x, meter_y), (meter_x + fill_w, meter_y + meter_h), meter_color, -1)
    cv2.putText(frame, "PINCH METER", (meter_x, meter_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    # Status toast notification
    if calib_status:
        cv2.putText(frame, calib_status, (15, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    # 4. Large center banner when PAUSED
    if is_paused:
        cv2.rectangle(frame, (w // 2 - 140, h // 2 - 35), (w // 2 + 140, h // 2 + 35), (0, 0, 0), -1)
        cv2.rectangle(frame, (w // 2 - 140, h // 2 - 35), (w // 2 + 140, h // 2 + 35), (0, 0, 255), 2)
        cv2.putText(frame, "PAUSED", (w // 2 - 75, h // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

    return frame

def draw_calibration_menu(frame: np.ndarray, pose_manager: PoseManager, status_msg: str) -> np.ndarray:
    """Render the Live Pose Scanning Studio overlay."""
    h, w = frame.shape[:2]

    # Dark translucent backdrop
    overlay = frame.copy()
    cv2.rectangle(overlay, (40, 40), (w - 40, h - 40), (15, 15, 15), -1)
    frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    # Title
    cv2.putText(frame, "=== REAL-TIME GESTURE SCANNING STUDIO ===", (70, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Show your custom hand pose to camera and press a key to scan:", (70, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

    # Instructions
    options = [
        ("[1] Scan current pose for: LEFT CLICK", LEFT_CLICK in pose_manager.poses),
        ("[2] Scan current pose for: RIGHT CLICK", RIGHT_CLICK in pose_manager.poses),
        ("[3] Scan current pose for: SCROLL", SCROLL in pose_manager.poses),
        ("[4] Scan current pose for: PAUSE", PAUSE in pose_manager.poses),
        ("[S] SAVE & APPLY UPDATED POSES", False),
        ("[R] RESET TO FACTORY DEFAULT POSES", False),
        ("[C] RETURN TO VIRTUAL MOUSE", False),
    ]

    y = 160
    for text, is_custom in options:
        color = (0, 255, 128) if is_custom else (255, 255, 255)
        badge = " (CUSTOMIZED)" if is_custom else ""
        cv2.putText(frame, text + badge, (70, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2 if is_custom else 1, cv2.LINE_AA)
        y += 35

    if status_msg:
        cv2.rectangle(frame, (60, h - 90), (w - 60, h - 55), (0, 100, 0), -1)
        cv2.putText(frame, status_msg, (70, h - 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    return frame

def main():
    print("=" * 60)
    print("  HIGH-ACCURACY HAND GESTURE VIRTUAL MOUSE")
    print("  Press 'C' to scan and customize your hand poses!")
    print("  Press 'ESC' or 'Q' to quit anytime.")
    print("=" * 60)

    # Open webcam
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    # Initialize components
    hand_tracker = HandTracker(
        detection_confidence=config.DETECTION_CONFIDENCE,
        tracking_confidence=config.TRACKING_CONFIDENCE,
    )

    pose_manager = PoseManager()

    gesture_recognizer = GestureRecognizer(
        pinch_activate=config.PINCH_ACTIVATION_THRESHOLD,
        pinch_release=config.PINCH_RELEASE_THRESHOLD,
        confirmation_frames=config.GESTURE_CONFIRMATION_FRAMES,
        double_click_interval=config.DOUBLE_CLICK_MAX_INTERVAL,
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

    # Query monitor dimensions
    try:
        monitors = screeninfo.get_monitors()
        monitor = monitors[0]
        screen_size = (monitor.width, monitor.height)
    except Exception:
        screen_size = (1920, 1080)

    prev_scroll_y = None
    calibration_mode = False
    calib_status = ""

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror webcam horizontally
            frame = cv2.flip(frame, 1)

            # Process hand landmarks
            landmarks, annotated = hand_tracker.process(frame)
            fps = fps_counter.tick()

            gesture = NONE
            is_paused = False

            if calibration_mode:
                # ----------------- Calibration / Scanning Mode -----------------
                annotated = draw_calibration_menu(annotated, pose_manager, calib_status)
            else:
                # ----------------- Normal Virtual Mouse Mode ------------------
                if config.SHOW_ACTIVE_BOX:
                    annotated = draw_active_region_box(annotated, config.ACTIVE_REGION)

                if landmarks:
                    # Convex Hull visualization
                    if config.SHOW_CONVEX_HULL:
                        lm_2d = [(p[0], p[1]) for p in landmarks]
                        mask = get_hand_mask(annotated, lm_2d)
                        contour = find_contour(mask)
                        if contour is not None:
                            hull, defects = convex_hull_and_defects(contour)
                            if hull.size > 0:
                                annotated = draw_hull_and_defects(annotated, contour, hull, defects)

                    # Gesture recognition
                    gesture = gesture_recognizer.recognize(landmarks, frame.shape)
                    is_paused = (gesture == PAUSE)

                    # Dual-landmark stabilized index anchor (75% tip, 25% DIP)
                    tip = landmarks[8]
                    dip = landmarks[7]
                    anchor_x = tip[0] * 0.75 + dip[0] * 0.25
                    anchor_y = tip[1] * 0.75 + dip[1] * 0.25

                    smooth_pt = cursor_filter((anchor_x, anchor_y))
                    screen_pt = map_to_screen(
                        smooth_pt,
                        (frame.shape[1], frame.shape[0]),
                        screen_size,
                        config.ACTIVE_REGION,
                    )

                    # Reticle on tracking index finger
                    reticle_color = (0, 255, 255) if gesture == DRAG else (0, 255, 0)
                    cv2.circle(annotated, (int(smooth_pt[0]), int(smooth_pt[1])), 9, reticle_color, 2)
                    cv2.circle(annotated, (int(smooth_pt[0]), int(smooth_pt[1])), 3, (0, 0, 255), -1)

                    # Mouse Control
                    if not is_paused:
                        if gesture == CURSOR_MOVE:
                            mouse.move_to(*screen_pt)
                            mouse.drag_end()
                            prev_scroll_y = None

                        elif gesture == LEFT_CLICK:
                            mouse.left_click()
                            prev_scroll_y = None

                        elif gesture == RIGHT_CLICK:
                            mouse.right_click()
                            prev_scroll_y = None

                        elif gesture == DOUBLE_CLICK:
                            mouse.double_click()
                            prev_scroll_y = None

                        elif gesture == DRAG:
                            mouse.move_to(*screen_pt)
                            mouse.drag_start()
                            prev_scroll_y = None

                        elif gesture == SCROLL:
                            mouse.drag_end()
                            current_scroll_y = np.mean([p[1] for p in landmarks])
                            if prev_scroll_y is not None:
                                dy_px = prev_scroll_y - current_scroll_y
                                if abs(dy_px) >= config.SCROLL_DEADZONE:
                                    scroll_amount = int(dy_px * config.SCROLL_SPEED_MULTIPLIER)
                                    mouse.scroll(scroll_amount)
                            prev_scroll_y = current_scroll_y
                    else:
                        mouse.drag_end()
                        prev_scroll_y = None
                else:
                    mouse.drag_end()
                    prev_scroll_y = None

                # Render HUD
                annotated = draw_hud(
                    annotated,
                    fps=fps,
                    gesture=gesture,
                    pinch_ratio=gesture_recognizer.last_pinch_ratio,
                    pinch_threshold=config.PINCH_ACTIVATION_THRESHOLD,
                    is_paused=is_paused,
                    calib_status=calib_status,
                )

            cv2.imshow("Hand Gesture Virtual Mouse", annotated)
            key = cv2.waitKey(1) & 0xFF

            # Global hotkeys
            if key in (27, ord('q'), ord('Q')):
                break
            elif key in (ord('c'), ord('C')):
                calibration_mode = not calibration_mode
                calib_status = "Entered Calibration Studio" if calibration_mode else "Back to Mouse Control"
            elif calibration_mode:
                if key == ord('1') and landmarks:
                    pose_manager.capture_pose(LEFT_CLICK, landmarks)
                    calib_status = "Captured pose for [LEFT CLICK]! Press 'S' to save."
                elif key == ord('2') and landmarks:
                    pose_manager.capture_pose(RIGHT_CLICK, landmarks)
                    calib_status = "Captured pose for [RIGHT CLICK]! Press 'S' to save."
                elif key == ord('3') and landmarks:
                    pose_manager.capture_pose(SCROLL, landmarks)
                    calib_status = "Captured pose for [SCROLL]! Press 'S' to save."
                elif key == ord('4') and landmarks:
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

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_tracker.close()
        mouse.drag_end()
        print("\nVirtual Mouse stopped safely.")

if __name__ == "__main__":
    main()
