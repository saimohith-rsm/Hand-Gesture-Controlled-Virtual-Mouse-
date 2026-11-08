"""test_gestures.py
Tests for GestureRecognizer:
1. OPEN PALM (all 5 fingers extended) -> CURSOR_MOVE (never conflicts with scroll).
2. CUSTOM LEFT CLICK POSE (4 fingers together, thumb folded inward) -> LEFT_CLICK.
3. CUSTOM RIGHT CLICK POSE (4 fingers together, thumb extended outward) -> RIGHT_CLICK.
4. SCROLL -> Index + Middle extended, Ring + Little folded ("two fingers up").
5. CLOSED FIST -> PAUSE. Open palm -> RESUME.
6. DRAG -> Thumb-index pinch held.
7. DOUBLE_CLICK -> Two left clicks within interval.
8. CUSTOM POSES -> Custom pose in PoseManager matches and triggers gesture.
"""

from gesture_recognition import (
    GestureRecognizer,
    CURSOR_MOVE,
    LEFT_CLICK,
    RIGHT_CLICK,
    DOUBLE_CLICK,
    MIDDLE_CLICK,
    SCROLL,
    SCROLL_V,
    SCROLL_H,
    ZOOM,
    BACK,
    FORWARD,
    DRAG,
    PAUSE,
    NONE,
)
from pose_calibration import PoseManager

def make_hand(four_together=False, thumb_state="outward", four_extended=True, two_fingers_only=False):
    """Generate 21-landmark dummy hand with specific configuration.
    Palm size = 100 (Wrist at (100, 200), Middle MCP at (100, 100)).
    """
    lm = [(100, 200, 0.0) for _ in range(21)]
    lm[0] = (100, 200, 0.0)  # Wrist
    lm[9] = (100, 100, 0.0)  # Middle MCP

    if two_fingers_only:
        # Index & Middle extended, Ring & Little folded
        lm[8] = (80, 50, 0.0)    # Index tip
        lm[6] = (80, 120, 0.0)   # Index pip
        lm[12] = (120, 50, 0.0)  # Middle tip
        lm[10] = (120, 120, 0.0) # Middle pip
        lm[16] = (150, 160, 0.0) # Ring tip (folded)
        lm[14] = (150, 120, 0.0) # Ring pip
        lm[20] = (180, 160, 0.0) # Little tip (folded)
        lm[18] = (180, 120, 0.0) # Little pip
    else:
        tip_y = 50 if four_extended else 160
        pip_y = 120
        if four_together:
            tips_x = {8: 90, 12: 100, 16: 110, 20: 120}
            pips_x = {6: 90, 10: 100, 14: 110, 18: 120}
        else:
            tips_x = {8: 60, 12: 100, 16: 140, 20: 180}
            pips_x = {6: 60, 10: 100, 14: 140, 18: 180}

        for tip, x in tips_x.items():
            lm[tip] = (x, tip_y, 0.0)
        for pip, x in pips_x.items():
            lm[pip] = (x, pip_y, 0.0)

    # Index MCP for thumb reference
    lm[5] = (90, 140, 0.0)

    # Thumb configuration
    if thumb_state == "inward":
        lm[4] = (95, 125, 0.0)
        lm[3] = (95, 150, 0.0)
    elif thumb_state == "outward":
        lm[4] = (10, 80, 0.0)
        lm[3] = (25, 140, 0.0)
    elif thumb_state == "pinch":
        # Thumb tip pinched close to index tip
        lm[4] = (lm[8][0] + 2, lm[8][1] + 2, 0.0)
        lm[3] = (lm[8][0] + 10, lm[8][1] + 20, 0.0)
    else: # folded
        lm[4] = (80, 170, 0.0)
        lm[3] = (80, 180, 0.0)

    return lm

def dummy_landmarks(pairs=None):
    """Backwards-compatible dummy landmarks generator."""
    lm = make_hand(four_together=False, thumb_state="outward", four_extended=True)
    if pairs:
        for idx, (x, y) in pairs.items():
            lm[idx] = (x, y, 0.0)
    return lm


def test_gesture_1_open_palm():
    """Gesture 1: Open Palm (all 5 fingers extended) -> Move cursor, never scrolls."""
    recognizer = GestureRecognizer()
    lm = make_hand(four_together=False, thumb_state="outward", four_extended=True)
    gesture = recognizer.recognize(lm, (480, 640))
    assert gesture == CURSOR_MOVE


def test_gesture_2_custom_left_click():
    """Gesture 2: Four fingers together, thumb folded inward -> Left click."""
    recognizer = GestureRecognizer()
    lm = make_hand(four_together=True, thumb_state="inward", four_extended=True)
    gesture = recognizer.recognize(lm, (480, 640))
    assert gesture == LEFT_CLICK


def test_gesture_2_left_click_debounce():
    """Verify click debounce: holding the pose does not repeatedly fire clicks."""
    recognizer = GestureRecognizer()
    lm = make_hand(four_together=True, thumb_state="inward", four_extended=True)
    g1 = recognizer.recognize(lm, (480, 640))
    assert g1 == LEFT_CLICK
    g2 = recognizer.recognize(lm, (480, 640))
    assert g2 == NONE


def test_gesture_3_custom_right_click():
    """Gesture 3: Four fingers together, thumb extended outward -> Right click."""
    recognizer = GestureRecognizer()
    lm = make_hand(four_together=True, thumb_state="outward", four_extended=True)
    gesture = recognizer.recognize(lm, (480, 640))
    assert gesture == RIGHT_CLICK


def test_gesture_3_right_click_debounce():
    """Verify right click debounce: holding the pose does not repeat clicks."""
    recognizer = GestureRecognizer()
    lm = make_hand(four_together=True, thumb_state="outward", four_extended=True)
    g1 = recognizer.recognize(lm, (480, 640))
    assert g1 == RIGHT_CLICK
    g2 = recognizer.recognize(lm, (480, 640))
    assert g2 == NONE


def test_gesture_4_two_fingers_scroll():
    """Gesture 4: Index + Middle extended, Ring + Little folded -> Scroll."""
    recognizer = GestureRecognizer()
    lm = make_hand(two_fingers_only=True)
    gesture = recognizer.recognize(lm, (480, 640))
    assert gesture == SCROLL


def test_gesture_5_closed_fist_pause_and_open_palm_resume():
    """Gesture 5: Closed fist -> Pause tracking. Open palm -> Resume tracking."""
    recognizer = GestureRecognizer()
    lm_fist = make_hand(four_together=True, thumb_state="folded", four_extended=False)
    assert recognizer.recognize(lm_fist, (480, 640)) == PAUSE
    assert recognizer.recognize(lm_fist, (480, 640)) == PAUSE

    # Open palm resumes
    lm_open = make_hand(four_together=False, thumb_state="outward", four_extended=True)
    assert recognizer.recognize(lm_open, (480, 640)) == CURSOR_MOVE


def test_drag_detection():
    """Thumb-index pinch held past confirmation frames -> DRAG."""
    recognizer = GestureRecognizer(confirmation_frames=1)
    lm_pinch = make_hand(thumb_state="pinch", four_extended=True)
    # Pinch frames increment
    for _ in range(4):
        recognizer.recognize(lm_pinch, (480, 640))
    assert recognizer.recognize(lm_pinch, (480, 640)) == DRAG


def test_double_click_detection():
    """Two intentional left clicks within interval -> DOUBLE_CLICK."""
    recognizer = GestureRecognizer(double_click_interval=10)
    lm_click = make_hand(four_together=True, thumb_state="inward", four_extended=True)
    lm_open = make_hand(four_together=False, thumb_state="outward", four_extended=True)

    # First click
    assert recognizer.recognize(lm_click, (480, 640)) == LEFT_CLICK
    # Release pose
    recognizer.recognize(lm_open, (480, 640))
    # Second click inside window
    assert recognizer.recognize(lm_click, (480, 640)) == DOUBLE_CLICK


def test_custom_pose_matching_in_recognizer():
    """Item 6: PoseManager.match_pose() is checked and triggers custom pose with confirmation."""
    pm = PoseManager(filepath=None)
    # Create and register a custom scroll pose
    lm_custom = make_hand(four_together=False, thumb_state="folded", four_extended=True)
    pm.capture_pose(SCROLL, lm_custom)

    recognizer = GestureRecognizer(pose_manager=pm, confirmation_frames=2)
    # Frame 1: matching begins
    recognizer.recognize(lm_custom, (480, 640))
    # Frame 2: confirmation reached -> SCROLL
    assert recognizer.recognize(lm_custom, (480, 640)) == SCROLL


def test_click_stability_requirement():
    """Verify that a click gesture must remain stable for N frames before triggering."""
    recognizer = GestureRecognizer(click_stability_frames=3)
    lm_click = make_hand(four_together=True, thumb_state="inward", four_extended=True)

    # Frame 1: candidate frame 1 -> NONE
    assert recognizer.recognize(lm_click, (480, 640)) == NONE
    # Frame 2: candidate frame 2 -> NONE
    assert recognizer.recognize(lm_click, (480, 640)) == NONE
    # Frame 3: candidate frame 3 >= 3 -> LEFT_CLICK
    assert recognizer.recognize(lm_click, (480, 640)) == LEFT_CLICK
    # Frame 4: debounced -> NONE
    assert recognizer.recognize(lm_click, (480, 640)) == NONE


def test_click_speed_suppression():
    """Verify that clicks are suppressed when hand speed exceeds threshold."""
    recognizer = GestureRecognizer(click_stability_frames=1, max_click_speed=20.0)
    lm_click = make_hand(four_together=True, thumb_state="inward", four_extended=True)

    # Moving at speed 30 > 20 -> Click suppressed
    assert recognizer.recognize(lm_click, (480, 640), hand_speed=30.0) == NONE
    # Moving slowly at speed 5 <= 20 -> Click allowed
    assert recognizer.recognize(lm_click, (480, 640), hand_speed=5.0) == LEFT_CLICK


def test_handedness_thumb_detection():
    """Verify thumb extension and fold detection works with handedness metadata."""
    recognizer = GestureRecognizer(click_stability_frames=1)

    # Simulated Left hand where landmarks are mirrored across the center
    lm_left_outward = make_hand(four_together=True, thumb_state="outward", four_extended=True)
    # Mirror x around center 100
    for i in range(len(lm_left_outward)):
        x, y, z = lm_left_outward[i]
        lm_left_outward[i] = (200 - x, y, z)

    # Right click pose on Left hand
    g = recognizer.recognize(lm_left_outward, (480, 640), handedness="Left")
    assert g in (RIGHT_CLICK, CURSOR_MOVE)


def test_pinch_double_click_and_middle_click():
    """Verify distinct pinch targets: thumb touches middle -> DOUBLE_CLICK, ring -> MIDDLE_CLICK."""
    recognizer = GestureRecognizer(confirmation_frames=1)

    # 1. Thumb touches Middle tip, index/ring/little extended -> DOUBLE_CLICK
    lm_double = make_hand(thumb_state="outward", four_extended=True)
    # Move thumb tip close to middle tip (landmark 12)
    lm_double[4] = (lm_double[12][0] + 1, lm_double[12][1] + 1, 0.0)
    assert recognizer.recognize(lm_double, (480, 640)) == DOUBLE_CLICK
    # Holding pinch fires only once (debounce)
    assert recognizer.recognize(lm_double, (480, 640)) == NONE

    # 2. Thumb touches Ring tip, index/middle/little extended -> MIDDLE_CLICK
    lm_middle = make_hand(thumb_state="outward", four_extended=True)
    # Move thumb tip close to ring tip (landmark 16)
    lm_middle[4] = (lm_middle[16][0] + 1, lm_middle[16][1] + 1, 0.0)
    assert recognizer.recognize(lm_middle, (480, 640)) == MIDDLE_CLICK
    assert recognizer.recognize(lm_middle, (480, 640)) == NONE


def test_zoom_gesture():
    """Verify Zoom pose: index + middle + ring extended, little folded."""
    recognizer = GestureRecognizer()
    lm_zoom = make_hand(thumb_state="outward", four_extended=True)
    # Fold little finger (tip at 20, pip at 18)
    lm_zoom[20] = (180, 160, 0.0)
    lm_zoom[18] = (180, 130, 0.0)
    assert recognizer.recognize(lm_zoom, (480, 640)) == ZOOM


def test_scroll_v_and_scroll_h():
    """Verify Scroll direction: larger vertical displacement -> SCROLL_V, larger horizontal -> SCROLL_H."""
    recognizer = GestureRecognizer()
    lm_scroll = make_hand(two_fingers_only=True)

    # First frame initializes reference
    g1 = recognizer.recognize(lm_scroll, (480, 640))
    assert g1 in (SCROLL_V, SCROLL)

    # Move vertically by 15px
    lm_scroll_v = [list(pt) for pt in lm_scroll]
    lm_scroll_v[8][1] += 15
    g_v = recognizer.recognize(lm_scroll_v, (480, 640))
    assert g_v == SCROLL_V

    # Move horizontally by 25px
    lm_scroll_h = [list(pt) for pt in lm_scroll_v]
    lm_scroll_h[8][0] += 25
    g_h = recognizer.recognize(lm_scroll_h, (480, 640))
    assert g_h == SCROLL_H


def test_shaka_swipe_back_and_forward():
    """Verify Shaka pose with swipe velocity triggers Back and Forward navigation."""
    recognizer = GestureRecognizer(swipe_speed_threshold=15.0, swipe_cooldown_frames=5)
    # Shaka pose: thumb (4) and little (20) extended, index/middle/ring folded
    lm_shaka = [(100, 200, 0.0) for _ in range(21)]
    lm_shaka[0] = (200, 200, 0.0)  # Wrist
    lm_shaka[9] = (200, 100, 0.0)  # Middle MCP

    # Thumb extended outward
    lm_shaka[4] = (50, 120, 0.0)
    lm_shaka[3] = (100, 150, 0.0)
    lm_shaka[2] = (150, 170, 0.0)

    # Little finger extended
    lm_shaka[20] = (320, 80, 0.0)
    lm_shaka[18] = (250, 140, 0.0)

    # Index, middle, ring folded
    lm_shaka[8] = (180, 160, 0.0)
    lm_shaka[6] = (180, 130, 0.0)
    lm_shaka[12] = (200, 160, 0.0)
    lm_shaka[10] = (200, 130, 0.0)
    lm_shaka[16] = (220, 160, 0.0)
    lm_shaka[14] = (220, 130, 0.0)

    # Frame 1: establishes position
    assert recognizer.recognize(lm_shaka, (480, 640)) == NONE

    # Frame 2: Fast swipe left (dx = -30 < 0) -> BACK
    lm_swipe_left = [list(pt) for pt in lm_shaka]
    lm_swipe_left[0][0] -= 30
    assert recognizer.recognize(lm_swipe_left, (480, 640)) == BACK

    # Cooldown in effect
    assert recognizer.recognize(lm_swipe_left, (480, 640)) == NONE

    # Reset cooldown
    recognizer._swipe_cooldown = 0
    recognizer._prev_shaka_x = 200

    # Fast swipe right (dx = +30 > 0) -> FORWARD
    lm_swipe_right = [list(pt) for pt in lm_shaka]
    lm_swipe_right[0][0] = 230
    assert recognizer.recognize(lm_swipe_right, (480, 640)) == FORWARD


