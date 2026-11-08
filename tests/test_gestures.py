"""test_gestures.py
Tests for the 5 exact gestures in GestureRecognizer:
1. OPEN PALM (all 5 fingers extended) -> Move cursor.
2. CUSTOM LEFT CLICK POSE (4 fingers together, thumb folded inward) -> Left click.
3. CUSTOM RIGHT CLICK POSE (4 fingers together, thumb extended outward) -> Right click.
4. FOUR-FINGER VERTICAL MOVEMENT -> Scroll up / Scroll down.
5. CLOSED FIST -> Pause tracking. Open palm -> Resume tracking.
Also tests click debouncing.
"""

from gesture_recognition import (
    GestureRecognizer,
    CURSOR_MOVE,
    LEFT_CLICK,
    RIGHT_CLICK,
    SCROLL,
    PAUSE,
    NONE,
)

def make_hand(four_together=False, thumb_state="outward", four_extended=True):
    """Generate 21-landmark dummy hand with specific configuration.
    Palm size = 100 (Wrist at (100, 200), Middle MCP at (100, 100)).
    """
    lm = [(100, 200, 0.0) for _ in range(21)]
    lm[0] = (100, 200, 0.0)  # Wrist
    lm[9] = (100, 100, 0.0)  # Middle MCP

    # Fingers 8, 12, 16, 20 tips and 6, 10, 14, 18 PIPs
    if four_extended:
        tip_y = 50
        pip_y = 120
    else:
        tip_y = 160
        pip_y = 120

    if four_together:
        # Four fingers tightly together: small gaps
        tips_x = {8: 90, 12: 100, 16: 110, 20: 120}
        pips_x = {6: 90, 10: 100, 14: 110, 18: 120}
    else:
        # Fingers spread apart (Open Palm)
        tips_x = {8: 60, 12: 100, 16: 140, 20: 180}
        pips_x = {6: 60, 10: 100, 14: 140, 18: 180}

    for tip, x in tips_x.items():
        lm[tip] = (x, tip_y, 0.0)
    for pip, x in pips_x.items():
        lm[pip] = (x, pip_y, 0.0)

    # Index MCP for thumb reference
    lm[5] = (tips_x[8], 140, 0.0)

    # Thumb configuration
    if thumb_state == "inward":
        # Folded across palm close to Index/Middle MCP
        lm[4] = (95, 125, 0.0)
        lm[3] = (95, 150, 0.0)
    elif thumb_state == "outward":
        # Extended outward away from index MCP
        lm[4] = (10, 80, 0.0)
        lm[3] = (25, 140, 0.0)
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
    """Gesture 1: Open Palm (all 5 fingers extended) -> Move cursor."""
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
    # Frame 1: fires click
    g1 = recognizer.recognize(lm, (480, 640))
    assert g1 == LEFT_CLICK
    # Frame 2 (held): debounced to NONE
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


def test_gesture_4_four_finger_vertical_movement_scroll():
    """Gesture 4: Four fingers moving vertically -> Scroll."""
    recognizer = GestureRecognizer(scroll_velocity_threshold=5.0)
    lm1 = make_hand(four_together=False, thumb_state="outward", four_extended=True)
    # Initial frame
    recognizer.recognize(lm1, (480, 640))

    # Move four fingers upward by 20 pixels
    lm2 = [(x, y - 20, z) for x, y, z in lm1]
    gesture = recognizer.recognize(lm2, (480, 640))
    assert gesture == SCROLL


def test_gesture_5_closed_fist_pause_and_open_palm_resume():
    """Gesture 5: Closed fist -> Pause tracking. Open palm -> Resume tracking."""
    recognizer = GestureRecognizer()
    # 1. Closed fist (all fingers folded)
    lm_fist = make_hand(four_together=True, thumb_state="folded", four_extended=False)
    g_pause = recognizer.recognize(lm_fist, (480, 640))
    assert g_pause == PAUSE

    # 2. While still in fist, remains paused
    g_still_pause = recognizer.recognize(lm_fist, (480, 640))
    assert g_still_pause == PAUSE

    # 3. Open palm resumes tracking
    lm_open = make_hand(four_together=False, thumb_state="outward", four_extended=True)
    g_resume = recognizer.recognize(lm_open, (480, 640))
    assert g_resume == CURSOR_MOVE
