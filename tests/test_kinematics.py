"""test_kinematics.py
Tests for kinematics utilities.
"""
import math
from kinematics import euclidean_distance, angle_between, finger_is_extended

def test_euclidean_distance():
    assert euclidean_distance((0, 0), (3, 4)) == 5

def test_angle_between_zero_vectors():
    # Zero vector should give angle 0
    assert angle_between([0, 0], [1, 0]) == 0.0

def test_finger_is_extended_true():
    # fingertip y lower (higher on screen) than pip y -> extended
    landmarks = [(0, 100, 0), (0, 200, 0)]  # tip index 0, pip index 1
    assert finger_is_extended(landmarks, 0, 1) is True

def test_finger_is_extended_false():
    # fingertip y higher (lower on screen) than pip y -> folded
    landmarks = [(0, 200, 0), (0, 100, 0)]  # tip index 0, pip index 1
    assert finger_is_extended(landmarks, 0, 1) is False

def test_finger_is_extended_rotated_horizontal():
    # Hand pointing horizontally right (y coordinates identical)
    # index 0: wrist at (100, 200)
    # index 8: tip at (300, 200) -> dist = 200
    # index 6: pip at (200, 200) -> dist = 100
    landmarks = [(100, 200, 0)] * 21
    landmarks[0] = (100, 200, 0)
    landmarks[6] = (200, 200, 0)
    landmarks[8] = (300, 200, 0)
    assert finger_is_extended(landmarks, 8, 6, wrist_idx=0) is True

    # Folded back towards palm
    landmarks[8] = (150, 200, 0)  # dist = 50 < 100 * 1.1
    assert finger_is_extended(landmarks, 8, 6, wrist_idx=0) is False

def test_finger_is_extended_rotated_upside_down():
    # Hand pointing downwards (y increases downwards)
    # index 0: wrist at (200, 100)
    # index 6: pip at (200, 200) -> dist = 100
    # index 8: tip at (200, 300) -> dist = 200
    landmarks = [(200, 100, 0)] * 21
    landmarks[0] = (200, 100, 0)
    landmarks[6] = (200, 200, 0)
    landmarks[8] = (200, 300, 0)
    # In old code: tip.y (300) < pip.y (200) was False. With wrist dist, it is True!
    assert finger_is_extended(landmarks, 8, 6, wrist_idx=0) is True
