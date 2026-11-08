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

