"""test_convex_hull.py
Tests for convex_hull utilities.
"""
import numpy as np
import cv2
from convex_hull import get_hand_mask, find_contour, convex_hull_and_defects

def test_hand_mask_nonzero():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # simple square hand landmarks
    landmarks = [(30, 30), (70, 30), (70, 70), (30, 70)]
    mask = get_hand_mask(img, landmarks)
    assert mask.sum() > 0

def test_find_contour_and_hull():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    landmarks = [(30, 30), (70, 30), (70, 70), (30, 70)]
    mask = get_hand_mask(img, landmarks)
    contour = find_contour(mask)
    assert contour is not None
    hull, defects = convex_hull_and_defects(contour)
    assert hull.size > 0
    # defects may be None for a perfect square; ensure function returns None gracefully
    assert defects is None or isinstance(defects, np.ndarray)

