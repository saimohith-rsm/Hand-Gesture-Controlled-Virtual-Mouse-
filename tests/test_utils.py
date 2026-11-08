"""test_utils.py
Tests for utils functions.
"""
from utils import (
    map_to_screen,
    PositionHistory,
    FPSCounter,
    check_lighting_and_enhance,
    load_calibration_region,
    save_calibration_region,
)
import time
import numpy as np
import tempfile
import os

def test_map_to_screen_clamping():
    # Top-left of frame -> should clamp to at least (1, 1), avoiding (0, 0) failsafe
    screen_x, screen_y = map_to_screen((0, 0), (640, 480), (1920, 1080), (0.1, 0.1, 0.9, 0.9))
    assert screen_x == 1
    assert screen_y == 1

    # Bottom-right of frame -> should clamp to at most (w-2, h-2)
    screen_x, screen_y = map_to_screen((640, 480), (640, 480), (1920, 1080), (0.1, 0.1, 0.9, 0.9))
    assert screen_x == 1920 - 2
    assert screen_y == 1080 - 2

def test_position_history_click_freeze():
    history = PositionHistory(maxlen=5)
    assert history.get_frozen_point() is None
    assert len(history) == 0

    # Feed 10 positions
    for i in range(10):
        history.append((100 + i, 200 + i))

    # With maxlen=5 (buffer size 6), the oldest point is from 5 frames ago:
    # Most recent added was (109, 209). 5 frames earlier was (104, 204).
    assert history.get_frozen_point() == (104, 204)
    assert len(history) == 6

    # Test clear
    history.clear()
    assert len(history) == 0
    assert history.get_frozen_point() is None

def test_fps_counter_smoothing():
    counter = FPSCounter(window_size=5)
    for _ in range(5):
        counter.tick()
        time.sleep(0.005)
    fps = counter.tick()
    assert fps > 0


def test_lighting_and_clahe():
    # Bright frame -> not low light
    bright_frame = np.full((100, 100, 3), 150, dtype=np.uint8)
    _, is_low, avg = check_lighting_and_enhance(bright_frame, threshold=70.0)
    assert not is_low
    assert avg == 150.0

    # Dark frame -> low light, enhanced with CLAHE
    dark_frame = np.full((100, 100, 3), 30, dtype=np.uint8)
    enhanced, is_low, avg = check_lighting_and_enhance(dark_frame, threshold=70.0, enable_clahe=True)
    assert is_low
    assert avg == 30.0
    assert enhanced.shape == dark_frame.shape


def test_calibration_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_calib = os.path.join(tmpdir, "calib.json")
        default_reg = (0.1, 0.1, 0.9, 0.9)
        # Missing file returns default
        assert load_calibration_region(tmp_calib, default_reg) == default_reg

        # Save and reload
        custom_reg = (0.2, 0.25, 0.8, 0.75)
        save_calibration_region(custom_reg, tmp_calib)
        assert load_calibration_region(tmp_calib, default_reg) == custom_reg

