"""tests/test_main_ui.py
Unit tests for UI functions in main.py:
- draw_hud
- draw_cheat_sheet
- draw_calibration_menu
- draw_reach_calibration
"""

import numpy as np
import pytest

from main import (
    draw_hud,
    draw_cheat_sheet,
    draw_calibration_menu,
    draw_reach_calibration,
)
from gesture_recognition import (
    CURSOR_MOVE,
    LEFT_CLICK,
    RIGHT_CLICK,
    DOUBLE_CLICK,
    MIDDLE_CLICK,
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


def test_draw_cheat_sheet():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    rendered = draw_cheat_sheet(frame)
    assert isinstance(rendered, np.ndarray)
    assert rendered.shape == (720, 1280, 3)
    # Ensure pixels were actually drawn (not all zero)
    assert np.any(rendered > 0)


def test_draw_cheat_sheet_fallback_res():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rendered = draw_cheat_sheet(frame)
    assert isinstance(rendered, np.ndarray)
    assert rendered.shape == (480, 640, 3)
    assert np.any(rendered > 0)


@pytest.mark.parametrize("gesture", [
    CURSOR_MOVE,
    LEFT_CLICK,
    RIGHT_CLICK,
    DOUBLE_CLICK,
    MIDDLE_CLICK,
    SCROLL_V,
    SCROLL_H,
    ZOOM,
    BACK,
    FORWARD,
    DRAG,
    PAUSE,
    NONE,
])
def test_draw_hud_gestures(gesture):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    rendered = draw_hud(
        frame=frame,
        fps=30.0,
        gesture=gesture,
        pinch_ratio=0.04,
        pinch_threshold=0.055,
        is_paused=False,
        control_mode="absolute",
        precision_mode=True,
        is_low_light=True,
        is_hand_lost=False,
        active_monitor_str="MON 1/2",
        calib_status="Status message",
    )
    assert isinstance(rendered, np.ndarray)
    assert rendered.shape == (720, 1280, 3)
    assert np.any(rendered > 0)


def test_draw_calibration_menu(tmp_path):
    pose_file = tmp_path / "test_poses.json"
    pm = PoseManager(filepath=str(pose_file))
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    rendered = draw_calibration_menu(frame, pm, "Ready")
    assert isinstance(rendered, np.ndarray)
    assert rendered.shape == (720, 1280, 3)
    assert np.any(rendered > 0)


def test_draw_reach_calibration():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    rendered1 = draw_reach_calibration(frame.copy(), step=1, pt1=None)
    assert isinstance(rendered1, np.ndarray)
    assert np.any(rendered1 > 0)

    rendered2 = draw_reach_calibration(frame.copy(), step=2, pt1=(200, 200))
    assert isinstance(rendered2, np.ndarray)
    assert np.any(rendered2 > 0)

