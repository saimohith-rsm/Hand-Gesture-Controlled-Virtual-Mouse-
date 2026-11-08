"""test_mouse_control.py
Tests for the MouseController utility in safe test mode.
"""
from mouse_control import MouseController

def test_mouse_controller_init():
    mc = MouseController(test_mode=True, click_cooldown_frames=5)
    assert mc.test_mode is True
    assert mc.click_cooldown_frames == 5

def test_mouse_controller_move():
    mc = MouseController(test_mode=True)
    coords = mc.move_to(100, 200)
    assert coords == (100, 200)

def test_mouse_controller_click_cooldown():
    mc = MouseController(test_mode=True, click_cooldown_frames=5)
    # Initially ready
    assert mc._ready_for_click() is True
    mc.left_click()
    assert mc.last_click_frame == 0
    # Not ready right after click on same frame
    assert mc._ready_for_click() is False
    # Advance frames
    for _ in range(5):
        mc.update_frame_counter()
    # Ready again after cooldown frames passed
    assert mc._ready_for_click() is True

