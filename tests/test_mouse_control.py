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

def test_mouse_controller_relative_move():
    mc = MouseController(test_mode=True)
    pos = mc.get_position()
    assert pos == (960, 540)
    rel_pos = mc.move_rel(20, -30, screen_size=(1920, 1080))
    assert rel_pos == (980, 510)

def test_mouse_controller_new_actions():
    mc = MouseController(test_mode=True)
    # Test methods execute safely in test_mode without error
    mc.middle_click()
    mc.hscroll(100)
    mc.zoom(5)
    assert mc.is_ctrl_down is False  # In test_mode, no keys actually pressed
    mc.release_ctrl()
    mc.zoom_end()
    mc.back()
    mc.forward()



