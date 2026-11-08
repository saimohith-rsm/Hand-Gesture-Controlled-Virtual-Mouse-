"""mouse_control.py
Utility class that abstracts pyautogui mouse actions.
Implements safety features (failsafe, zero delay, click debounce, drag handling)
and a test mode that only visualises actions.
"""

from typing import Tuple, Optional

class MouseController:
    def __init__(self, test_mode: bool = False, click_cooldown_frames: int = 8):
        """Create a controller for the OS mouse.
        Args:
            test_mode: If True, no real mouse actions are performed.
            click_cooldown_frames: Minimum number of frames between successive click
                events to avoid accidental multi-clicking.
        """
        self.test_mode = test_mode
        self.click_cooldown_frames = click_cooldown_frames
        self.last_click_frame = -click_cooldown_frames
        self.frame_counter = 0
        self.is_dragging = False

        if not test_mode:
            import pyautogui
            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = 0.0  # Crucial: removes PyAutoGUI's default 100ms lag per call
            self.pg = pyautogui
        else:
            self.pg = None

    def _ready_for_click(self) -> bool:
        return (self.frame_counter - self.last_click_frame) >= self.click_cooldown_frames

    def move_to(self, x: int, y: int):
        """Move cursor immediately to screen coordinates (x, y)."""
        if self.test_mode:
            return (x, y)
        self.pg.moveTo(x, y)

    def left_click(self):
        if not self._ready_for_click():
            return
        if not self.test_mode:
            self.pg.click()
        self.last_click_frame = self.frame_counter

    def right_click(self):
        if not self._ready_for_click():
            return
        if not self.test_mode:
            self.pg.rightClick()
        self.last_click_frame = self.frame_counter

    def double_click(self):
        if not self._ready_for_click():
            return
        if not self.test_mode:
            self.pg.doubleClick()
        self.last_click_frame = self.frame_counter

    def scroll(self, dy: int):
        if not self.test_mode:
            self.pg.scroll(dy)

    def drag_start(self):
        if not self.is_dragging:
            self.is_dragging = True
            if not self.test_mode:
                self.pg.mouseDown()

    def drag_end(self):
        if self.is_dragging:
            self.is_dragging = False
            if not self.test_mode:
                self.pg.mouseUp()

    def update_frame_counter(self):
        self.frame_counter += 1
