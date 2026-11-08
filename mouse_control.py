"""mouse_control.py
Utility class that abstracts mouse actions.
Implements safety features (failsafe, zero delay, click debounce, drag handling, direct win32 cursor positioning)
and a test mode that only visualises actions.
"""

import sys
import ctypes
from typing import Tuple, Optional

if sys.platform == "win32":
    class _POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

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
        self.is_ctrl_down = False

        if not test_mode:
            import pyautogui
            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = 0.0  # Crucial: removes PyAutoGUI's default 100ms lag per call
            self.pg = pyautogui
        else:
            self.pg = None

    def _ready_for_click(self) -> bool:
        return (self.frame_counter - self.last_click_frame) >= self.click_cooldown_frames

    def get_position(self) -> Tuple[int, int]:
        """Return current OS cursor coordinates (x, y)."""
        if self.test_mode:
            return (960, 540)
        if sys.platform == "win32":
            pt = _POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
            return (int(pt.x), int(pt.y))
        else:
            pos = self.pg.position()
            return (int(pos.x), int(pos.y))

    def move_to(self, x: int, y: int):
        """Move cursor immediately to screen coordinates (x, y).
        Uses ctypes.windll.user32.SetCursorPos on Windows for sub-millisecond execution,
        falling back to pyautogui on other platforms.
        """
        if self.test_mode:
            return (x, y)
        if sys.platform == "win32":
            ctypes.windll.user32.SetCursorPos(int(x), int(y))
        else:
            self.pg.moveTo(x, y)

    def move_rel(self, dx: int, dy: int, screen_size: Tuple[int, int] = (1920, 1080)) -> Tuple[int, int]:
        """Move cursor relative to current position by (dx, dy), clamped to screen boundaries."""
        cur_x, cur_y = self.get_position()
        target_x = max(1, min(cur_x + dx, screen_size[0] - 2))
        target_y = max(1, min(cur_y + dy, screen_size[1] - 2))
        self.move_to(target_x, target_y)
        return (target_x, target_y)

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

    def middle_click(self):
        if not self._ready_for_click():
            return
        if not self.test_mode:
            self.pg.middleClick()
        self.last_click_frame = self.frame_counter

    def scroll(self, dy: int):
        if not self.test_mode:
            self.pg.scroll(dy)

    def hscroll(self, dx: int):
        """Horizontal scroll (dx > 0 right, dx < 0 left)."""
        if not self.test_mode:
            self.pg.hscroll(dx)

    def zoom(self, dy: int):
        """Zoom in (dy > 0) or out (dy < 0) via Ctrl + scroll."""
        if not self.test_mode:
            if not self.is_ctrl_down:
                self.pg.keyDown("ctrl")
                self.is_ctrl_down = True
            self.pg.scroll(dy)

    def release_ctrl(self):
        """Ensure Ctrl key is released when zoom gesture ends or hand is lost."""
        if self.is_ctrl_down:
            self.is_ctrl_down = False
            if not self.test_mode:
                self.pg.keyUp("ctrl")

    def zoom_end(self):
        """Alias for releasing Ctrl at the end of zoom actions."""
        self.release_ctrl()

    def back(self):
        """Trigger Back navigation (Alt + Left)."""
        if not self.test_mode:
            self.pg.hotkey("alt", "left")

    def forward(self):
        """Trigger Forward navigation (Alt + Right)."""
        if not self.test_mode:
            self.pg.hotkey("alt", "right")

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
