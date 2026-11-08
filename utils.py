"""utils.py
Utility helpers for the virtual mouse project.
"""

import cv2
import time
from typing import Tuple

def map_to_screen(coord: Tuple[int, int],
                  frame_size: Tuple[int, int],
                  screen_size: Tuple[int, int],
                  active_region: Tuple[float, float, float, float]) -> Tuple[int, int]:
    """Map a point from the webcam frame to absolute screen coordinates.

    ``active_region`` is a normalized rectangle (x1, y1, x2, y2) where
    (0,0) is the top‑left of the frame and (1,1) the bottom‑right.
    The function clamps the point to that region, normalises it and then
    scales it to the physical screen size.
    """
    fx1, fy1, fx2, fy2 = active_region
    w_f, h_f = frame_size
    w_s, h_s = screen_size

    # Clamp inside the active region
    x = max(fx1 * w_f, min(coord[0], fx2 * w_f))
    y = max(fy1 * h_f, min(coord[1], fy2 * h_f))

    # Normalise to [0,1] within the region
    norm_x = (x - fx1 * w_f) / ((fx2 - fx1) * w_f)
    norm_y = (y - fy1 * h_f) / ((fy2 - fy1) * h_f)

    # Scale to screen dimensions
    screen_x = int(norm_x * w_s)
    screen_y = int(norm_y * h_s)
    return screen_x, screen_y


class FPSCounter:
    """Simple frames‑per‑second counter.
    Call :meth:`tick` once per loop iteration; it returns the current FPS.
    """
    def __init__(self):
        self._prev = time.time()
        self.fps = 0.0

    def tick(self) -> float:
        now = time.time()
        dt = now - self._prev
        self.fps = 1.0 / dt if dt > 0 else 0.0
        self._prev = now
        return self.fps


def draw_overlay(frame: 'cv2.Mat', lines: list, origin: Tuple[int, int] = (10, 20)) -> 'cv2.Mat':
    """Draw a list of text lines onto ``frame``.
    ``origin`` is the top‑left corner for the first line; each subsequent
    line is offset vertically by ~22 px.
    """
    x, y = origin
    for line in lines:
        cv2.putText(frame, str(line), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y += 22
    return frame

