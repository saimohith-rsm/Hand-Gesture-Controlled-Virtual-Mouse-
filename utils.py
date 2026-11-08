"""utils.py
Utility helpers for the virtual mouse project:
- Screen mapping with failsafe clamping
- Moving-average smoothed FPSCounter
- ThreadedCamera for zero-buffer-lag frame acquisition
- HUD overlay rendering
"""

import cv2
import os
import json
import time
import threading
import collections
import numpy as np
from typing import Tuple, Optional

def map_to_screen(coord: Tuple[float, float],
                  frame_size: Tuple[int, int],
                  screen_size: Tuple[int, int],
                  active_region: Tuple[float, float, float, float],
                  monitor_offset: Tuple[int, int] = (0, 0)) -> Tuple[int, int]:
    """Map a point from the webcam frame to absolute screen coordinates.

    ``active_region`` is a normalized rectangle (x1, y1, x2, y2).
    Clamps output to [1, width-2] and [1, height-2] to avoid triggering
    PyAutoGUI's FailSafe at (0, 0), with optional multi-monitor offset.
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

    # Clamp output to [1, width-2] and [1, height-2]
    screen_x = max(1, min(screen_x, w_s - 2)) + monitor_offset[0]
    screen_y = max(1, min(screen_y, h_s - 2)) + monitor_offset[1]
    return screen_x, screen_y


class PositionHistory:
    """Buffer of recent screen positions for click freeze / anti-drift."""
    def __init__(self, maxlen: int = 5):
        self.maxlen = max(1, maxlen)
        self._history = collections.deque(maxlen=self.maxlen + 1)

    def append(self, pt: Tuple[int, int]):
        self._history.append(pt)

    def get_frozen_point(self) -> Optional[Tuple[int, int]]:
        """Return the point from maxlen frames ago (or the oldest available)."""
        if not self._history:
            return None
        return self._history[0]

    def clear(self):
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history)


class FPSCounter:
    """Smoothed frames‑per‑second counter using a moving average over 10 frames."""
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.timestamps = collections.deque(maxlen=window_size)
        self.fps = 0.0

    def tick(self) -> float:
        now = time.time()
        self.timestamps.append(now)
        if len(self.timestamps) > 1:
            elapsed = self.timestamps[-1] - self.timestamps[0]
            if elapsed > 0:
                self.fps = (len(self.timestamps) - 1) / elapsed
        return self.fps


class ThreadedCamera:
    """Camera capture on a separate background thread that always holds only the freshest frame.
    Requests high resolution (1280x720) and falls back to standard (640x480) if unavailable.
    """
    def __init__(self,
                 src: int = 0,
                 width: int = 1280,
                 height: int = 720,
                 fallback_width: int = 640,
                 fallback_height: int = 480,
                 fps: int = 30,
                 buffer_size: int = 1,
                 fourcc: str = "MJPG",
                 use_dshow: bool = True):
        backend = cv2.CAP_DSHOW if (use_dshow and os.name == "nt") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(src, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        if fourcc and len(fourcc) == 4:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

        self.ret, self.frame = self.cap.read()
        # Fallback to lower resolution if camera fails or doesn't support 1280x720
        if not self.ret or self.frame is None or self.frame.shape[1] < width * 0.7:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, fallback_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, fallback_height)
            self.ret, self.frame = self.cap.read()

        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)
        self.cap.release()


def check_lighting_and_enhance(frame: np.ndarray,
                               threshold: float = 70.0,
                               enable_clahe: bool = True,
                               clip_limit: float = 2.0,
                               grid_size: Tuple[int, int] = (8, 8)) -> Tuple[np.ndarray, bool, float]:
    """Check average frame brightness and apply CLAHE contrast enhancement if in low-light conditions.

    Returns:
        (processed_frame, is_low_light, average_brightness)
    """
    if frame is None or frame.size == 0:
        return frame, False, 128.0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = float(np.mean(gray))
    is_low_light = avg_brightness < threshold

    if is_low_light and enable_clahe:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        l_enhanced = clahe.apply(l_channel)
        enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced_bgr, is_low_light, avg_brightness

    return frame, is_low_light, avg_brightness


def load_calibration_region(filepath: str = "calibration.json",
                            default_region: Tuple[float, float, float, float] = (0.1, 0.1, 0.9, 0.9)) -> Tuple[float, float, float, float]:
    """Load normalized active screen control region from calibration.json if present."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                region = data.get("active_region")
                if region and len(region) == 4:
                    return tuple(region)
        except Exception as e:
            print(f"[Calibration] Error reading {filepath}: {e}")
    return default_region


def save_calibration_region(region: Tuple[float, float, float, float],
                            filepath: str = "calibration.json"):
    """Save normalized active screen control region to calibration.json."""
    try:
        with open(filepath, "w") as f:
            json.dump({"active_region": list(region)}, f, indent=2)
    except Exception as e:
        print(f"[Calibration] Error saving {filepath}: {e}")


def draw_overlay(frame: 'cv2.Mat', lines: list, origin: Tuple[int, int] = (10, 20)) -> 'cv2.Mat':
    """Draw a list of text lines onto ``frame``."""
    x, y = origin
    for line in lines:
        cv2.putText(frame, str(line), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y += 22
    return frame
