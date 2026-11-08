import math
import numpy as np
from typing import Tuple, List

def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Return Euclidean distance between two 2D points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def vector(p1: Tuple[float, float], p2: Tuple[float, float]) -> np.ndarray:
    """Return the vector from p1 to p2 as a NumPy array."""
    return np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)

def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return the angle in degrees between two vectors.
    Handles zero‑length vectors safely by returning 0.
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))

def finger_is_extended(landmarks: List[Tuple[int, int, float]], tip_idx: int, pip_idx: int) -> bool:
    """Simple heuristic: fingertip y must be above (lower value) the PIP joint for an extended finger.
    Works for a right‑hand image where y increases downwards.
    """
    tip_y = landmarks[tip_idx][1]
    pip_y = landmarks[pip_idx][1]
    return tip_y < pip_y  # higher on image = lower y value

def pinch_distance(landmarks: List[Tuple[int, int, float]], idx1: int, idx2: int) -> float:
    """Normalized pinch distance between two landmarks (ratio of frame diagonal)."""
    p1 = landmarks[idx1][:2]
    p2 = landmarks[idx2][:2]
    return euclidean_distance(p1, p2)

