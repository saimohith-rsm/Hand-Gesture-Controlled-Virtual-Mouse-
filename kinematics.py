import math
import numpy as np
from typing import Tuple, List, Optional
import config

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

def finger_is_extended(landmarks: List[Tuple[int, int, float]],
                       tip_idx: int,
                       pip_idx: int,
                       wrist_idx: int = 0,
                       margin: Optional[float] = None) -> bool:
    """A finger is extended if dist(wrist, tip) > dist(wrist, pip) * margin.
    This is rotation- and tilt-invariant.
    """
    if margin is None:
        margin = getattr(config, "FINGER_EXTENSION_MARGIN", 1.1)

    # If wrist is present and distinct from tip and pip
    if len(landmarks) > wrist_idx and wrist_idx not in (tip_idx, pip_idx):
        wrist = landmarks[wrist_idx][:2]
        tip = landmarks[tip_idx][:2]
        pip = landmarks[pip_idx][:2]
        d_tip = euclidean_distance(wrist, tip)
        d_pip = euclidean_distance(wrist, pip)
        return d_tip > (d_pip * margin)

    # Fallback for simple 2-landmark test structures where wrist is missing
    return landmarks[tip_idx][1] < landmarks[pip_idx][1]

def pinch_distance(landmarks: List[Tuple[int, int, float]], idx1: int, idx2: int) -> float:
    """Normalized pinch distance between two landmarks (ratio of frame diagonal)."""
    p1 = landmarks[idx1][:2]
    p2 = landmarks[idx2][:2]
    return euclidean_distance(p1, p2)


def is_thumb_folded_inward(landmarks: List[Tuple[int, int, float]],
                           palm_size: float,
                           handedness: Optional[str] = None) -> bool:
    """Check if thumb is folded inward across the palm for either Left or Right hand."""
    if len(landmarks) < 5:
        return False
    thumb_tip = landmarks[4][:2]
    index_mcp = landmarks[5][:2]
    d_index_mcp = euclidean_distance(thumb_tip, index_mcp) / palm_size

    # Handedness-aware transverse axis check if 21 landmarks exist
    if len(landmarks) > 17:
        middle_mcp = landmarks[9][:2]
        ring_mcp = landmarks[13][:2]
        little_mcp = landmarks[17][:2]
        d_middle = euclidean_distance(thumb_tip, middle_mcp) / palm_size
        d_ring = euclidean_distance(thumb_tip, ring_mcp) / palm_size
        d_little = euclidean_distance(thumb_tip, little_mcp) / palm_size

        # Vector from pinky MCP to index MCP (transverse palm vector)
        v_trans = (index_mcp[0] - little_mcp[0], index_mcp[1] - little_mcp[1])
        len_trans = math.hypot(v_trans[0], v_trans[1])
        if len_trans > 1e-4:
            v_thumb = (thumb_tip[0] - index_mcp[0], thumb_tip[1] - index_mcp[1])
            proj = (v_thumb[0] * v_trans[0] + v_thumb[1] * v_trans[1]) / (len_trans * palm_size)
            if proj < 0.15 and (d_index_mcp < 0.70 or d_middle < 0.65 or d_ring < 0.75 or d_little < 0.75):
                return True

    # Fallback to direct distance ratios
    d_middle_mcp = (euclidean_distance(thumb_tip, landmarks[9][:2]) / palm_size) if len(landmarks) > 9 else 1.0
    return (d_index_mcp < 0.62) or (d_middle_mcp < 0.65)


def is_thumb_extended_outward(landmarks: List[Tuple[int, int, float]],
                             palm_size: float,
                             handedness: Optional[str] = None) -> bool:
    """Check if thumb is extended outward away from palm for either Left or Right hand."""
    if len(landmarks) < 5:
        return False
    thumb_tip = landmarks[4][:2]
    index_mcp = landmarks[5][:2]
    d_index_mcp = euclidean_distance(thumb_tip, index_mcp) / palm_size

    if len(landmarks) > 17:
        little_mcp = landmarks[17][:2]
        d_little = euclidean_distance(thumb_tip, little_mcp) / palm_size
        v_trans = (index_mcp[0] - little_mcp[0], index_mcp[1] - little_mcp[1])
        len_trans = math.hypot(v_trans[0], v_trans[1])
        if len_trans > 1e-4:
            v_thumb = (thumb_tip[0] - index_mcp[0], thumb_tip[1] - index_mcp[1])
            proj = (v_thumb[0] * v_trans[0] + v_thumb[1] * v_trans[1]) / (len_trans * palm_size)
            if proj > 0.15 and d_little > 0.75 and d_index_mcp > 0.45:
                return True

    return d_index_mcp >= 0.65

