"""convex_hull.py
Utility for extracting hand contour, convex hull and defects using OpenCV.
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional

def get_hand_mask(frame: np.ndarray, landmarks: List[Tuple[int, int]]) -> np.ndarray:
    """Create a binary mask of the hand based on landmark convex hull."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    if not landmarks or len(landmarks) < 3:
        return mask
    points = np.array(landmarks, dtype=np.int32)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask

def find_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Find the largest contour in the binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 100:
        return None
    return contour

def convex_hull_and_defects(contour: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Calculate convex hull and convexity defects for a given contour."""
    if contour is None or len(contour) < 3:
        return np.array([]), None
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return np.array([]), None
    try:
        defects = cv2.convexityDefects(contour, hull)
    except cv2.error:
        defects = None
    return hull, defects

def draw_hull_and_defects(frame: np.ndarray,
                          contour: np.ndarray,
                          hull: np.ndarray,
                          defects: Optional[np.ndarray]) -> np.ndarray:
    """Overlay hull (green) and defects (red) on the frame for visualization."""
    if hull is not None and len(hull) >= 3:
        hull_indices = hull.flatten()
        hull_points = contour[hull_indices]
        cv2.polylines(frame, [hull_points], isClosed=True, color=(0, 255, 0), thickness=2)

    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            if d > 800:  # Filter out trivial noise
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                cv2.circle(frame, far, 4, (0, 0, 255), -1)
                cv2.line(frame, start, end, (255, 128, 0), 1)

    return frame
