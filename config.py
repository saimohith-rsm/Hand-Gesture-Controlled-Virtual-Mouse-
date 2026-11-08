"""Configuration parameters for Hand Gesture Virtual Mouse.
All parameters can be tuned without modifying the core code.
"""

# Webcam settings
CAMERA_INDEX = 0  # Default webcam index
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# MediaPipe detection thresholds
DETECTION_CONFIDENCE = 0.70
TRACKING_CONFIDENCE = 0.70

# Gesture thresholds (normalized distance ratio)
PINCH_ACTIVATION_THRESHOLD = 0.055  # Distance ratio for pinch trigger
PINCH_RELEASE_THRESHOLD = 0.085     # Distance ratio for pinch release (hysteresis)

# Ultra-Accurate Cursor Smoothing (One Euro filter parameters)
SMOOTHING_FREQ = 30.0
SMOOTHING_MIN_CUTOFF = 0.40  # Lower value = rock-steady when pointing/hovering
SMOOTHING_BETA = 0.25        # High responsiveness on fast hand movements
DEADZONE_PIXELS = 1.6        # Complete suppression of micro-finger tremors

# Click debounce settings (in frames at ~30 FPS)
CLICK_COOLDOWN_FRAMES = 8
DOUBLE_CLICK_MAX_INTERVAL = 14  # Frames between first and second pinch (~0.45s)
GESTURE_CONFIRMATION_FRAMES = 2 # Frames required to confirm pinch

# Scroll settings
SCROLL_DEADZONE = 4            # Minimum vertical movement in pixels before scrolling
SCROLL_SPEED_MULTIPLIER = 1.6  # Scaling for scroll responsiveness

# Screen active region (normalized rectangle: x1, y1, x2, y2)
# Generous 80% screen coverage for high precision reachability
ACTIVE_REGION = (0.10, 0.10, 0.90, 0.90)

# Visualization toggles
SHOW_FPS = True
SHOW_CONVEX_HULL = True
SHOW_ACTIVE_BOX = True
