"""Configuration parameters for Hand Gesture Virtual Mouse.
All parameters can be tuned without modifying the core code.
"""

# Webcam settings
CAMERA_INDEX = 0  # Default webcam index
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FALLBACK_WIDTH = 640
CAMERA_FALLBACK_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_BUFFER_SIZE = 1
CAMERA_FOURCC = "MJPG"
CAMERA_USE_DSHOW = True  # Use cv2.CAP_DSHOW on Windows for fast startup & low latency

# Model settings
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# MediaPipe detection thresholds
DETECTION_CONFIDENCE = 0.70
TRACKING_CONFIDENCE = 0.70

# Safety thresholds
FRAME_EDGE_MARGIN = 20           # Margin in pixels from frame edges where tracking auto-pauses
CLICK_STABILITY_FRAMES = 2       # Consecutive frames required to confirm a click gesture
MAX_CLICK_SPEED_THRESHOLD = 32.0 # Maximum allowable hand velocity (px/frame) to trigger a click

# Finger extension margin: a finger is extended if dist(wrist, tip) > dist(wrist, pip) * margin
FINGER_EXTENSION_MARGIN = 1.1

# Gesture thresholds (normalized distance ratio)
PINCH_ACTIVATION_THRESHOLD = 0.055  # Distance ratio for pinch trigger
PINCH_RELEASE_THRESHOLD = 0.085     # Distance ratio for pinch release (hysteresis)

# Ultra-Accurate Cursor Smoothing (One Euro filter parameters)
SMOOTHING_FREQ = 30.0        # Default sampling frequency (Hz)
SMOOTHING_MIN_CUTOFF = 0.40  # Lower value = rock-steady when pointing/hovering
SMOOTHING_BETA = 0.25        # High responsiveness on fast hand movements
DEADZONE_PIXELS = 1.6        # Complete suppression of micro-finger tremors

# Precision Mode settings (Toggled with 'P')
PRECISION_SPEED_MULTIPLIER = 0.4
PRECISION_MIN_CUTOFF = 0.15  # Heavy smoothing when precision mode is active
PRECISION_BETA = 0.05

# Arm Fatigue & Control Modes (Toggled with 'M')
DEFAULT_CONTROL_MODE = "absolute"  # "absolute" or "relative"
RELATIVE_SENSITIVITY = 1.8         # Motion multiplier for touchpad-like relative control
CALIBRATION_FILE = "calibration.json"

# Lighting & CLAHE settings
ENABLE_CLAHE = True
LOW_LIGHT_THRESHOLD = 70.0   # Average frame brightness (0-255) below which LOW LIGHT activates
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)

# Click freeze history: move to position from N frames earlier upon click to prevent cursor drift
CLICK_FREEZE_FRAMES = 5

# Click debounce settings (in frames at ~30 FPS)
CLICK_COOLDOWN_FRAMES = 8
DOUBLE_CLICK_MAX_INTERVAL = 14  # Frames between first and second pinch (~0.45s)
GESTURE_CONFIRMATION_FRAMES = 2 # Frames required to confirm pinch

# Scroll settings
SCROLL_DEADZONE = 4            # Minimum vertical movement in pixels before scrolling
SCROLL_UNITS_PER_PIXEL = 60    # Windows wheel notch units per pixel moved (default ~60)
SCROLL_H_DEADZONE = 4          # Minimum horizontal movement in pixels before scrolling
SCROLL_H_UNITS_PER_PIXEL = 60  # Horizontal wheel notch units per pixel

# Zoom settings (Ctrl + Scroll)
ZOOM_DEADZONE = 4              # Minimum vertical movement in pixels before zooming
ZOOM_UNITS_PER_PIXEL = 1       # Zoom increments per pixel of vertical hand displacement

# Swipe navigation settings (Shaka pose Back / Forward)
SWIPE_SPEED_THRESHOLD = 18.0   # Minimum horizontal hand velocity (px/frame) to trigger a swipe
SWIPE_COOLDOWN_FRAMES = 15     # Cooldown frames preventing repeated accidental swipe navigation

# Screen active region (normalized rectangle: x1, y1, x2, y2)
ACTIVE_REGION = (0.10, 0.10, 0.90, 0.90)

# Visualization toggles
SHOW_FPS = True
SHOW_CONVEX_HULL = True
SHOW_ACTIVE_BOX = True
