# Hand Gesture Controlled Virtual Mouse 🖱️🖐️

A real-time, high-accuracy computer vision application developed in Python using **OpenCV**, **MediaPipe**, **NumPy**, and **PyAutoGUI** that translates natural hand gestures captured via a webcam into operating system mouse actions.

---

## 🌟 Key Features & Engineering Highlights

- **Offline-First Model Architecture**: Automatically loads `hand_landmarker.task` locally from the project directory. If absent, attempts automatic download; if network access fails, displays clear, copy-pasteable manual instructions with `curl.exe` without crashing.
- **Dual-Handedness Recognition**: Reads MediaPipe handedness (`Left`/`Right`) and computes projection along the transverse palm axis ($\vec{P}_5 - \vec{P}_{17}$) to classify folded inward vs. extended outward thumbs reliably for both left and right hands.
- **Accidental Action Prevention**:
  - **Edge-Margin Auto-Pause**: Auto-pauses mouse actions when the wrist or index fingertip is within `FRAME_EDGE_MARGIN` pixels of the camera border, displaying `HAND LOST` in the HUD.
  - **Click Stability**: Requires click gestures to remain stable for $N$ consecutive frames (`CLICK_STABILITY_FRAMES`) before triggering.
  - **Speed-Threshold Suppression**: Rejects click triggers when hand velocity exceeds `MAX_CLICK_SPEED_THRESHOLD` px/frame, eliminating misfires during rapid hand sweeps.
- **Precision Mode (`P` key)**: Instantly toggles cursor speed to 0.4x with heavier smoothing parameters (`PRECISION_MIN_CUTOFF`, `PRECISION_BETA`) for micro-targeting and UI control.
- **Arm Fatigue Reduction & Relative Mode (`M` key)**:
  - **Touchpad-like Relative Mode**: Moves the cursor by delta displacements ($\Delta x, \Delta y$).
  - **Mid-Air Repositioning**: Folding a fist (`PAUSE`) or lifting the hand out of view lets you reposition your arm freely without moving the cursor, just like lifting a mouse off a mousepad.
- **Comfortable Reach Calibration (`K` key)**:
  - First-run wizard or interactive calibration saves top-left and bottom-right reach bounds to `calibration.json`, enabling small hand movements to span the entire monitor.
- **Low-Light Detection & CLAHE Contrast Enhancement**:
  - Dynamically calculates average frame brightness. If below `LOW_LIGHT_THRESHOLD`, displays a `LOW LIGHT` indicator and applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to the LAB color space for consistent landmark detection.
- **Zero-Latency Win32 Cursor Positioning**: Direct OS-level positioning via `ctypes.windll.user32.SetCursorPos` and `GetCursorPos` on Windows (falling back to PyAutoGUI cross-platform) for sub-millisecond execution.
- **Threaded Camera Pipeline**: Decoupled background capture thread with `cv2.CAP_DSHOW`, `CAP_PROP_BUFFERSIZE=1`, and `MJPG` format, requesting 1280x720 with automatic fallback to 640x480.
- **Live Pose Scanning Studio (`C` key)**: Real-time interactive scanning studio to customize, save, and apply personalized hand poses with persistent JSON storage.

---

## 🖐️ Gesture Control Mapping

| # | Gesture Pose | Finger Anatomy / Trigger Condition | Mouse Action |
| :-: | :--- | :--- | :--- |
| **1** | **Open Palm** | All 5 fingers extended outward | **Move Mouse Cursor** *(smooth tracking, never conflicts with scroll)* |
| **2** | **Left Click** | 4 fingers together + **thumb folded inward** (or custom pose) | **Left Click** *(click-freeze cursor, debounced)* |
| **3** | **Right Click** | 4 fingers together + **thumb extended outward** (or custom pose) | **Right Click** *(click-freeze cursor, debounced)* |
| **4** | **Drag & Drop** | **Pinch Thumb + Index tip** (hold & move) | **Drag & Drop** *(mouse down on hold $\rightarrow$ move $\rightarrow$ mouse up on release)* |
| **5** | **Double Click** | **Pinch Thumb + Middle tip** (index/ring/little extended) | **Double Click** *(fires once per pinch, requires release before re-fire)* |
| **6** | **Middle Click** | **Pinch Thumb + Ring tip** (index/middle/little extended) | **Middle Click** *(fires once per pinch, requires release before re-fire)* |
| **7** | **Scroll (V & H)** | **Index + Middle extended**, Ring + Little folded | **Scroll V / Scroll H** *(vertical or horizontal based on displacement axis)* |
| **8** | **Zoom In / Out** | **Index + Middle + Ring extended**, Little folded | **Zoom In / Out** *(hand up = zoom in, down = zoom out via Ctrl + Scroll)* |
| **9** | **Back / Forward** | **Shaka Pose** (Thumb + Little extended, others folded) | **Back / Forward** *(fast horizontal swipe left = Alt+Left, right = Alt+Right)* |
| **10** | **Closed Fist** | All fingers folded into a fist | **Pause Tracking** *(open palm resumes; reposition arm freely)* |

---

## ⌨️ Keyboard Shortcuts Reference

| Key | Function | Description |
| :---: | :--- | :--- |
| **`M`** | **Toggle Control Mode** | Switch between **ABSOLUTE** (direct mapping) and **RELATIVE** (touchpad-like delta). |
| **`P`** | **Toggle Precision Mode** | Switch Precision Mode ON/OFF (0.4x speed, heavier smoothing for micro-targeting). |
| **`N`** | **Cycle Active Monitor** | Cycle cursor control across connected displays (multi-monitor support with offset). |
| **`H`** | **Toggle Cheat Sheet** | Show / hide on-screen semi-transparent HUD cheat sheet overlay with all gestures. |
| **`K`** | **Calibrate Reach** | Launch 2-step comfortable reach calibration (saves custom bounds to `calibration.json`). |
| **`C`** | **Custom Pose Studio** | Enter / Exit real-time gesture customization studio. |
| **`1` - `9`** | **Scan Custom Poses** | While in Pose Studio: `1`=Left Click, `2`=Right Click, `3`=Double Click, `4`=Middle Click, `5`=Scroll, `6`=Zoom, `7`=Back, `8`=Forward, `9`=Pause. |
| **`S`** | **Save Custom Poses** | Persist scanned custom poses to `custom_poses.json`. |
| **`R`** | **Reset Factory Defaults** | Clear custom poses and restore built-in gesture rules. |
| **`ESC` / `Q`** | **Quit Application** | Safely release camera, keys, and terminate program cleanly. |

---

## 🛠️ Technology Stack

- **Python 3.12**
- **OpenCV (`opencv-python`)**: Multi-threaded camera acquisition, CLAHE enhancement, image processing, and HUD overlay.
- **MediaPipe Hands (`mediapipe`)**: Tasks vision HandLandmarker running in video mode with handedness classification.
- **NumPy (`<2.0.0`)**: Vector math, transverse palm projection, and coordinate transforms.
- **PyAutoGUI & ctypes**: Direct Windows Win32 cursor control and mouse event synthesis with fail-safe support.
- **Screeninfo**: Multi-monitor screen dimension detection.
- **PyTest**: Unit testing suite.

---

## 📂 Project Structure

```text
├── config.py                 # Tunable system parameters (thresholds, smoothing, lighting, edge margin)
├── convex_hull.py            # Contour extraction, convex hull, and convexity defects
├── filters.py                # One-Euro filter with real dt timing, deadzone smoothing, and reset()
├── gesture_recognition.py    # Gesture recognition engine, handedness, stability, drag & double click
├── hand_tracking.py          # MediaPipe Tasks API HandTracker running in RunningMode.VIDEO
├── kinematics.py             # Dual-handed thumb classification and rotation-independent extension
├── main.py                   # Real-time multi-threaded loop, HUD dashboard, modes, and safety checks
├── mouse_control.py          # Win32 ctypes cursor positioning with PyAutoGUI fallback, cooldowns, and drag
├── pose_calibration.py       # Custom real-time pose scanner, JSON storage, and pose matching
├── pytest.ini                # PyTest configuration for path resolution
├── requirements.txt          # Python dependencies (clean, minimal, numpy<2.0 pinned)
├── tests/                    # 55 comprehensive unit tests across 8 modules
│   ├── test_convex_hull.py
│   ├── test_filters.py
│   ├── test_gestures.py
│   ├── test_kinematics.py
│   ├── test_main_ui.py
│   ├── test_mouse_control.py
│   ├── test_pose_calibration.py
│   └── test_utils.py
├── utils.py                  # ThreadedCamera, PositionHistory, FPSCounter, lighting/CLAHE, reach calib
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/saimohith-rsm/Hand-Gesture-Controlled-Virtual-Mouse-.git
cd Hand-Gesture-Controlled-Virtual-Mouse-
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate

pip install -r requirements.txt
```

### 3. MediaPipe Model Setup (Offline Ready)
The model file `hand_landmarker.task` is loaded directly from the project directory. If missing, it will automatically attempt to download, or you can manually download it:
```powershell
curl.exe -o hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
```

### 4. Run the Application
```bash
python main.py
```

---

## 🧪 Running Unit Tests

To run the complete test suite:
```bash
pytest tests
```
*(All 55 unit tests pass without requiring a physical camera).*

---

## 🛑 Safety & Fail-Safe

- **Emergency Exit**: Push your physical mouse into the top-left corner `(0, 0)` of your monitor to trigger PyAutoGUI's emergency fail-safe (cleanly caught with graceful shutdown).
- **Edge Margin Auto-Pause**: Moving your hand outside the safe camera boundary automatically pauses all cursor and click events.
- **Exit App**: Press **`ESC`** or **`Q`** on your keyboard while focused on the application window.
