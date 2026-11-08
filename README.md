# Hand Gesture Controlled Virtual Mouse 🖱️🖐️

A real-time, high-accuracy computer vision application developed in Python using **OpenCV**, **MediaPipe**, **NumPy**, and **PyAutoGUI** that translates natural hand gestures captured via a laptop webcam into actual operating system mouse actions.

---

## 🌟 Features

- **21-Landmark Hand Tracking**: Uses Google MediaPipe to track 3D hand joints in real-time.
- **Ultra-Accurate Cursor Positioning**: Employs an adaptive One-Euro filter with micro-jitter deadzone suppression and dual-landmark bone anchoring (75% Tip + 25% DIP Joint) for rock-steady cursor control.
- **Convex Hull & Defect Analysis**: Uses OpenCV `findContours`, `convexHull`, and `convexityDefects` to compute hand boundary shapes.
- **Zero-Latency Mouse Control**: Configured PyAutoGUI with `0.0s` delay for buttery-smooth 30+ FPS responsiveness.
- **State Debouncing**: Prevents accidental repeated clicks; requires gesture separation before triggering another action.
- **Custom Gesture Scanning Studio**: Press `C` in real-time to scan, capture, and customize your own unique hand poses.
- **Interactive HUD Overlay**: Displays live FPS, current mode, control bounding area, and an index fingertip aiming reticle.

---

## 🖐️ Gesture Control Mapping

| # | Gesture Pose | Trigger Condition | Mouse Action |
| :-: | :--- | :--- | :--- |
| **1** | **Open Palm** | All 5 fingers extended outward | **Move Mouse Cursor** |
| **2** | **Custom Left Click** | 4 fingers extended together + **thumb folded inward** | **Left Click** *(debounced, 1 click per gesture)* |
| **3** | **Custom Right Click** | 4 fingers extended together + **thumb extended outward** | **Right Click** *(debounced, 1 click per gesture)* |
| **4** | **Four-Finger Motion** | 4 fingers extended moving vertically: <br>• Hand moves **Upward**<br>• Hand moves **Downward** | <br>**Scroll Up**<br>**Scroll Down** |
| **5** | **Closed Fist** | All fingers folded into a fist | **Pause Tracking** *(open palm resumes)* |

---

## 🛠️ Technology Stack

- **Python 3.12**
- **OpenCV (`opencv-python`)**: Webcam acquisition, image processing, contours, and HUD visualization.
- **MediaPipe Hands (`mediapipe`)**: Real-time hand landmark and skeleton detection.
- **NumPy**: Vector math, angle computation, and coordinate transforms.
- **PyAutoGUI**: Windows operating system cursor and mouse event automation.
- **Screeninfo**: Multi-monitor screen dimension detection.
- **PyTest**: Unit testing suite.

---

## 📂 Project Structure

```text
s:\cv project\
├── config.py                 # Tunable system parameters (thresholds, smoothing, active zone)
├── convex_hull.py            # Contour extraction, convex hull, and convexity defects
├── filters.py                # One-Euro filter implementation with deadzone smoothing
├── gesture_recognition.py    # Stateful gesture recognition engine and debouncing
├── hand_tracking.py          # MediaPipe HandTracker supporting both Tasks and Solutions APIs
├── kinematics.py             # Vector math, Euclidean distances, and finger extension checks
├── main.py                   # Main real-time webcam loop and HUD dashboard
├── mouse_control.py          # Safe PyAutoGUI abstraction with click cooldown and failsafe
├── pose_calibration.py       # Custom real-time pose scanner, JSON storage, and pose matching
├── requirements.txt          # Python dependencies
├── tests/                    # Comprehensive unit tests (18 passed)
│   ├── test_convex_hull.py
│   ├── test_filters.py
│   ├── test_gestures.py
│   ├── test_kinematics.py
│   ├── test_mouse_control.py
│   └── test_pose_calibration.py
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/hand-gesture-virtual-mouse.git
cd hand-gesture-virtual-mouse
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Download the MediaPipe Model File
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
*(All 18 unit tests should pass in < 1 second).*

---

## 📸 Real-Time Pose Scanning Studio

To customize your own hand poses:
1. While running the app, press **`C`** on your keyboard to enter the **Scanning Studio**.
2. Form your preferred hand pose in front of the camera:
   - Press **`1`** $\rightarrow$ Scan pose as **Left Click**
   - Press **`2`** $\rightarrow$ Scan pose as **Right Click**
   - Press **`3`** $\rightarrow$ Scan pose as **Scroll**
   - Press **`4`** $\rightarrow$ Scan pose as **Pause**
3. Press **`S`** to Save & Apply your custom poses (stored in `custom_poses.json`).
4. Press **`C`** to exit and use your new gestures!
*(Press `R` anytime to reset back to factory defaults).*

---

## 🛑 Safety & Fail-Safe

- **Emergency Exit**: Push your physical mouse into the top-left corner `(0, 0)` of your monitor to trigger PyAutoGUI's emergency fail-safe.
- **Exit App**: Press **`ESC`** or **`Q`** on your keyboard while focused on the webcam window.

