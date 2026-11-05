# Real-Time Driver Monitoring & Voice Alert System

A real-time driver-safety system in Python. It reads live webcam video, tracks the driver's face with MediaPipe Face Mesh, and raises an audible alarm and a spoken voice warning when it detects drowsiness, yawning or distraction.

## Features

- **Drowsiness detection** using the Eye Aspect Ratio (EAR) computed from facial landmarks.
- **Yawn detection** using the Mouth Aspect Ratio (MAR), with a yawn counter and a cooldown to avoid double counting.
- **Distraction detection** by estimating head pose (pitch and yaw) with OpenCV `solvePnP`.
- **Voice and sound alerts** through an alarm sound and text-to-speech (`pyttsx3`), run on background threads so the video loop does not stall.
- **Live dashboard** drawn on the video feed: EAR, MAR, pitch, yaw, yawn count and current status.

## How it works

```
Webcam frame -> MediaPipe Face Mesh (facial landmarks)
             -> EAR (eyes)  |  MAR (mouth)  |  Head pose (pitch, yaw)
             -> Time-based rules -> Status
             -> Alarm sound + spoken warning + on-screen dashboard
```

| Signal | How it is measured | Rule (default values) |
|---|---|---|
| Eyes closed | EAR averaged over both eyes | EAR < 0.20 for 3 s, or for 1 s once 2 or more yawns are counted, gives **DROWSINESS ALERT** |
| Yawning | MAR from inner-lip landmarks | MAR > 0.60 for more than 1.5 s gives **YAWNING**; counted at most once every 5 s |
| Looking away | Head pose from `solvePnP` | \|yaw\| > 40 or \|pitch\| > 25 for more than 1.5 s gives **LOOKING AWAY** |
| No face | No landmarks found | Alarm stops, status is **NO FACE** |

Spoken messages: "Warning. Stop the car." (drowsiness) and "Please look forward." (distraction).

## Project structure

```
.
├── maindriver.py     # main application (webcam loop, detection, alerts)
├── alarm.wav         # alarm sound played on alerts
├── driver.ipynb      # notebook with environment setup / imports
└── README.md
```

## Requirements

- Windows (the alarm uses the built-in `winsound` module)
- Python 3.8 or newer
- A working webcam

Python packages:

```
opencv-python
mediapipe
numpy
pyttsx3
```

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
venv\Scripts\activate

pip install opencv-python mediapipe numpy pyttsx3
```

## Usage

Make sure `alarm.wav` is in the same folder as the script, then run:

```bash
python maindriver.py
```

Press **Esc** to quit.

## Configuration

Thresholds are constants near the top of `maindriver.py` and can be tuned for your camera position and lighting:

```python
EAR_THRESHOLD = 0.20              # below this, eyes count as closed
MOUTH_THRESHOLD = 0.60            # above this, mouth counts as open (yawn)
LOOK_AWAY_YAW_THRESHOLD = 40      # degrees
LOOK_AWAY_PITCH_THRESHOLD = 25    # degrees
```

## Limitations

- Rule-based and threshold-driven: it does not use a trained model, so the thresholds may need tuning for different faces, glasses and lighting.
- Not evaluated on a labelled dataset; no accuracy figures are claimed.
- Works with one camera at index 0 and is Windows only (`winsound`).
- This is a learning and research prototype, not a certified safety product. Do not rely on it while driving.

## Future work

- Evaluate on a labelled drowsiness dataset and report precision and recall.
- Per-user calibration of the EAR and MAR thresholds.
- Cross-platform alarm playback.
- Combine with a learned classifier for more robust detection.

## Tech stack

Python, OpenCV, MediaPipe, NumPy, pyttsx3, multithreading

## Author

anmolkaur aidan, B.Tech CSE (AI & DS), MIT World Peace University, Pune
