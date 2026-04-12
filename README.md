# VIGIL: Vision-Based Real-Time Drowsiness Detection System

VIGIL is a fully offline Python computer vision project that detects drowsiness from live webcam input using MediaPipe FaceMesh and the Eye Aspect Ratio (EAR) formula.

## Features

- Real-time webcam processing with OpenCV.
- 468-point face mesh extraction via MediaPipe.
- EAR-based drowsiness logic:
  - `EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)`
  - Alert triggers when `EAR < 0.25` for `20` consecutive frames.
- Audible alarm when drowsiness is detected.
- Dark-theme UI rendered directly in OpenCV:
  - Background tint: dark navy `#0A0F1E`
  - EAR HUD: cyan `#00FFFF`
  - Eye landmarks: lime green `#39FF14`
  - Eye boxes: electric blue `#1F8EF1`
  - Alert banner: semi-transparent red `#FF4C4C`
- Timestamped terminal logs for each alert event.
- Optional annotated output video recording via `config.py`.

## Project Structure

```text
vigil/
├── main.py
├── detector.py
├── alert.py
├── config.py
├── utils.py
├── assets/
│   └── alert.wav
└── README.md
```

## Requirements

- Python 3.9
- OpenCV
- MediaPipe
- NumPy
- SciPy
- pygame or playsound

Install dependencies:

```bash
pip install opencv-python mediapipe numpy scipy pygame playsound
```

If you use `pygame`, it will be preferred automatically for looped alarms.

## Run

From the `vigil/` directory:

```bash
python main.py
```

Press `q` to quit.

## Configuration

All runtime settings are in `config.py`, including:

- `EAR_THRESHOLD` (default `0.25`)
- `CONSECUTIVE_FRAMES` (default `20`)
- `SAVE_OUTPUT_VIDEO` (`True/False`)
- audio path and volume
- MediaPipe confidence settings

## Notes

- The app is fully offline and does not use cloud APIs.
- Uses lightweight geometry + landmark tracking (no heavy deep learning model loading).
- For best results, ensure your face is well-lit and visible to the webcam.
