"""Configuration constants for the VIGIL drowsiness detection system."""

from __future__ import annotations


# Camera and window settings
CAMERA_INDEX = 0
WINDOW_NAME = "VIGIL - Real-Time Drowsiness Detection"

# Detection thresholds
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 20

# MediaPipe settings
FACE_MESH_MAX_FACES = 1
FACE_MESH_REFINE_LANDMARKS = True
FACE_MESH_MIN_DETECTION_CONFIDENCE = 0.5
FACE_MESH_MIN_TRACKING_CONFIDENCE = 0.5

# Eye landmark indices (MediaPipe FaceMesh)
# EAR uses p1..p6 in this order:
# p1: outer corner, p2/p3: upper eyelid, p4: inner corner, p5/p6: lower eyelid.
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# UI colors in BGR format (OpenCV uses BGR, not RGB)
DARK_NAVY_BGR = (30, 15, 10)  # #0A0F1E
CYAN_BGR = (255, 255, 0)  # #00FFFF
LIME_GREEN_BGR = (20, 255, 57)  # #39FF14
ALERT_RED_BGR = (76, 76, 255)  # #FF4C4C
ELECTRIC_BLUE_BGR = (241, 142, 31)  # #1F8EF1
WHITE_BGR = (255, 255, 255)

# UI sizing
HUD_FONT_SCALE = 0.7
HUD_FONT_THICKNESS = 2
STATUS_FONT_SCALE = 1.4
STATUS_FONT_THICKNESS = 3
LANDMARK_RADIUS = 2
BOX_THICKNESS = 2
ALERT_BANNER_ALPHA = 0.35

# Output recording
SAVE_OUTPUT_VIDEO = False
OUTPUT_DIR = "output"
OUTPUT_FPS = 20.0
VIDEO_CODEC = "XVID"

# Audio settings
ALERT_AUDIO_PATH = "assets/alert.wav"
ALERT_AUDIO_VOLUME = 0.9
