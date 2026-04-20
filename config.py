"""Configuration constants for the VIGIL drowsiness detection system."""

from __future__ import annotations


# Camera and window settings
CAMERA_INDEX = 0
WINDOW_NAME = "VIGIL - Real-Time Drowsiness Detection"

# Detection thresholds
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 20          # frames of low EAR before alert triggers (~0.67s @ 30fps)

DROWSY_SECONDS_BEFORE_ALERT = 5.0   # 3 seconds of sustained low EAR

# Cancel window: user has this many seconds to cancel after alert fires
CANCEL_WINDOW_SECONDS = 5.0

# MediaPipe Tasks settings
MODEL_PATH = "assets/face_landmarker.task"  # 478-point Tasks API model
FACE_MESH_MAX_FACES = 1
MIN_FACE_DETECTION_CONFIDENCE = 0.5
MIN_FACE_PRESENCE_CONFIDENCE  = 0.5
MIN_TRACKING_CONFIDENCE       = 0.5

# Eye landmark indices (MediaPipe FaceMesh)
LEFT_EYE_INDICES  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# ── Layout ────────────────────────────────────────────────────────────────────
SIDE_PANEL_WIDTH   = 280          # pixels, right-side panel
CANVAS_PADDING     = 18           # grey border around the video feed

# Background / panel colours (BGR)
BG_GREY_BGR        = (60, 60, 60)       # outer canvas background
PANEL_DARK_BGR     = (35, 35, 35)       # side panel background
PANEL_BORDER_BGR   = (90, 90, 90)       # side panel left border

# UI colours (BGR)
DARK_NAVY_BGR      = (30, 15, 10)
CYAN_BGR           = (255, 255, 0)
LIME_GREEN_BGR     = (20, 255, 57)
ALERT_RED_BGR      = (76, 76, 255)
ELECTRIC_BLUE_BGR  = (241, 142, 31)
WHITE_BGR          = (255, 255, 255)
ORANGE_BGR         = (0, 165, 255)      # countdown colour
GREEN_OK_BGR       = (50, 200, 50)      # cancel / normal accent

# UI sizing
HUD_FONT_SCALE      = 0.6
HUD_FONT_THICKNESS  = 2
STATUS_FONT_SCALE   = 1.2
STATUS_FONT_THICKNESS = 3
LANDMARK_RADIUS     = 2
BOX_THICKNESS       = 2
ALERT_BANNER_ALPHA  = 0.40

# Cancel button geometry (relative to side-panel origin)
CANCEL_BTN_MARGIN   = 20
CANCEL_BTN_HEIGHT   = 52

# Output recording
SAVE_OUTPUT_VIDEO = False
OUTPUT_DIR        = "output"
OUTPUT_FPS        = 20.0
VIDEO_CODEC       = "XVID"

# Audio
ALERT_AUDIO_PATH   = "assets/alert.wav"
ALERT_AUDIO_VOLUME = 0.9