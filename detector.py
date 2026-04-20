"""Core drowsiness detection logic using MediaPipe Tasks Face Landmarker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from scipy.spatial.distance import euclidean

import config


@dataclass
class FrameDetection:
    """Stores per-frame eye landmark and EAR information."""
    face_found: bool
    left_eye_points: Optional[np.ndarray]
    right_eye_points: Optional[np.ndarray]
    left_ear: Optional[float]
    right_ear: Optional[float]
    avg_ear: Optional[float]


class DrowsinessDetector:
    """Wraps MediaPipe Tasks FaceLandmarker and computes per-frame EAR metrics."""

    def __init__(self) -> None:
        """Initialize Face Landmarker using the modern Tasks API."""
        base_options = python.BaseOptions(model_asset_path=config.MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=config.FACE_MESH_MAX_FACES,
            min_face_detection_confidence=config.MIN_FACE_DETECTION_CONFIDENCE,
            min_face_presence_confidence=config.MIN_FACE_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            running_mode=vision.RunningMode.IMAGE,  # manual loop control
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def _landmark_to_pixel(
        self, landmark, width: int, height: int
    ) -> Tuple[int, int]:
        return int(landmark.x * width), int(landmark.y * height)

    def process_frame(self, frame_bgr: np.ndarray) -> FrameDetection:
        """Run landmarker and return eye landmarks plus EAR values."""
        frame_height, frame_width = frame_bgr.shape[:2]

        # Convert BGR -> RGB -> MediaPipe Image
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return FrameDetection(False, None, None, None, None, None)

        face_landmarks = result.face_landmarks[0]

        left_points = np.array(
            [
                self._landmark_to_pixel(
                    face_landmarks[idx], frame_width, frame_height
                )
                for idx in config.LEFT_EYE_INDICES
            ],
            dtype=np.float32,
        )
        right_points = np.array(
            [
                self._landmark_to_pixel(
                    face_landmarks[idx], frame_width, frame_height
                )
                for idx in config.RIGHT_EYE_INDICES
            ],
            dtype=np.float32,
        )

        left_ear  = self.compute_ear(left_points)
        right_ear = self.compute_ear(right_points)
        avg_ear   = (left_ear + right_ear) / 2.0

        return FrameDetection(
            True, left_points, right_points, left_ear, right_ear, avg_ear
        )

    def compute_ear(self, eye_points: np.ndarray) -> float:
        """Compute Eye Aspect Ratio (EAR)."""
        p1, p2, p3, p4, p5, p6 = eye_points
        vertical_1 = euclidean(p2, p6)
        vertical_2 = euclidean(p3, p5)
        horizontal = euclidean(p1, p4)
        return (
            (vertical_1 + vertical_2) / (2.0 * horizontal)
            if horizontal > 0
            else 0.0
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()