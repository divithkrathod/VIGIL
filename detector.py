"""Core drowsiness detection logic using MediaPipe FaceMesh and EAR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
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


def _landmark_to_pixel(
    landmark: object, frame_width: int, frame_height: int
) -> Tuple[int, int]:
    """Convert a normalized MediaPipe landmark to pixel coordinates."""
    x_px = int(landmark.x * frame_width)
    y_px = int(landmark.y * frame_height)
    return x_px, y_px


def extract_eye_points(
    face_landmarks: object, frame_width: int, frame_height: int, indices: list[int]
) -> np.ndarray:
    """Extract 2D pixel coordinates for one eye from landmark indices."""
    points = [
        _landmark_to_pixel(face_landmarks.landmark[idx], frame_width, frame_height)
        for idx in indices
    ]
    return np.array(points, dtype=np.float32)


def compute_ear(eye_points: np.ndarray) -> float:
    """Compute Eye Aspect Ratio (EAR) from six ordered eye points."""
    p1, p2, p3, p4, p5, p6 = eye_points
    vertical_1 = euclidean(p2, p6)
    vertical_2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


class DrowsinessDetector:
    """Wraps MediaPipe FaceMesh and computes per-frame EAR metrics."""

    def __init__(self) -> None:
        """Initialize MediaPipe FaceMesh for real-time webcam processing."""
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=config.FACE_MESH_MAX_FACES,
            refine_landmarks=config.FACE_MESH_REFINE_LANDMARKS,
            min_detection_confidence=config.FACE_MESH_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.FACE_MESH_MIN_TRACKING_CONFIDENCE,
        )

    def process_frame(self, frame_bgr: np.ndarray) -> FrameDetection:
        """Run face mesh and return eye landmarks plus EAR values."""
        frame_height, frame_width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(frame_rgb)

        if not result.multi_face_landmarks:
            return FrameDetection(False, None, None, None, None, None)

        face_landmarks = result.multi_face_landmarks[0]

        left_points = extract_eye_points(
            face_landmarks, frame_width, frame_height, config.LEFT_EYE_INDICES
        )
        right_points = extract_eye_points(
            face_landmarks, frame_width, frame_height, config.RIGHT_EYE_INDICES
        )

        left_ear = compute_ear(left_points)
        right_ear = compute_ear(right_points)
        avg_ear = (left_ear + right_ear) / 2.0

        return FrameDetection(True, left_points, right_points, left_ear, right_ear, avg_ear)

    def close(self) -> None:
        """Release MediaPipe FaceMesh resources."""
        self._face_mesh.close()
