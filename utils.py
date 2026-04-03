"""UI drawing and overlay helpers for the VIGIL OpenCV window."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

import config


def apply_dark_theme_overlay(frame: np.ndarray) -> np.ndarray:
    """Blend frame with dark navy tint for consistent dark-theme rendering."""
    tint = np.full_like(frame, config.DARK_NAVY_BGR, dtype=np.uint8)
    return cv2.addWeighted(frame, 0.75, tint, 0.25, 0.0)


def draw_eye_landmarks(frame: np.ndarray, eye_points: np.ndarray) -> None:
    """Draw eye landmark points in lime green."""
    for x, y in eye_points.astype(int):
        cv2.circle(frame, (x, y), config.LANDMARK_RADIUS, config.LIME_GREEN_BGR, -1)


def draw_eye_bounding_box(frame: np.ndarray, eye_points: np.ndarray) -> None:
    """Draw electric-blue bounding box around eye landmarks."""
    x, y, w, h = cv2.boundingRect(eye_points.astype(int))
    cv2.rectangle(frame, (x, y), (x + w, y + h), config.ELECTRIC_BLUE_BGR, config.BOX_THICKNESS)


def draw_alert_banner(frame: np.ndarray, text: str = "DROWSINESS ALERT!") -> None:
    """Draw red semi-transparent alert banner near the top of frame."""
    overlay = frame.copy()
    height, width = frame.shape[:2]
    top = int(height * 0.12)
    bottom = int(height * 0.24)

    cv2.rectangle(overlay, (0, top), (width, bottom), config.ALERT_RED_BGR, -1)
    cv2.addWeighted(
        overlay, config.ALERT_BANNER_ALPHA, frame, 1 - config.ALERT_BANNER_ALPHA, 0, frame
    )

    put_centered_text(
        frame=frame,
        text=text,
        y=int((top + bottom) / 2 + 10),
        color=config.WHITE_BGR,
        font_scale=0.9,
        thickness=2,
    )


def put_centered_text(
    frame: np.ndarray,
    text: str,
    y: int,
    color: Tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    """Draw a single centered text line at a given y-coordinate."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = max((frame.shape[1] - text_width) // 2, 0)
    baseline_y = max(y, text_height + 2)
    cv2.putText(frame, text, (x, baseline_y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, ear: float, frame_counter: int, is_alerting: bool) -> None:
    """Render EAR and state HUD text on top of frame."""
    status = "ALERT" if is_alerting else "NORMAL"

    cv2.putText(
        frame,
        f"EAR: {ear:.3f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        config.HUD_FONT_SCALE,
        config.CYAN_BGR,
        config.HUD_FONT_THICKNESS,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Low-EAR Frames: {frame_counter}/{config.CONSECUTIVE_FRAMES}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        config.CYAN_BGR,
        2,
        cv2.LINE_AA,
    )
    put_centered_text(
        frame=frame,
        text=status,
        y=55,
        color=config.ALERT_RED_BGR if is_alerting else config.CYAN_BGR,
        font_scale=config.STATUS_FONT_SCALE,
        thickness=config.STATUS_FONT_THICKNESS,
    )
