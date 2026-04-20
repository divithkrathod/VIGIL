"""VIGIL — main entry point with canvas layout, side panel, and cancel support."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

import config
from alert import AlertManager
from detector import DrowsinessDetector
from utils import (
    apply_dark_theme_overlay,
    build_canvas,
    draw_alert_banner,
    draw_eye_bounding_box,
    draw_eye_landmarks,
    draw_feed_hud,
    draw_side_panel,
    paste_feed,
)


# ── Globals shared with the mouse callback ────────────────────────────────────

_cancel_btn_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)   # (x1, y1, x2, y2)
_alert_manager_ref: Optional[AlertManager] = None


def _mouse_callback(event, x, y, flags, param) -> None:
    """Handle left-click on the cancel button."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if _alert_manager_ref is None:
        return
    x1, y1, x2, y2 = _cancel_btn_rect
    if x1 <= x <= x2 and y1 <= y <= y2:
        _alert_manager_ref.cancel()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _build_video_writer(
    canvas_w: int, canvas_h: int
) -> Optional[cv2.VideoWriter]:
    if not config.SAVE_OUTPUT_VIDEO:
        return None
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    fname  = f"vigil_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
    fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
    writer = cv2.VideoWriter(
        str(output_dir / fname), fourcc, config.OUTPUT_FPS, (canvas_w, canvas_h)
    )
    if writer.isOpened():
        print(f"[{_timestamp()}] Recording to: {output_dir / fname}")
        return writer
    print(f"[{_timestamp()}] Warning: could not open VideoWriter.")
    return None


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    global _cancel_btn_rect, _alert_manager_ref

    detector      = DrowsinessDetector()
    alert_manager = AlertManager(
        sound_path=config.ALERT_AUDIO_PATH,
        volume=config.ALERT_AUDIO_VOLUME,
    )
    _alert_manager_ref = alert_manager

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            "Unable to open webcam. Check camera permissions / device index."
        )

    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(config.WINDOW_NAME, _mouse_callback)

    low_ear_frame_count = 0
    writer: Optional[cv2.VideoWriter] = None
    canvas_size: Optional[Tuple[int, int]] = None   # (w, h) of full canvas

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[{_timestamp()}] Warning: failed to read frame.")
                break

            feed_h, feed_w = frame.shape[:2]

            # ── Build canvas on first frame (we now know feed size) ──────────
            if canvas_size is None:
                canvas, feed_origin = build_canvas(feed_h, feed_w)
                canvas_size = (canvas.shape[1], canvas.shape[0])
                # Resize window to fit canvas exactly
                cv2.resizeWindow(config.WINDOW_NAME, canvas_size[0], canvas_size[1])
                writer = _build_video_writer(canvas_size[0], canvas_size[1])
            else:
                canvas, feed_origin = build_canvas(feed_h, feed_w)

            # ── Process video feed ───────────────────────────────────────────
            frame = apply_dark_theme_overlay(frame)
            detection = detector.process_frame(frame)

            avg_ear = 0.0
            ear_is_low = False

            if detection.face_found and detection.avg_ear is not None:
                avg_ear = detection.avg_ear
                ear_is_low = avg_ear < config.EAR_THRESHOLD

                draw_eye_landmarks(frame, detection.left_eye_points)
                draw_eye_landmarks(frame, detection.right_eye_points)
                draw_eye_bounding_box(frame, detection.left_eye_points)
                draw_eye_bounding_box(frame, detection.right_eye_points)

                if ear_is_low:
                    low_ear_frame_count += 1
                else:
                    low_ear_frame_count = 0
            else:
                low_ear_frame_count = 0

            # ── Update alert state machine ───────────────────────────────────
            alert_manager.update(ear_is_low)

            # ── Video-feed overlays ──────────────────────────────────────────
            if alert_manager.is_alerting:
                draw_alert_banner(frame)

            draw_feed_hud(
                frame=frame,
                ear=avg_ear,
                frame_counter=low_ear_frame_count,
                is_alerting=alert_manager.is_alerting,
            )

            # ── Compose canvas ───────────────────────────────────────────────
            paste_feed(canvas, frame, feed_origin)

            _cancel_btn_rect = draw_side_panel(
                canvas=canvas,
                feed_w=feed_w,
                feed_h=feed_h,
                ear=avg_ear,
                low_ear_frames=low_ear_frame_count,
                is_alerting=alert_manager.is_alerting,
                in_cancel_window=alert_manager.in_cancel_window,
                cancel_secs_remaining=alert_manager.cancel_seconds_remaining(),
                contact_sent=alert_manager.contact_sent,
                secs_until_alert=alert_manager.seconds_until_alert(),
            )

            # ── Display ──────────────────────────────────────────────────────
            cv2.imshow(config.WINDOW_NAME, canvas)

            if writer is not None:
                writer.write(canvas)

            # ── Key handling ─────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord("c") or key == ord("C"):
                # Keyboard shortcut for Cancel
                alert_manager.cancel()

            if key == ord("q") or key == 27:
                print(f"[{_timestamp()}] Exit requested by user.")
                break

            if cv2.getWindowProperty(
                config.WINDOW_NAME, cv2.WND_PROP_VISIBLE
            ) < 1:
                print(f"[{_timestamp()}] Window closed.")
                break

    finally:
        if writer is not None:
            writer.release()
        cap.release()
        detector.close()
        alert_manager.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()