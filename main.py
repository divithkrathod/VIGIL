from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

import config
from alert import AlertManager
from detector import DrowsinessDetector
from utils import (
    apply_dark_theme_overlay,
    draw_alert_banner,
    draw_eye_bounding_box,
    draw_eye_landmarks,
    draw_hud,
)


def _timestamp() -> str:
    """Return human-readable timestamp for terminal logs."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _build_video_writer(frame_width: int, frame_height: int) -> Optional[cv2.VideoWriter]:
    """Create output video writer if saving is enabled in config."""
    if not config.SAVE_OUTPUT_VIDEO:
        return None

    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"vigil_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
    output_path = output_dir / file_name
    fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        config.OUTPUT_FPS,
        (frame_width, frame_height),
    )

    if writer.isOpened():
        print(f"[{_timestamp()}] Recording output video to: {output_path}")
        return writer

    print(f"[{_timestamp()}] Warning: could not initialize video writer.")
    return None


def main() -> None:
    """Run webcam loop, detect low EAR streaks, and trigger alerts."""
    detector = DrowsinessDetector()
    alert_manager = AlertManager(
        sound_path=config.ALERT_AUDIO_PATH,
        volume=config.ALERT_AUDIO_VOLUME,
    )

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam. Check camera permissions/device index.")

    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)

    low_ear_frame_count = 0
    writer: Optional[cv2.VideoWriter] = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[{_timestamp()}] Warning: failed to read webcam frame.")
                break

            if writer is None:
                writer = _build_video_writer(frame.shape[1], frame.shape[0])

            frame = apply_dark_theme_overlay(frame)
            detection = detector.process_frame(frame)

            avg_ear = 0.0
            if detection.face_found and detection.avg_ear is not None:
                avg_ear = detection.avg_ear

                draw_eye_landmarks(frame, detection.left_eye_points)
                draw_eye_landmarks(frame, detection.right_eye_points)
                draw_eye_bounding_box(frame, detection.left_eye_points)
                draw_eye_bounding_box(frame, detection.right_eye_points)

                if avg_ear < config.EAR_THRESHOLD:
                    low_ear_frame_count += 1
                    if low_ear_frame_count >= config.CONSECUTIVE_FRAMES:
                        is_new_alert = alert_manager.trigger()
                        if is_new_alert:
                            print(
                                f"[{_timestamp()}] ALERT: Drowsiness detected "
                                f"(EAR={avg_ear:.3f}, frames={low_ear_frame_count})."
                            )
                else:
                    low_ear_frame_count = 0
                    alert_manager.reset()
            else:
                low_ear_frame_count = 0
                alert_manager.reset()

            if alert_manager.is_alerting:
                draw_alert_banner(frame)

            draw_hud(
                frame=frame,
                ear=avg_ear,
                frame_counter=low_ear_frame_count,
                is_alerting=alert_manager.is_alerting,
            )

            cv2.imshow(config.WINDOW_NAME, frame)

            if writer is not None:
                writer.write(frame)

            key = cv2.waitKey(1) & 0xFF

# Exit on 'q' or ESC
            if key == ord("q") or key == 27:
                print(f"[{_timestamp()}] Exit requested by user.")
                break

# Exit if window is closed manually
            if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print(f"[{_timestamp()}] Exit requested: window closed.")
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
