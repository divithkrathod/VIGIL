"""Audio + contact alert manager for VIGIL drowsiness detection."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional

try:
    import pygame
    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False
    pygame = None  # type: ignore

try:
    from playsound import playsound
    _HAS_PLAYSOUND = True
except ImportError:
    _HAS_PLAYSOUND = False
    playsound = None  # type: ignore

import config


class AlertManager:
    """
    Manages the full drowsiness alert lifecycle:

    1.  EAR stays low for DROWSY_SECONDS_BEFORE_ALERT  → audio alarm fires,
        cancel window opens (CANCEL_WINDOW_SECONDS).
    2a. User clicks Cancel within the window            → alarm stops, no contact alert.
    2b. Cancel window expires without interaction       → contact alert sent.

    All timing is driven by wall-clock (time.monotonic) so it is independent
    of the frame rate.
    """

    def __init__(
        self,
        sound_path: str,
        volume: float = 1.0,
        on_contact_alert: Optional[Callable[[], None]] = None,
    ) -> None:
        self.sound_path = Path(sound_path)
        self.volume = max(0.0, min(volume, 1.0))
        self.on_contact_alert = on_contact_alert or self._default_contact_alert

        # ── public state (read by main / utils) ──────────────────────────────
        self.is_alerting       = False   # audio alarm active
        self.in_cancel_window  = False   # 5-s cancel countdown running
        self.cancel_deadline   = 0.0     # monotonic time when cancel window closes
        self.contact_sent      = False   # True once the contact alert fired

        # ── internal ─────────────────────────────────────────────────────────
        self._low_ear_since: Optional[float] = None   # when sustained closure began
        self._backend = "none"
        self._cancel_timer: Optional[threading.Timer] = None

        # audio backend init
        if _HAS_PYGAME:
            pygame.mixer.init()
            if self.sound_path.exists():
                pygame.mixer.music.load(str(self.sound_path))
                pygame.mixer.music.set_volume(self.volume)
            self._backend = "pygame"
        elif _HAS_PLAYSOUND:
            self._backend = "playsound"

    # ── called every frame from main.py ──────────────────────────────────────

    def update(self, ear_is_low: bool) -> None:
        """
        Feed the current frame's eye state.  Handles the 3-second sustained
        closure timer; call this once per frame instead of the old trigger/reset
        pattern.
        """
        now = time.monotonic()

        if ear_is_low:
            if self._low_ear_since is None:
                self._low_ear_since = now          # start timing

            elapsed = now - self._low_ear_since
            if elapsed >= config.DROWSY_SECONDS_BEFORE_ALERT and not self.is_alerting:
                self._fire_alarm(now)

        else:
            # Eyes opened — reset sustained-closure timer
            self._low_ear_since = None

            # Only stop audio/alarm if cancel window is NOT active.
            # If cancel window IS active, keep alarm ringing until cancelled or expired.
            if self.is_alerting and not self.in_cancel_window:
                self._stop_audio()
                self.is_alerting = False

    def seconds_until_alert(self) -> float:
        """Remaining seconds before a sustained-closure alert fires. Returns 0 if idle or already alerting."""
        if self.is_alerting or self._low_ear_since is None:
            return 0.0
        elapsed = time.monotonic() - self._low_ear_since
        remaining = config.DROWSY_SECONDS_BEFORE_ALERT - elapsed
        # Only return a meaningful value while actively counting down
        if remaining >= config.DROWSY_SECONDS_BEFORE_ALERT:
            return 0.0
        return max(0.0, remaining)

    def cancel_seconds_remaining(self) -> float:
        """Seconds left in the cancel window (0 if not in window)."""
        if not self.in_cancel_window:
            return 0.0
        return max(0.0, self.cancel_deadline - time.monotonic())

    # ── user action ──────────────────────────────────────────────────────────

    def cancel(self) -> None:
        """User pressed Cancel — abort the contact alert and stop the alarm."""
        if not self.in_cancel_window:
            return
        if self._cancel_timer is not None:
            self._cancel_timer.cancel()
            self._cancel_timer = None
        self._stop_audio()
        self.is_alerting      = False
        self.in_cancel_window = False
        self._low_ear_since   = None
        print("[VIGIL] Alert cancelled by user.")

    # ── internal ─────────────────────────────────────────────────────────────

    def _fire_alarm(self, now: float) -> None:
        """Start audio alarm and open the cancel window."""
        self.is_alerting      = True
        self.in_cancel_window = True
        self.cancel_deadline  = now + config.CANCEL_WINDOW_SECONDS
        self.contact_sent     = False
        self._play_audio()

        # Schedule contact alert after cancel window expires
        self._cancel_timer = threading.Timer(
            config.CANCEL_WINDOW_SECONDS, self._send_contact_alert
        )
        self._cancel_timer.daemon = True
        self._cancel_timer.start()
        print(f"[VIGIL] ALARM: drowsiness confirmed. "
              f"{config.CANCEL_WINDOW_SECONDS:.0f}s cancel window opened.")

    def _send_contact_alert(self) -> None:
        """Called when cancel window expires without user action."""
        self.in_cancel_window = False
        self.contact_sent     = True
        print("[VIGIL] *** CONTACT ALERT SENT *** (cancel window expired)")
        try:
            self.on_contact_alert()
        except Exception as exc:
            print(f"[VIGIL] Contact alert callback error: {exc}")

    @staticmethod
    def _default_contact_alert() -> None:
        """
        Stub — replace with SMS/WhatsApp/email logic when ready.
        Currently just prints a prominent message.
        """
        print("=" * 60)
        print("  VIGIL: DROWSINESS ALERT SENT TO REGISTERED CONTACTS")
        print("  (Integrate Twilio / SMTP here to send real messages)")
        print("=" * 60)

    def _play_audio(self) -> None:
        if self._backend == "pygame" and self.sound_path.exists():
            pygame.mixer.music.play(loops=-1)
        elif self._backend == "playsound":
            t = threading.Thread(
                target=lambda: playsound(str(self.sound_path), block=True),
                daemon=True,
            )
            t.start()

    def _stop_audio(self) -> None:
        if self._backend == "pygame":
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass

    # ── cleanup ──────────────────────────────────────────────────────────────

    def reset_after_cancel(self) -> None:
        """Full state reset after the user has acknowledged the alert."""
        self.cancel()

    def close(self) -> None:
        if self._cancel_timer is not None:
            self._cancel_timer.cancel()
        self._stop_audio()
        if self._backend == "pygame":
            try:
                pygame.mixer.quit()
            except Exception:
                pass