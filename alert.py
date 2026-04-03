"""Audio alert manager for drowsiness events."""

from __future__ import annotations

import threading
from pathlib import Path


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


class AlertManager:
    """Triggers and resets audible alerts when drowsiness is detected."""

    def __init__(self, sound_path: str, volume: float = 1.0) -> None:
        """Initialize alert backend using pygame or playsound."""
        self.sound_path = Path(sound_path)
        self.volume = max(0.0, min(volume, 1.0))
        self.is_alerting = False
        self._backend = "none"

        if _HAS_PYGAME:
            pygame.mixer.init()
            pygame.mixer.music.load(str(self.sound_path))
            pygame.mixer.music.set_volume(self.volume)
            self._backend = "pygame"
        elif _HAS_PLAYSOUND:
            self._backend = "playsound"

    def _play_once_playsound(self) -> None:
        """Play sound once in a background thread using playsound."""
        if _HAS_PLAYSOUND and self.sound_path.exists():
            playsound(str(self.sound_path), block=True)

    def trigger(self) -> bool:
        """Start alert sound if not already active; return True on new trigger."""
        if self.is_alerting:
            return False

        self.is_alerting = True

        if self._backend == "pygame":
            pygame.mixer.music.play(loops=-1)
        elif self._backend == "playsound":
            thread = threading.Thread(target=self._play_once_playsound, daemon=True)
            thread.start()

        return True

    def reset(self) -> None:
        """Stop active alert sound and mark system as normal."""
        if not self.is_alerting:
            return

        self.is_alerting = False
        if self._backend == "pygame":
            pygame.mixer.music.stop()

    def close(self) -> None:
        """Release audio resources and stop any active alarm."""
        self.reset()
        if self._backend == "pygame":
            pygame.mixer.quit()
