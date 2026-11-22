"""Plays the two doorbell ringtones (normal / mask) without blocking."""
import logging

log = logging.getLogger(__name__)

NORMAL, MASK = "normal", "mask"


class Ringer:
    """``backends`` maps kind -> object with ``play()``; only one plays per call."""

    def __init__(self, backends):
        self._b = backends

    @classmethod
    def create(cls, normal_path, mask_path) -> "Ringer":
        import pygame
        pygame.mixer.music.load(str(normal_path))      # original doorbell, unchanged

        class _Normal:
            def play(self):
                pygame.mixer.music.play()
        backends = {NORMAL: _Normal()}
        try:
            snd = pygame.mixer.Sound(str(mask_path))
            ch = pygame.mixer.Channel(5)

            class _Mask:
                def play(self):
                    ch.play(snd)
            backends[MASK] = _Mask()
        except Exception as exc:
            log.warning("Mask doorbell unavailable (%s); using normal doorbell for masks", exc)
            backends[MASK] = backends[NORMAL]
        return cls(backends)

    def ring(self, kind: str) -> bool:
        try:
            self._b[kind].play()
            return True
        except Exception as exc:
            log.warning("Could not play %s doorbell: %s", kind, exc)
            return False
