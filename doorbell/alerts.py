"""Alert state machines (multi-person, mask) and non-blocking alert sounds."""
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Decision:
    fired: bool = False       # rising edge: the condition just started
    play_sound: bool = False  # rising edge and cooldown elapsed


class _EdgeAlert:
    def __init__(self, cooldown: float):
        self.cooldown = cooldown
        self.active = False
        self._last_sound: Optional[float] = None

    def _rise(self, now: float) -> Decision:
        self.active = True
        ready = self._last_sound is None or (now - self._last_sound) >= self.cooldown
        if ready:
            self._last_sound = now
        return Decision(fired=True, play_sound=ready)


class MultiPersonAlert(_EdgeAlert):
    """Fires once when the person count reaches the threshold; resets as soon
    as the count drops below it. Sound is additionally rate-limited."""

    def __init__(self, threshold: int = 2, cooldown: float = 10.0):
        super().__init__(cooldown)
        self.threshold = threshold

    def update(self, count: int, now: float) -> Decision:
        if count >= self.threshold:
            return Decision() if self.active else self._rise(now)
        self.active = False
        return Decision()


class MaskAlert(_EdgeAlert):
    """Active while a reliably masked face is present; clears only after
    ``clear_after`` seconds without one (avoids flicker)."""

    def __init__(self, cooldown: float = 8.0, clear_after: float = 1.5):
        super().__init__(cooldown)
        self.clear_after = clear_after
        self._last_masked: Optional[float] = None

    def update(self, masked_count: int, now: float) -> Decision:
        if masked_count > 0:
            self._last_masked = now
            return Decision() if self.active else self._rise(now)
        if self.active and self._last_masked is not None \
                and (now - self._last_masked) >= self.clear_after:
            self.active = False
        return Decision()


# ------------------------------------------------------------------ sound
def _tone(freq: float, dur: float, rate: int, vol: float) -> np.ndarray:
    t = np.arange(int(rate * dur)) / rate
    wave = np.sin(2 * np.pi * freq * t)
    fade = min(int(rate * 0.01), len(wave) // 2)
    env = np.ones_like(wave)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return wave * env * vol


def synth_multi_person(rate: int = 44100) -> np.ndarray:
    """Soft ascending three-note chime (distinct from the doorbell)."""
    gap = np.zeros(int(rate * 0.05))
    parts = []
    for f in (523.25, 659.25, 783.99):
        parts += [_tone(f, 0.18, rate, 0.45), gap]
    return np.concatenate(parts)


def synth_security(rate: int = 44100) -> np.ndarray:
    """Loud, harsh two-tone siren (clearly an alarm)."""
    parts = []
    for _ in range(4):
        parts += [_tone(1400, 0.18, rate, 0.9), _tone(800, 0.18, rate, 0.9)]
    return np.concatenate(parts)


class SoundPlayer:
    """Plays pre-built sounds on dedicated mixer channels. ``Sound.play`` /
    ``Channel.play`` return immediately, so the detection loop never waits."""

    def __init__(self, multi_sound, security_sound):
        self._sounds = {"multi": multi_sound, "security": security_sound}

    @classmethod
    def create(cls) -> "SoundPlayer":
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2)
            rate, fmt, channels = pygame.mixer.get_init()
            if fmt != -16:
                raise RuntimeError(f"unsupported mixer format {fmt}")

            def build(mono: np.ndarray, idx: int):
                data = (np.clip(mono, -1, 1) * 32767).astype(np.int16)
                if channels > 1:
                    data = np.repeat(data[:, None], channels, axis=1)
                return _ChannelSound(pygame.mixer.Sound(buffer=data.tobytes()), idx)

            return cls(build(synth_multi_person(rate), 6), build(synth_security(rate), 7))
        except Exception as exc:  # no audio device etc.
            log.warning("Alert audio unavailable (%s); alerts will be silent", exc)
            return cls(None, None)

    def play(self, name: str) -> bool:
        snd = self._sounds.get(name)
        if snd is None:
            return False
        try:
            snd.play()
            return True
        except Exception as exc:
            log.warning("Could not play %s sound: %s", name, exc)
            return False

    def play_multi(self) -> bool:
        return self.play("multi")

    def play_security(self) -> bool:
        return self.play("security")


class _ChannelSound:
    def __init__(self, sound, channel_index: int):
        import pygame
        self.sound, self.channel = sound, pygame.mixer.Channel(channel_index)

    def play(self):
        self.channel.play(self.sound)
