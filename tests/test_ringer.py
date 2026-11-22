import wave
from pathlib import Path

import numpy as np

from doorbell.config import Config
from doorbell.ringer import MASK, NORMAL, Ringer


class Fake:
    def __init__(self):
        self.n = 0

    def play(self):
        self.n += 1


def test_ring_routes_to_correct_backend_only():
    a, b = Fake(), Fake()
    r = Ringer({NORMAL: a, MASK: b})
    r.ring(NORMAL)
    assert (a.n, b.n) == (1, 0)
    r.ring(MASK)
    assert (a.n, b.n) == (1, 1)


def test_ring_survives_backend_error():
    class Boom:
        def play(self):
            raise RuntimeError("x")
    assert Ringer({NORMAL: Boom()}).ring(NORMAL) is False


def _load(path):
    with wave.open(str(path)) as w:
        d = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(float)
        if w.getnchannels() == 2:
            d = d[::2]
        return d, w.getframerate()


def test_two_distinct_sound_files_exist_and_original_untouched():
    cfg = Config()
    assert cfg.doorbell_sound != cfg.mask_doorbell_sound
    assert cfg.doorbell_sound.exists() and cfg.mask_doorbell_sound.exists()
    assert cfg.doorbell_sound.read_bytes() != cfg.mask_doorbell_sound.read_bytes()
    a, ra = _load(cfg.doorbell_sound)
    b, rb = _load(cfg.mask_doorbell_sound)

    def centroid(x, r):
        s = np.abs(np.fft.rfft(x))
        return (np.fft.rfftfreq(len(x), 1 / r) * s).sum() / s.sum()
    assert abs(len(a) / ra - len(b) / rb) > 1.0 or abs(centroid(a, ra) - centroid(b, rb)) > 300
    assert b.std() > 1000                       # audible, not silent


def test_real_ringer_plays_both_without_blocking(monkeypatch):
    import os
    import time
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    import pygame
    pygame.mixer.quit()
    pygame.mixer.init(frequency=44100, size=-16, channels=2)
    cfg = Config()
    r = Ringer.create(cfg.doorbell_sound, cfg.mask_doorbell_sound)
    t0 = time.perf_counter()
    assert r.ring(NORMAL) and r.ring(MASK)
    assert time.perf_counter() - t0 < 0.1
    pygame.mixer.quit()
