"""Smoke test of the real main loop with a simulated camera, window and audio."""
import os
import sys

import numpy as np

os.environ["SDL_AUDIODRIVER"] = "dummy"
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2  # noqa: E402
import pygame  # noqa: E402

import main  # noqa: E402
from doorbell import pipeline  # noqa: E402
from doorbell.config import Config  # noqa: E402

BOX = (60, 100, 120, 120)


class FakeCap:
    def __init__(self, n):
        self.n, self.released = n, False

    def read(self):
        if self.n <= 0:
            return False, None
        self.n -= 1
        return True, np.full((480, 640, 3), 90, np.uint8)

    def release(self):
        self.released = True


def test_main_loop_rings_doorbell_logs_and_cleans_up(tmp_path, monkeypatch):
    cfg = Config(data_dir=tmp_path)
    cap = FakeCap(90)
    shown, rings = [], []
    monkeypatch.setattr(main, "Config", lambda: cfg)
    monkeypatch.setattr(cv2, "VideoCapture", lambda i: cap)
    monkeypatch.setattr(cv2, "imshow", lambda name, img: shown.append(img.shape))
    monkeypatch.setattr(cv2, "waitKey", lambda ms: 0)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(main, "load_detectors", lambda c: (None, lambda f: [BOX]))
    monkeypatch.setattr(pygame.mixer.music, "play", lambda *a, **k: rings.append(1))
    clock = iter(1000.0 + i / 30 for i in range(10**6))        # simulated 30 fps clock
    monkeypatch.setattr(main.time, "time", lambda: next(clock))
    main.main()
    assert cap.released and len(shown) == 90 and shown[0][1] == 980
    assert len(rings) == 1                                   # one ring per arrival, not per frame
    lines = cfg.csv_path.read_text().splitlines()
    assert len(lines) == 2 and ",entry," in lines[1]         # one person, one entry
