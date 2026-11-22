"""Generates doorbell_mask.wav: a bright, staccato rising triple-note fanfare
played twice -- clearly different from doorbell.wav (slow 600 Hz bell)."""
import wave
from pathlib import Path

import numpy as np

RATE = 44100


def note(freq, dur, vol=0.7):
    t = np.arange(int(RATE * dur)) / RATE
    w = np.sin(2 * np.pi * freq * t) + 0.4 * np.sin(2 * np.pi * 2 * freq * t)
    env = np.exp(-t * 6) * np.minimum(1, t * 400)
    return w / 1.4 * env * vol


def build():
    gap = np.zeros(int(RATE * 0.06))
    seq = []
    for _ in range(2):
        for f in (880.0, 1174.66, 1567.98):
            seq += [note(f, 0.22), gap]
        seq.append(np.zeros(int(RATE * 0.25)))
    return np.concatenate(seq)


if __name__ == "__main__":
    mono = np.clip(build(), -1, 1)
    data = np.repeat((mono * 32767).astype(np.int16)[:, None], 2, axis=1)
    out = Path(__file__).resolve().parent.parent / "doorbell_mask.wav"
    with wave.open(str(out), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(data.tobytes())
    print("wrote", out)
