import os
import time

import numpy as np

from doorbell.alerts import (MaskAlert, MultiPersonAlert, SoundPlayer,
                             synth_multi_person, synth_security)


class FakeSound:
    def __init__(self):
        self.plays = 0

    def play(self):
        self.plays += 1


def test_multi_fires_once_per_event_not_per_frame():
    a, sounds = MultiPersonAlert(2, cooldown=10), 0
    fired = 0
    for i in range(300):                      # 300 frames with 2 people
        d = a.update(2, i * 0.03)
        fired += d.fired
        sounds += d.play_sound
    assert fired == 1 and sounds == 1 and a.active


def test_single_person_never_triggers():
    a = MultiPersonAlert(2)
    assert not any(a.update(1, i).fired for i in range(50))


def test_multi_resets_when_count_drops_and_retriggers():
    a = MultiPersonAlert(2, cooldown=0)
    assert a.update(3, 0).fired
    assert not a.update(1, 1).fired and not a.active
    assert a.update(2, 2).fired


def test_multi_cooldown_suppresses_sound_but_not_event():
    a = MultiPersonAlert(2, cooldown=10)
    d1 = a.update(2, 0)
    a.update(1, 1)
    d2 = a.update(2, 2)                        # inside cooldown
    d3 = a.update(1, 3) and a.update(2, 20)    # after cooldown
    assert d1.play_sound and d2.fired and not d2.play_sound
    assert d3.fired and d3.play_sound


def test_mask_alert_cooldown_and_hysteresis_clear():
    a = MaskAlert(cooldown=8, clear_after=1.5)
    sounds = 0
    for i in range(200):
        sounds += a.update(1, i * 0.03).play_sound
    assert sounds == 1 and a.active
    last = 199 * 0.03
    a.update(0, last + 0.5)
    assert a.active                            # within hysteresis window
    a.update(0, last + 1.4)
    assert a.active
    a.update(0, last + 1.6)
    assert not a.active


def test_mask_alert_flicker_does_not_spam_sound():
    a = MaskAlert(cooldown=8, clear_after=1.0)
    sounds = 0
    for i in range(100):                       # on/off every 0.1s, 10s total
        sounds += a.update(i % 2, i * 0.1).play_sound
    assert sounds <= 2


def test_mask_alert_retrigger_after_clear_and_cooldown():
    a = MaskAlert(cooldown=5, clear_after=1)
    assert a.update(1, 0).play_sound
    a.update(0, 2)
    assert not a.active
    d = a.update(1, 10)
    assert d.fired and d.play_sound


def test_sounds_are_distinct():
    m, s = synth_multi_person(), synth_security()

    def centroid(x):
        spec = np.abs(np.fft.rfft(x))
        return (np.fft.rfftfreq(len(x), 1 / 44100) * spec).sum() / spec.sum()

    def band(x, lo, hi):
        spec = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(len(x), 1 / 44100)
        return spec[(f > lo) & (f < hi)].sum() / spec.sum()
    assert centroid(s) > centroid(m) + 200
    assert band(s, 1300, 1500) > 0.2 > band(m, 1300, 1500)
    assert np.abs(s).max() > np.abs(m).max()
    assert len(m) != len(s)
    assert np.abs(m).max() <= 1 and np.abs(s).max() <= 1


def test_sound_player_routes_and_survives_missing_audio():
    m, s = FakeSound(), FakeSound()
    p = SoundPlayer(m, s)
    assert p.play_multi() and p.play_security()
    assert (m.plays, s.plays) == (1, 1)
    silent = SoundPlayer(None, None)
    assert not silent.play_multi() and not silent.play_security()


def test_sound_player_swallow_playback_errors():
    class Boom:
        def play(self):
            raise RuntimeError("device lost")
    assert SoundPlayer(Boom(), Boom()).play_multi() is False


def test_real_mixer_playback_returns_immediately():
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    import pygame
    pygame.mixer.quit()
    pygame.mixer.init(frequency=44100, size=-16, channels=2)
    p = SoundPlayer.create()
    assert p._sounds["security"] is not None
    t0 = time.perf_counter()
    assert p.play_security() and p.play_multi()
    assert time.perf_counter() - t0 < 0.1       # 1.4s sound must not block
    pygame.mixer.quit()
