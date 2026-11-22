import csv
import time

import numpy as np
import pytest

from doorbell.config import Config
from doorbell.dashboard import render
from doorbell.database import VisitorDatabase
from doorbell.mask import MaskDetector
from doorbell.pipeline import DoorbellPipeline
from doorbell.alerts import SoundPlayer

SKIN, MASK_BLUE = (105, 172, 224), (220, 170, 120)
FPS = 30


class FakeSound:
    def __init__(self):
        self.plays = 0

    def play(self):
        self.plays += 1


class Scene:
    """Simulated camera: a scripted list of faces per frame."""

    def __init__(self):
        self.faces = []          # list of (box, masked)
        self.frame = None

    def render(self):
        f = np.full((480, 640, 3), 90, np.uint8)
        for (x, y, w, h), masked in self.faces:
            f[y:y + h, x:x + w] = SKIN
            if masked:
                f[y + h // 2:y + h, x:x + w] = MASK_BLUE
        return f

    def detector(self, frame):
        return [b for b, _ in self.faces]


@pytest.fixture
def env(tmp_path):
    cfg = Config(data_dir=tmp_path)
    scene = Scene()
    multi, sec = FakeSound(), FakeSound()
    db = VisitorDatabase(cfg.csv_path)
    pipe = DoorbellPipeline(cfg, db, SoundPlayer(multi, sec), scene.detector,
                            MaskDetector(eye_finder=lambda g: [1, 2]))
    clock = {"t": 1000.0}

    def run(seconds, faces=None):
        if faces is not None:
            scene.faces = faces
        res = None
        for _ in range(int(seconds * FPS)):
            clock["t"] += 1 / FPS
            res = pipe.process(scene.render(), clock["t"])
        return res

    yield type("E", (), dict(cfg=cfg, scene=scene, multi=multi, sec=sec, db=db, pipe=pipe, run=run))
    db.close()


A, B, C = (60, 100, 120, 120), (300, 100, 120, 120), (480, 250, 100, 100)


def csv_rows(env):
    env.db.close()
    with open(env.cfg.csv_path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_single_person_logs_one_entry_no_alerts(env):
    res = env.run(5, [(A, False)])
    assert res.people_count == 1 and not res.multi_active and not res.mask_active
    assert env.multi.plays == 0 and env.sec.plays == 0
    assert env.db.stats()["total_people"] == 1


def test_person_leaving_logs_exit_and_reentry_new_entry(env):
    env.run(2, [(A, False)])
    env.run(4, [])
    env.run(2, [(A, False)])
    kinds = [r["event_type"] for r in csv_rows(env)]
    assert kinds == ["entry", "exit", "entry"]


def test_two_people_one_sound_one_event_and_group_size(env):
    res = env.run(10, [(A, False), (B, False)])      # 300 frames
    assert res.multi_active
    assert env.multi.plays == 1 and env.sec.plays == 0
    rows = csv_rows(env)
    assert [r["event_type"] for r in rows].count("multi_person") == 1
    entries = [r for r in rows if r["event_type"] == "entry"]
    assert len(entries) == 2 and {r["group_size"] for r in entries} == {"2"}


def test_multi_resets_when_one_leaves_and_cooldown_limits_sound(env):
    env.run(3, [(A, False), (B, False)])
    res = env.run(4, [(A, False)])                   # B gone > max_age
    assert not res.multi_active
    env.run(3, [(A, False), (B, False)])             # again inside 10 s cooldown
    assert env.multi.plays == 1
    assert env.db.stats()["multi_total"] == 2


def test_three_people_entering_together(env):
    res = env.run(3, [(A, False), (B, False), (C, False)])
    assert res.people_count == 3
    rows = csv_rows(env)
    assert [r["group_size"] for r in rows if r["event_type"] == "entry"] == ["3"] * 3


def test_masked_face_alerts_logs_and_banner_clears(env):
    res = env.run(0.1, [(A, True)])                   # 3 frames: smoothing not yet satisfied
    assert not res.mask_active and env.sec.plays == 0
    res = env.run(3, [(A, True)])
    assert res.mask_active and env.sec.plays == 1
    assert env.multi.plays == 0
    res = env.run(1, [(A, False)])                    # mask removed, within hysteresis
    res = env.run(3, [(A, False)])
    assert not res.mask_active
    rows = csv_rows(env)
    masks = [r for r in rows if r["event_type"] == "mask_alert"]
    assert len(masks) == 1 and masks[0]["timestamp"] and masks[0]["detail"]
    assert env.sec.plays == 1                         # no spam


def test_visible_face_never_triggers_mask(env):
    res = env.run(10, [(A, False)])
    assert not res.mask_active and res.views[0].status == "visible"
    assert env.sec.plays == 0


def test_unreadable_face_is_unknown_not_masked(env):
    env.pipe.mask = MaskDetector(eye_finder=lambda g: [])   # turned away: no eyes
    res = env.run(10, [(A, True)])
    assert not res.mask_active and res.views[0].status == "unknown"
    assert env.sec.plays == 0 and env.db.stats()["mask_total"] == 0


def test_masked_face_turning_away_is_not_counted_masked(env):
    env.run(3, [(A, True)])
    assert env.sec.plays == 1
    env.pipe.mask = MaskDetector(eye_finder=lambda g: [])
    res = env.run(4, [(A, True)])
    assert not res.mask_active and res.views[0].status == "unknown"


def test_mask_cooldown_after_clear(env):
    env.run(3, [(A, True)])
    env.run(3, [(A, False)])
    env.run(3, [(A, True)])
    assert env.sec.plays == 1                         # second alert inside 8 s cooldown
    assert env.db.stats()["mask_total"] == 2          # but still logged


def test_state_persists_across_restart(env, tmp_path):
    env.run(3, [(A, False), (B, True)])
    before = env.db.stats()
    env.db.close()
    db2 = VisitorDatabase(env.cfg.csv_path)
    assert db2.stats() == before and before["total_people"] == 2 and before["mask_total"] == 1
    db2.close()


def test_processing_is_fast_with_slow_disk(env):
    orig = env.db._flush
    env.db._flush = lambda b: (time.sleep(0.5), orig(b))[1]
    t0 = time.perf_counter()
    env.run(3, [(A, True), (B, False)])
    per_frame = (time.perf_counter() - t0) / 90
    assert per_frame < 0.02, per_frame


def test_dashboard_renders_and_banners_show_and_hide(env):
    frame = env.scene.render()
    res = env.run(1, [(A, False)])
    calm = render(frame, res, env.db)
    assert calm.shape[1] == 640 + 340
    res = env.run(3, [(A, True), (B, False)])
    alert = render(frame, res, env.db)
    top = alert[5:25, 5:300]
    assert (top.reshape(-1, 3) == (0, 0, 200)).all(axis=1).sum() > 500   # red banner
    res = env.run(4, [(A, False)])
    cleared = render(frame, res, env.db)
    assert not (cleared[5:25, 5:300].reshape(-1, 3) == (0, 0, 200)).all(axis=1).any()
    assert not np.array_equal(calm, alert)


def test_dashboard_with_empty_database(tmp_path):
    from doorbell.pipeline import FrameResult
    db = VisitorDatabase(tmp_path / "x.csv")
    out = render(np.zeros((480, 640, 3), np.uint8), FrameResult([], [], 0, False, False), db)
    assert out.shape == (480, 980, 3)
    db.close()
