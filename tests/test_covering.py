"""Face-covering detection with the landmark pipeline (YuNet-style Face objects).

Frames are synthetic; the thresholds themselves were tuned on real webcam
recordings (see tools/replay_frames.py)."""
import numpy as np
import pytest

from doorbell.alerts import SoundPlayer
from doorbell.config import Config
from doorbell.database import VisitorDatabase
from doorbell.detectors import Face
from doorbell.mask import MASKED, UNKNOWN, VISIBLE, CoverAnalyzer, MaskStatusFilter
from doorbell.pipeline import DoorbellPipeline

SKIN, LIPS, BG = (105, 172, 224), (110, 100, 200), (90, 90, 90)
CLOTH = (60, 60, 60)          # dark scarf / cloth
SCARF = (40, 40, 200)         # red scarf
OBJECT = (200, 120, 40)       # blue phone / paper
FPS = 30

# kind -> (score the detector would give, painter)
SCORES = {"clear": 0.93, "hand_mouth": 0.80, "palm": 0.42, "cloth": 0.50,
          "scarf": 0.50, "object": 0.50, "mask": 0.55}


def make_face(x=250, y=150, w=140, kind="clear", score=None, lighting=1.0):
    h = int(w * 1.25)
    box = (x, y, w, h)
    lm = np.array([[x + .3 * w, y + .38 * h], [x + .7 * w, y + .38 * h], [x + .5 * w, y + .58 * h],
                   [x + .35 * w, y + .78 * h], [x + .65 * w, y + .78 * h]])
    return Face(box, SCORES[kind] if score is None else score, lm), kind, lighting


def paint(frame, spec):
    face, kind, lighting = spec
    x, y, w, h = face.box
    patch = np.zeros((h, w, 3), np.uint8)
    patch[:] = SKIN
    if kind == "clear":                                   # visible lips + mouth shadow
        patch[int(.70 * h):int(.86 * h), int(.30 * w):int(.70 * w)] = LIPS
    elif kind == "hand_mouth":                            # flat skin-toned hand over mouth/nose
        patch[int(.5 * h):int(.95 * h), int(.1 * w):int(.9 * w)] = (96, 162, 215)
    elif kind == "palm":                                  # flat palm over the whole face
        patch[:] = (96, 162, 215)
    elif kind in ("cloth", "scarf", "object", "mask"):
        colour = {"cloth": CLOTH, "scarf": SCARF, "object": OBJECT, "mask": (220, 170, 120)}[kind]
        patch[int(.5 * h):] = colour
    patch = np.clip(patch * lighting, 0, 255).astype(np.uint8)
    y0, y1, x0, x1 = max(y, 0), min(y + h, frame.shape[0]), max(x, 0), min(x + w, frame.shape[1])
    frame[y0:y1, x0:x1] = patch[y0 - y:y1 - y, x0 - x:x1 - x]


def frame_for(*specs):
    f = np.full((480, 640, 3), BG, np.uint8)
    for s in specs:
        paint(f, s)
    return f


# ----------------------------------------------------------------- analyzer
@pytest.mark.parametrize("kind", ["cloth", "scarf", "object", "mask"])
def test_non_skin_cover_is_strong_covered(kind):
    spec = make_face(kind=kind)
    r = CoverAnalyzer().analyze(frame_for(spec), spec[0])
    assert r.status == MASKED and r.strong and r.category == "mask/cloth/object"


def test_clear_face_is_visible_and_never_covered():
    spec = make_face(kind="clear")
    r = CoverAnalyzer().analyze(frame_for(spec), spec[0])
    assert r.status == VISIBLE and r.category == "normal"


def test_hand_over_mouth_is_weak_covered():
    spec = make_face(kind="hand_mouth")
    r = CoverAnalyzer().analyze(frame_for(spec), spec[0])
    assert r.status == MASKED and not r.strong and r.category == "hand/skin-tone cover"


def test_palm_over_face_low_face_score_is_weak_covered():
    spec = make_face(kind="palm")
    r = CoverAnalyzer().analyze(frame_for(spec), spec[0])
    assert r.status == MASKED and not r.strong


def test_poor_lighting_is_unknown():
    spec = make_face(kind="cloth", lighting=0.12)
    assert CoverAnalyzer().analyze(frame_for(spec), spec[0]).status == UNKNOWN


def test_face_partly_below_frame_is_unknown_not_covered():
    spec = make_face(y=330, kind="cloth")                 # chin/mouth fall off the bottom
    r = CoverAnalyzer().analyze(frame_for(spec), spec[0])
    assert r.status == UNKNOWN and r.category == "partial"


def test_small_face_unknown():
    spec = make_face(w=50, kind="cloth")
    assert CoverAnalyzer().analyze(frame_for(spec), spec[0]).status == UNKNOWN


def test_mid_score_clear_looking_face_is_uncertain_not_covered():
    spec = make_face(kind="clear", score=0.8)             # e.g. head slightly turned
    r = CoverAnalyzer().analyze(frame_for(spec), spec[0])
    assert r.status == UNKNOWN and r.category == "partial"


# ------------------------------------------------------------------- filter
def feed(flt, seconds, status, strong, start=0.0, fps=FPS):
    out = None
    for i in range(int(seconds * fps)):
        out = flt.observe(1, status, strong, start + i / fps)
    return out


def test_weak_evidence_needs_sustained_time():
    flt = MaskStatusFilter()
    assert feed(flt, 0.8, MASKED, False) != MASKED         # brief head turn / blur
    assert feed(flt, 2.0, MASKED, False, start=0.8) == MASKED


def test_interrupted_weak_evidence_never_confirms():
    flt = MaskStatusFilter()
    t, got = 0.0, set()
    for _ in range(6):                                     # 0.5 s covered / 0.5 s clear
        got.add(feed(flt, 0.5, MASKED, False, t)); t += 0.5
        got.add(feed(flt, 0.5, VISIBLE, True, t)); t += 0.5
    assert MASKED not in got


def test_strong_evidence_is_fast():
    flt = MaskStatusFilter()
    assert feed(flt, 0.25, MASKED, True) == MASKED


# ----------------------------------------------------------------- pipeline
@pytest.fixture
def env(tmp_path):
    cfg = Config(data_dir=tmp_path)
    state = {"faces": [], "persons": [], "t": 1000.0}
    multi, sec, rings = type("S", (), {"plays": 0, "play": lambda s: setattr(s, "plays", s.plays + 1)})(), \
        type("S", (), {"plays": 0, "play": lambda s: setattr(s, "plays", s.plays + 1)})(), []
    db = VisitorDatabase(cfg.csv_path)
    pipe = DoorbellPipeline(cfg, db, SoundPlayer(multi, sec),
                            face_detector=lambda f: [s[0] for s in state["faces"]],
                            person_detector=lambda f: list(state["persons"]))

    def run(seconds, faces=None, persons=None):
        if faces is not None:
            state["faces"] = faces
        if persons is not None:
            state["persons"] = persons
        res = None
        for _ in range(int(seconds * FPS)):
            state["t"] += 1 / FPS
            res = pipe.process(frame_for(*state["faces"]), state["t"])
            if res.ring:
                rings.append(res.ring)
        return res
    yield type("E", (), dict(run=run, rings=rings, sec=sec, db=db, pipe=pipe))
    db.close()


def person_for(spec):                       # SSD-style person box around the face
    x, y, w, h = spec[0].box
    return (x - w // 2, y - 10, 2 * w, 480 - y + 10)


def test_normal_visitor_gets_normal_bell(env):
    s = make_face(kind="clear")
    res = env.run(8, [s], [person_for(s)])
    assert env.rings == ["normal"] and not res.mask_active and env.sec.plays == 0


@pytest.mark.parametrize("kind", ["mask", "hand_mouth", "palm", "cloth", "scarf", "object"])
def test_every_covering_type_rings_the_same_special_bell_once(env, kind):
    s = make_face(kind=kind)
    res = env.run(10, [s], [])                          # person detector finds nothing
    assert env.rings == ["mask"], (kind, env.rings)
    assert res.mask_active and env.sec.plays == 0       # one bell, no stacked siren
    assert res.views[0].status == "masked"
    assert env.db.stats()["mask_total"] == 1 and env.db.stats()["total_people"] == 1


def test_covered_face_found_via_person_box_when_face_is_covered_and_weak(env):
    s = make_face(kind="cloth")
    env.run(8, [s], [person_for(s)])
    assert env.rings == ["mask"]


def test_covering_removed_clears_alert(env):
    s = make_face(kind="cloth")
    env.run(5, [s], [])
    res = env.run(5, [make_face(kind="clear")], [])
    assert not res.mask_active


def test_late_covering_after_normal_arrival_sounds_alert_not_second_bell(env):
    env.run(5, [make_face(kind="clear")], [])
    env.run(5, [make_face(kind="cloth")], [])
    assert env.rings == ["normal"] and env.sec.plays == 1


def test_person_with_no_detectable_face_is_not_treated_as_covered(env):
    p = (200, 100, 280, 380)
    res = env.run(8, [], [p])                            # person seen, face totally absent
    assert env.rings == ["normal"] and not res.mask_active and env.sec.plays == 0
    assert res.views[0].category == "partial"


def test_empty_scene_does_nothing(env):
    res = env.run(5, [], [])
    assert env.rings == [] and res.people_count == 0


def test_tiny_weak_blob_does_not_create_a_visitor(env):
    s = make_face(x=500, y=300, w=45, kind="palm", score=0.45)
    env.run(5, [s], [])
    assert env.db.stats()["total_people"] == 0 and env.rings == []


def test_poor_lighting_never_triggers_special_bell(env):
    s = make_face(kind="cloth", lighting=0.12)
    res = env.run(10, [s], [])
    assert env.rings in ([], ["normal"]) and not res.mask_active and env.sec.plays == 0


def test_partial_face_below_frame_no_special_bell(env):
    s = make_face(y=330, kind="cloth")
    res = env.run(10, [s], [])
    assert not res.mask_active and "mask" not in env.rings


def test_brief_head_turn_does_not_ring_special(env):
    env.run(2, [make_face(kind="clear")], [])
    env.run(0.7, [make_face(kind="palm")], [])           # 0.7 s weak face score only
    res = env.run(2, [make_face(kind="clear")], [])
    assert not res.mask_active and "mask" not in env.rings and env.sec.plays == 0


def test_standing_still_covered_rings_once(env):
    s = make_face(kind="hand_mouth")
    env.run(30, [s], [])
    assert env.rings == ["mask"]


def test_leave_and_return_with_cover_rings_again(env):
    env.run(8, [make_face(kind="clear")], [])
    env.run(4, [], [])
    env.run(8, [make_face(kind="scarf")], [])
    env.run(4, [], [])
    env.run(8, [make_face(kind="clear")], [])
    assert env.rings == ["normal", "mask", "normal"]


def test_group_with_one_covered_rings_single_special_bell(env):
    a, b = make_face(x=60, kind="clear"), make_face(x=380, kind="scarf")
    res = env.run(8, [a, b], [])
    assert env.rings == ["mask"] and res.people_count == 2
    assert env.db.stats()["total_people"] == 2 and env.db.stats()["multi_total"] == 1


def test_csv_tracking_still_logs_entries_and_alerts(env):
    env.run(8, [make_face(kind="cloth")], [])
    env.pipe.db.close()
    import csv
    with open(env.pipe.cfg.csv_path, newline="") as fh:
        kinds = [r["event_type"] for r in csv.DictReader(fh)]
    assert kinds.count("entry") == 1 and kinds.count("mask_alert") == 1
