import numpy as np

from doorbell.mask import (MASKED, UNKNOWN, VISIBLE, MaskDetector, MaskStatusFilter,
                           skin_ratio)

SKIN = (105, 172, 224)       # BGR light skin tone
DARK_SKIN = (60, 100, 150)
MASK_BLUE = (220, 170, 120)  # surgical-mask blue
SIZE = 120
TWO_EYES = lambda g: [(0, 0, 20, 20), (60, 0, 20, 20)]  # noqa: E731
NO_EYES = lambda g: []  # noqa: E731


def make_frame(lower=SKIN, upper=SKIN, brightness=1.0, size=SIZE):
    frame = np.full((400, 400, 3), 90, np.uint8)
    face = np.zeros((size, size, 3), np.uint8)
    face[:] = upper
    face[int(size * 0.5):] = lower
    face = np.clip(face * brightness, 0, 255).astype(np.uint8)
    frame[100:100 + size, 100:100 + size] = face
    return frame, (100, 100, size, size)


def det(eyes=TWO_EYES, **kw):
    return MaskDetector(eye_finder=eyes, **kw)


def test_skin_constants_are_actually_skin():
    assert skin_ratio(np.full((10, 10, 3), SKIN, np.uint8)) == 1.0
    assert skin_ratio(np.full((10, 10, 3), MASK_BLUE, np.uint8)) == 0.0


def test_visible_face():
    f, b = make_frame()
    r = det().classify(f, b)
    assert r.status == VISIBLE and r.confidence > 0.9


def test_masked_face():
    f, b = make_frame(lower=MASK_BLUE)
    r = det().classify(f, b)
    assert r.status == MASKED and r.confidence > 0.9


def test_turned_away_or_no_eyes_is_unknown_never_masked():
    f, b = make_frame(lower=MASK_BLUE)       # would look "masked" by colour alone
    assert det(eyes=NO_EYES).classify(f, b).status == UNKNOWN


def test_small_face_unknown():
    f, b = make_frame(lower=MASK_BLUE, size=60)
    assert det().classify(f, b).status == UNKNOWN


def test_face_touching_frame_edge_unknown():
    f, _ = make_frame(lower=MASK_BLUE)
    assert det().classify(f, (0, 100, SIZE, SIZE)).status == UNKNOWN
    assert det().classify(f, (300, 100, SIZE, SIZE)).status == UNKNOWN


def test_dark_or_overexposed_unknown():
    f, b = make_frame(lower=MASK_BLUE, brightness=0.15)
    assert det().classify(f, b).status == UNKNOWN
    f, b = make_frame(lower=(255, 255, 255), upper=(250, 250, 250))
    assert det().classify(f, b).status == UNKNOWN


def test_uncalibratable_skin_unknown():
    f, b = make_frame(lower=MASK_BLUE, upper=(30, 30, 30))   # hair/hat over forehead
    assert det().classify(f, b).status == UNKNOWN


def test_ambiguous_ratio_unknown():
    f, b = make_frame()
    x, y, w, h = b
    # half of the lower band becomes mask-coloured -> ratio ~0.5
    f[y + int(h * 0.5):y + h, x:x + w // 2] = MASK_BLUE
    r = det(min_ref_skin=0.3).classify(f, b)
    assert r.status == UNKNOWN


def test_real_cascade_on_blank_frame_is_unknown_not_crash():
    f = np.full((300, 300, 3), 120, np.uint8)
    assert MaskDetector().classify(f, (80, 80, 120, 120)).status == UNKNOWN


def test_filter_requires_sustained_masked():
    flt = MaskStatusFilter()
    out = [flt.observe(1, MASKED) for _ in range(5)]
    assert out[:4] == [UNKNOWN] * 4 and out[4] == MASKED


def test_filter_single_visible_blip_does_not_clear_masked():
    flt = MaskStatusFilter()
    for _ in range(6):
        flt.observe(1, MASKED)
    assert flt.observe(1, VISIBLE) == MASKED


def test_filter_unknown_stream_never_masked_and_visible_confirmed():
    flt = MaskStatusFilter()
    assert all(flt.observe(1, UNKNOWN) == UNKNOWN for _ in range(20))
    assert [flt.observe(2, VISIBLE) for _ in range(3)][-1] == VISIBLE


def test_filter_forget():
    flt = MaskStatusFilter()
    for _ in range(6):
        flt.observe(1, MASKED)
    flt.forget(1)
    assert flt.observe(1, MASKED) == UNKNOWN
