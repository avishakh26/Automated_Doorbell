from doorbell.tracker import FaceTracker

B = (100, 100, 80, 80)


def shifted(b, dx):
    return (b[0] + dx, b[1], b[2], b[3])


def test_same_face_many_frames_is_one_confirmed_entry():
    tr, entries = FaceTracker(min_hits=3), 0
    for i in range(100):
        entries += len(tr.update([shifted(B, i % 3)], i * 0.03).new_confirmed)
    assert entries == 1


def test_flicker_not_confirmed():
    tr = FaceTracker(min_hits=3)
    seen = 0
    for i in range(20):
        boxes = [B] if i % 2 == 0 else []
        seen += len(tr.update(boxes, i * 0.03).new_confirmed)
    assert seen == 0


def test_short_dropout_keeps_identity():
    tr = FaceTracker(min_hits=2, max_age=2.0)
    ids = set()
    for i, boxes in enumerate([[B], [B], [], [], [], [B], [B]]):
        u = tr.update(boxes, i * 0.1)
        ids |= {t.id for t in u.active}
    assert len(ids) == 1


def test_expiry_reports_lost_and_reentry_is_new_entry():
    tr = FaceTracker(min_hits=2, max_age=1.0)
    for i in range(3):
        tr.update([B], i * 0.1)
    lost = tr.update([], 5.0).lost
    assert len(lost) == 1
    entries = 0
    for i in range(3):
        entries += len(tr.update([B], 6 + i * 0.1).new_confirmed)
    assert entries == 1


def test_two_people_two_tracks_same_frame_group():
    tr = FaceTracker(min_hits=2)
    other = (400, 100, 80, 80)
    tr.update([B, other], 0.0)
    u = tr.update([B, other], 0.1)
    assert len(u.new_confirmed) == 2 and len(u.active) == 2
    assert len({t.id for t in u.active}) == 2


def test_person_entering_later_adds_track_without_duplicating_first():
    tr = FaceTracker(min_hits=2)
    other = (400, 100, 80, 80)
    total = 0
    for i in range(6):
        total += len(tr.update([B], i * 0.1).new_confirmed)
    for i in range(6, 12):
        total += len(tr.update([B, other], i * 0.1).new_confirmed)
    assert total == 2


def test_invisible_track_flagged_not_visible():
    tr = FaceTracker(min_hits=1)
    tr.update([B], 0.0)
    u = tr.update([], 0.1)
    assert len(u.active) == 1 and not u.active[0].visible
