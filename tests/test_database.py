import csv
import time
from datetime import datetime, date

from doorbell.database import FIELDS, VisitorDatabase, EVENT_ENTRY, EVENT_MULTI, EVENT_MASK


def rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


def test_auto_creates_file_with_header(tmp_path):
    p = tmp_path / "sub" / "log.csv"
    db = VisitorDatabase(p)
    db.close()
    assert rows(p) == [FIELDS]


def test_preserves_existing_rows_and_appends(tmp_path):
    p = tmp_path / "log.csv"
    db = VisitorDatabase(p)
    db.log_event(EVENT_ENTRY, "a-1", 1, 1, ts=datetime(2026, 1, 1, 9, 0, 0))
    db.close()
    db = VisitorDatabase(p)
    db.log_event(EVENT_ENTRY, "b-1", 1, 1, ts=datetime(2026, 1, 2, 10, 30, 5))
    db.close()
    r = rows(p)
    assert len(r) == 3 and r[1][4] == "a-1" and r[2][4] == "b-1"
    assert r[2][1] == "2026-01-02" and r[2][2] == "10:30:05"
    assert r[2][0].startswith("2026-01-02T10:30:05")


def test_stats_persist_across_restart(tmp_path):
    p = tmp_path / "log.csv"
    today = datetime.now()
    db = VisitorDatabase(p)
    for i in range(3):
        db.log_event(EVENT_ENTRY, f"s-{i}", 1, 1, ts=today)
    db.log_event(EVENT_ENTRY, "old", 1, 1, ts=datetime(2020, 1, 1))
    db.log_event(EVENT_MULTI, "", 2, 2, ts=today)
    db.log_event(EVENT_MASK, "s-1", 1, 1, ts=today)
    db.close()
    db2 = VisitorDatabase(p)
    s = db2.stats()
    assert s == {"total_people": 4, "entries_today": 3, "multi_total": 1, "multi_today": 1,
                 "mask_total": 1, "mask_today": 1}
    assert db2.daily_counts()[date.today().isoformat()] == 3
    assert db2.recent(EVENT_MASK)[0]["visitor_id"] == "s-1"
    db2.close()


def test_multi_person_group_logged(tmp_path):
    p = tmp_path / "log.csv"
    db = VisitorDatabase(p)
    db.log_event(EVENT_ENTRY, "x-1", 3, 3)
    db.log_event(EVENT_ENTRY, "x-2", 3, 3)
    db.log_event(EVENT_ENTRY, "x-3", 3, 3)
    db.close()
    assert [r[6] for r in rows(p)[1:]] == ["3", "3", "3"]


def test_empty_file_handled(tmp_path):
    p = tmp_path / "log.csv"
    p.write_text("")
    db = VisitorDatabase(p)
    db.log_event(EVENT_ENTRY, "a", 1, 1)
    db.close()
    assert rows(p)[0] == FIELDS and len(rows(p)) == 2


def test_malformed_rows_skipped_valid_kept_backup_made(tmp_path):
    p = tmp_path / "log.csv"
    good = ",".join(["2026-03-01T08:00:00.000", "2026-03-01", "08:00:00", "entry", "g-1", "1", "1", ""])
    p.write_text(",".join(FIELDS) + "\n" + good + "\ngarbage line\n2026-13-99,x,y,entry,a,b,c,d\n")
    db = VisitorDatabase(p)
    assert db.skipped_rows == 2
    assert db.stats()["total_people"] == 1
    assert db.recovered_from and db.recovered_from.exists()
    db.log_event(EVENT_ENTRY, "n-1", 1, 1)
    db.close()
    r = rows(p)
    assert r[0] == FIELDS and len(r) == 3


def test_binary_garbage_file_handled(tmp_path):
    p = tmp_path / "log.csv"
    p.write_bytes(b"\x00\xff\xfe\x01garbage\x00\x00" * 50)
    db = VisitorDatabase(p)
    db.log_event(EVENT_ENTRY, "a", 1, 1)
    db.close()
    assert rows(p)[0] == FIELDS and len(rows(p)) == 2
    assert db.recovered_from.exists()


def test_missing_header_recovered(tmp_path):
    p = tmp_path / "log.csv"
    p.write_text("2026-03-01T08:00:00.000,2026-03-01,08:00:00,entry,g-1,1,1,\n")
    db = VisitorDatabase(p)
    db.close()
    assert db.stats()["total_people"] == 1
    assert rows(p)[0] == FIELDS


def test_missing_trailing_newline_does_not_merge_rows(tmp_path):
    p = tmp_path / "log.csv"
    db = VisitorDatabase(p)
    db.close()
    with open(p, "a", newline="") as fh:   # valid row, no newline
        fh.write("2026-03-01T08:00:00.000,2026-03-01,08:00:00,entry,g-1,1,1,")
    db = VisitorDatabase(p)
    db.log_event(EVENT_ENTRY, "n", 1, 1)
    db.close()
    r = rows(p)
    assert len(r) == 3 and all(len(x) == len(FIELDS) for x in r)


def test_log_event_never_blocks_on_slow_disk(tmp_path):
    db = VisitorDatabase(tmp_path / "log.csv")
    orig = db._flush

    def slow(batch):
        time.sleep(0.3)
        return orig(batch)
    db._flush = slow
    t0 = time.perf_counter()
    for i in range(500):
        db.log_event(EVENT_ENTRY, f"v-{i}", 1, 1)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.25, elapsed
    db.close(timeout=10)
    assert len(rows(tmp_path / "log.csv")) == 501


def test_write_failure_is_retried_not_lost(tmp_path):
    db = VisitorDatabase(tmp_path / "log.csv")
    orig, state = db._flush, {"n": 0}

    def flaky(batch):
        state["n"] += 1
        if state["n"] == 1:
            db.write_errors += 1
            return False
        return orig(batch)
    db._flush = flaky
    db.log_event(EVENT_ENTRY, "v", 1, 1)
    db.close(timeout=10)
    assert db.write_errors == 1 and len(rows(tmp_path / "log.csv")) == 2


def test_unknown_event_rejected(tmp_path):
    import pytest
    db = VisitorDatabase(tmp_path / "log.csv")
    with pytest.raises(ValueError):
        db.log_event("bogus")
    db.close()
