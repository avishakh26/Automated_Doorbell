"""Persistent, non-blocking CSV event log for visitors and security alerts.

All public calls return immediately: records are applied to in-memory
statistics synchronously and written to disk by a background thread.
"""
import csv
import io
import logging
import queue
import threading
import time
from collections import deque
from datetime import date, datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional

log = logging.getLogger(__name__)

FIELDS = ["timestamp", "date", "time", "event_type", "visitor_id",
          "people_in_frame", "group_size", "detail"]

EVENT_ENTRY = "entry"
EVENT_EXIT = "exit"
EVENT_MULTI = "multi_person"
EVENT_MASK = "mask_alert"
EVENTS = {EVENT_ENTRY, EVENT_EXIT, EVENT_MULTI, EVENT_MASK}

_SENTINEL = object()


def _valid_row(row: List[str]) -> Optional[Dict[str, str]]:
    if len(row) != len(FIELDS):
        return None
    rec = dict(zip(FIELDS, row))
    if rec["event_type"] not in EVENTS:
        return None
    try:
        ts = datetime.fromisoformat(rec["timestamp"])
        int(rec["people_in_frame"] or 0)
        int(rec["group_size"] or 0)
    except ValueError:
        return None
    if rec["date"] != ts.date().isoformat():
        return None
    return rec


class VisitorDatabase:
    def __init__(self, path, recent_limit: int = 50):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._queue: "queue.Queue" = queue.Queue()
        self._recent: Deque[Dict[str, str]] = deque(maxlen=recent_limit)
        self._counts: Dict[str, Dict[str, int]] = {e: {} for e in EVENTS}  # event -> date -> n
        self.skipped_rows = 0
        self.write_errors = 0
        self.recovered_from: Optional[Path] = None
        self._load()
        self._thread = threading.Thread(target=self._writer, name="csv-writer", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------ loading
    def _load(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or self.path.stat().st_size == 0:
            self._write_fresh([])
            return
        try:
            text = self.path.read_bytes().decode("utf-8-sig", errors="replace").replace("\x00", "")
            rows = list(csv.reader(io.StringIO(text, newline="")))
        except (OSError, csv.Error) as exc:
            log.warning("Could not read %s (%s); starting a fresh log", self.path, exc)
            rows = []
        damaged = False
        good: List[Dict[str, str]] = []
        if not rows or rows[0] != FIELDS:
            damaged = True
            body = rows
        else:
            body = rows[1:]
        for row in body:
            if not row:
                continue
            rec = _valid_row(row)
            if rec is None:
                self.skipped_rows += 1
                damaged = True
            else:
                good.append(rec)
        if damaged:
            # Keep the original untouched as a backup; rewrite a clean file.
            backup = self.path.with_name(
                f"{self.path.stem}.corrupt-{datetime.now():%Y%m%d-%H%M%S}.bak")
            try:
                self.path.replace(backup)
                self.recovered_from = backup
                log.warning("Damaged log moved to %s; %d valid rows kept", backup, len(good))
            except OSError as exc:
                log.warning("Could not back up damaged log: %s", exc)
            self._write_fresh(good)
        for rec in good:
            self._apply(rec)

    def _write_fresh(self, records: List[Dict[str, str]]) -> None:
        with open(self.path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(FIELDS)
            for rec in records:
                w.writerow([rec[f] for f in FIELDS])

    # ----------------------------------------------------------- logging
    def _apply(self, rec: Dict[str, str]) -> None:
        by_day = self._counts[rec["event_type"]]
        by_day[rec["date"]] = by_day.get(rec["date"], 0) + 1
        if rec["event_type"] != EVENT_EXIT:
            self._recent.append(rec)

    def log_event(self, event_type: str, visitor_id: str = "", people_in_frame: int = 0,
                  group_size: int = 0, detail: str = "",
                  ts: Optional[datetime] = None) -> Dict[str, str]:
        if event_type not in EVENTS:
            raise ValueError(f"unknown event type {event_type!r}")
        ts = ts or datetime.now()
        rec = {
            "timestamp": ts.isoformat(timespec="milliseconds"),
            "date": ts.date().isoformat(),
            "time": ts.strftime("%H:%M:%S"),
            "event_type": event_type,
            "visitor_id": str(visitor_id),
            "people_in_frame": str(int(people_in_frame)),
            "group_size": str(int(group_size)),
            "detail": detail.replace("\r", " ").replace("\n", " "),
        }
        with self._lock:
            self._apply(rec)
        self._queue.put(rec)
        return rec

    # ------------------------------------------------------------ queries
    def stats(self, today: Optional[date] = None) -> Dict[str, int]:
        day = (today or date.today()).isoformat()
        with self._lock:
            c = self._counts
            return {
                "total_people": sum(c[EVENT_ENTRY].values()),
                "entries_today": c[EVENT_ENTRY].get(day, 0),
                "multi_total": sum(c[EVENT_MULTI].values()),
                "multi_today": c[EVENT_MULTI].get(day, 0),
                "mask_total": sum(c[EVENT_MASK].values()),
                "mask_today": c[EVENT_MASK].get(day, 0),
            }

    def daily_counts(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counts[EVENT_ENTRY])

    def recent(self, event_types, n: int = 5) -> List[Dict[str, str]]:
        """Newest-first events of the given types."""
        types = {event_types} if isinstance(event_types, str) else set(event_types)
        with self._lock:
            return [r for r in reversed(self._recent) if r["event_type"] in types][:n]

    # ------------------------------------------------------------- writer
    def _flush(self, batch: List[Dict[str, str]]) -> bool:
        try:
            needs_nl = False
            if self.path.exists() and self.path.stat().st_size > 0:
                with open(self.path, "rb") as fh:
                    fh.seek(-1, 2)
                    needs_nl = fh.read(1) not in (b"\n", b"\r")
            with open(self.path, "a", newline="", encoding="utf-8") as fh:
                if needs_nl:
                    fh.write("\r\n")
                w = csv.writer(fh)
                for rec in batch:
                    w.writerow([rec[f] for f in FIELDS])
            return True
        except OSError as exc:  # e.g. file locked by a spreadsheet; retry later
            self.write_errors += 1
            log.warning("CSV write failed (%s); will retry", exc)
            return False

    def _writer(self) -> None:
        pending: List[Dict[str, str]] = []
        stopping = False
        while True:
            try:
                item = self._queue.get(timeout=0.25)
                if item is _SENTINEL:
                    stopping = True
                else:
                    pending.append(item)
                    while True:  # drain what is already queued into one write
                        try:
                            item = self._queue.get_nowait()
                        except queue.Empty:
                            break
                        if item is _SENTINEL:
                            stopping = True
                        else:
                            pending.append(item)
            except queue.Empty:
                pass
            if pending and self._flush(pending):
                pending = []
            elif pending and not stopping:
                time.sleep(0.5)
            if stopping:
                if pending:
                    self._flush(pending)
                return

    def close(self, timeout: float = 5.0) -> None:
        self._queue.put(_SENTINEL)
        self._thread.join(timeout)
