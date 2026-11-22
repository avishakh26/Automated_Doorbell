"""Draws the live dashboard (stats side panel, alert banners, face boxes)."""
from datetime import datetime
from typing import Dict, List

import cv2
import numpy as np

from .database import EVENT_ENTRY, EVENT_MASK, EVENT_MULTI, VisitorDatabase
from .mask import MASKED, VISIBLE
from .pipeline import FrameResult

PANEL_W = 340
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLORS = {MASKED: (0, 0, 255), VISIBLE: (0, 255, 0)}   # BGR; anything else -> yellow
UNKNOWN_COLOR = (0, 215, 255)


def _text(img, s, x, y, scale=0.5, color=(235, 235, 235), thick=1):
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def _banner(img, y, msg, bg):
    cv2.rectangle(img, (0, y), (img.shape[1], y + 32), bg, -1)
    _text(img, msg, 10, y + 22, 0.65, (255, 255, 255), 2)
    return y + 32


def _event_line(rec: Dict[str, str]) -> str:
    who = rec["visitor_id"].split("-")[-1]
    if rec["event_type"] == EVENT_ENTRY:
        extra = f" (group of {rec['group_size']})" if int(rec["group_size"] or 0) > 1 else ""
        return f"{rec['time']}  visitor #{who}{extra}"
    if rec["event_type"] == EVENT_MULTI:
        return f"{rec['time']}  {rec['people_in_frame']} people together"
    return f"{rec['time']}  face covering"


def _ring_text(last):
    if not last:
        return "none yet"
    kind, ts = last
    return f"{kind.upper()} @ {datetime.fromtimestamp(ts):%H:%M:%S}"


def render(frame: np.ndarray, result: FrameResult, db: VisitorDatabase) -> np.ndarray:
    out = frame.copy()
    for v in result.views:
        x, y, w, h = v.box
        color = COLORS.get(v.status, UNKNOWN_COLOR)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        label = {MASKED: "COVERED (" + v.category + ")", VISIBLE: "normal"}.get(v.status, v.category)
        _text(out, f"#{v.track_id} {label}", x, max(14, y - 6), 0.5, color, 1)
    y = 0
    if result.mask_active:
        y = _banner(out, y, "SECURITY ALERT: FACE COVERING DETECTED", (0, 0, 200))
    if result.multi_active:
        y = _banner(out, y, f"MULTIPLE PEOPLE AT DOOR: {result.people_count}", (0, 120, 230))

    h = max(out.shape[0], 380)
    if out.shape[0] < h:
        out = cv2.copyMakeBorder(out, 0, h - out.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    panel = np.full((h, PANEL_W, 3), 32, np.uint8)
    s = db.stats()
    _text(panel, "DOORBELL DASHBOARD", 12, 26, 0.6, (255, 255, 255), 2)
    rows = [("People in view now", result.people_count),
            ("Total people detected", s["total_people"]),
            ("Visitor entries today", s["entries_today"]),
            ("Multi-person events", f"{s['multi_today']} today / {s['multi_total']}"),
            ("Mask alerts", f"{s['mask_today']} today / {s['mask_total']}"),
            ("Last doorbell", _ring_text(result.last_ring))]
    yy = 56
    for label, val in rows:
        _text(panel, label, 12, yy)
        _text(panel, str(val), 215, yy, 0.5, (120, 255, 120), 1)
        yy += 22
    for title, types, color in (("Recent visitor activity", [EVENT_ENTRY], (255, 200, 120)),
                                ("Latest security alerts", [EVENT_MASK, EVENT_MULTI], (120, 120, 255))):
        yy += 14
        _text(panel, title, 12, yy, 0.55, color, 1)
        recs = db.recent(types, 5)
        if not recs:
            yy += 20
            _text(panel, "none yet", 20, yy, 0.45, (150, 150, 150))
        for rec in recs:
            yy += 20
            _text(panel, _event_line(rec), 20, yy, 0.45)
    return np.hstack([out, panel])
