"""Detection pipeline: faces -> tracks -> visitor log, multi-person and mask alerts."""
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .alerts import MaskAlert, MultiPersonAlert, SoundPlayer
from .config import Config
from .database import EVENT_ENTRY, EVENT_EXIT, EVENT_MASK, EVENT_MULTI, VisitorDatabase
from .mask import MASKED, UNKNOWN, VISIBLE, MaskDetector, MaskStatusFilter
from .ringer import MASK as RING_MASK, NORMAL as RING_NORMAL
from .tracker import FaceTracker

Box = Tuple[int, int, int, int]


class HaarFaceDetector:
    """Same detector and parameters the original script used."""

    def __init__(self):
        self._c = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def __call__(self, frame: np.ndarray) -> List[Box]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return [tuple(int(v) for v in f) for f in self._c.detectMultiScale(gray, 1.3, 5)]


@dataclass
class TrackView:
    track_id: int
    box: Box
    status: str
    confidence: float
    reason: str


@dataclass
class FrameResult:
    faces: List[Box]
    views: List[TrackView]
    people_count: int
    multi_active: bool
    mask_active: bool
    ring: Optional[str] = None   # "normal" / "mask" when a doorbell should ring this frame
    last_ring: Optional[Tuple[str, float]] = None  # (kind, timestamp) of the latest ring


class DoorbellPipeline:
    def __init__(self, cfg: Config, db: VisitorDatabase, sounds: SoundPlayer,
                 face_detector: Optional[Callable] = None,
                 mask_detector: Optional[MaskDetector] = None):
        self.cfg, self.db, self.sounds = cfg, db, sounds
        self.detect = face_detector or HaarFaceDetector()
        self.mask = mask_detector or MaskDetector()
        self.tracker = FaceTracker(cfg.min_hits, cfg.track_max_age)
        self.filter = MaskStatusFilter()
        self.multi = MultiPersonAlert(cfg.multi_threshold, cfg.multi_cooldown)
        self.mask_alert = MaskAlert(cfg.mask_cooldown, cfg.mask_clear_after)
        self._session = f"{int(time.time()):x}"
        self._last_mask_info = {}
        self._pending = set()            # arrivals waiting for a ring decision
        self._pending_deadline = 0.0
        self.last_ring = None

    def _vid(self, track_id: int) -> str:
        return f"{self._session}-{track_id}"

    def process(self, frame: np.ndarray, now: float) -> FrameResult:
        faces = self.detect(frame)
        upd = self.tracker.update(faces, now)
        count = len(upd.active)

        for t in upd.lost:
            self.filter.forget(t.id)
            self.db.log_event(EVENT_EXIT, self._vid(t.id), count,
                              detail=f"duration={t.last_seen - t.first_seen:.1f}s")
        for t in upd.new_confirmed:
            self.db.log_event(EVENT_ENTRY, self._vid(t.id), count, len(upd.new_confirmed))
            if not self._pending:
                self._pending_deadline = now + self.cfg.ring_decision_window
            self._pending.add(t.id)

        d = self.multi.update(count, now)
        if d.fired:
            self.db.log_event(EVENT_MULTI, "", count, count, f"{count} people in frame")
            if d.play_sound:
                self.sounds.play_multi()

        views, masked_ids = [], []
        for t in upd.active:
            if t.visible:
                res = self.mask.classify(frame, t.box)
                status = self.filter.observe(t.id, res.status)
                self._last_mask_info[t.id] = (res.confidence, res.reason)
            else:
                status = UNKNOWN   # not seen this frame: never assume masked
            conf, reason = self._last_mask_info.get(t.id, (0.0, ""))
            if status == MASKED and t.visible:
                masked_ids.append(t.id)
            views.append(TrackView(t.id, t.box, status, conf, reason if t.visible else "not visible"))

        ring = self._decide_ring(views, now)

        d = self.mask_alert.update(len(masked_ids), now)
        if d.fired:
            self.db.log_event(EVENT_MASK, self._vid(masked_ids[0]), count, len(masked_ids),
                              "face covering detected")
            # The mask doorbell already announced this arrival; don't stack the siren on it.
            if d.play_sound and ring != RING_MASK:
                self.sounds.play_security()

        if ring:
            self.last_ring = (ring, now)
        return FrameResult(faces, views, count, self.multi.active, self.mask_alert.active,
                           ring, self.last_ring)

    def _decide_ring(self, views, now: float) -> Optional[str]:
        """One ring per arrival group: mask bell if anyone is reliably masked,
        normal bell once everyone is resolved non-masked or the window ends."""
        if not self._pending:
            return None
        status = {v.track_id: v.status for v in views}
        self._pending &= set(status)              # visitors that left are dropped
        if not self._pending:
            return None
        states = [status[i] for i in self._pending]
        kind = None
        if MASKED in states:
            kind = RING_MASK
        elif all(s == VISIBLE for s in states) or now >= self._pending_deadline:
            kind = RING_NORMAL                    # unknown never counts as masked
        if kind:
            self._pending.clear()
        return kind
