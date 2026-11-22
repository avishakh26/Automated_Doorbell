"""Detection pipeline: faces -> tracks -> visitor log, multi-person and mask alerts."""
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .alerts import MaskAlert, MultiPersonAlert, SoundPlayer
from .config import Config
from .database import EVENT_ENTRY, EVENT_EXIT, EVENT_MASK, EVENT_MULTI, VisitorDatabase
from .detectors import Face
from .mask import (CAT_PARTIAL, MASKED, UNKNOWN, VISIBLE, CoverAnalyzer, MaskDetector,
                   MaskResult, MaskStatusFilter)
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
    category: str = "unknown"


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
                 mask_detector: Optional[MaskDetector] = None,
                 person_detector: Optional[Callable] = None,
                 cover_analyzer: Optional[CoverAnalyzer] = None):
        """``face_detector`` returns ``Face`` objects (YuNet) or plain boxes (Haar).
        ``person_detector`` returns person boxes; it lets covered faces, which a face
        detector can miss, still be found as people."""
        self.cfg, self.db, self.sounds = cfg, db, sounds
        self.detect = face_detector or HaarFaceDetector()
        self.detect_persons = person_detector
        self.cover = cover_analyzer or CoverAnalyzer()
        self.mask = mask_detector or MaskDetector()
        self.tracker = FaceTracker(cfg.min_hits, cfg.track_max_age)
        self.filter = MaskStatusFilter()
        self.multi = MultiPersonAlert(cfg.multi_threshold, cfg.multi_cooldown)
        self.mask_alert = MaskAlert(cfg.mask_cooldown, cfg.mask_clear_after)
        self._session = f"{int(time.time()):x}"
        self._last_mask_info = {}
        self._pending = set()            # arrivals waiting for a ring decision
        self._pending_deadline = 0.0
        self._pending_start = 0.0
        self.last_ring = None

    @staticmethod
    def _plausible_entity(f: Face) -> bool:
        """Covered faces score low, but tiny weak blobs (a fingertip, a poster) must
        not become visitors: weak detections have to be reasonably large."""
        return (f.score >= 0.6 and f.box[2] >= 40) or f.box[2] >= 80

    def _vid(self, track_id: int) -> str:
        return f"{self._session}-{track_id}"

    def process(self, frame: np.ndarray, now: float) -> FrameResult:
        faces = [f if isinstance(f, Face) else Face(tuple(f), 1.0, None) for f in self.detect(frame)]
        persons = self.detect_persons(frame) if self.detect_persons else []
        # A visitor is a person box; a face outside every person box is a visitor too.
        entities = list(persons)
        for f in faces:
            cx, cy = f.box[0] + f.box[2] / 2, f.box[1] + f.box[3] / 2
            if not any(p[0] <= cx <= p[0] + p[2] and p[1] <= cy <= p[1] + p[3] for p in persons)                     and self._plausible_entity(f):
                entities.append(f.box)
        upd = self.tracker.update(entities, now)
        count = len(upd.active)

        for t in upd.lost:
            self.filter.forget(t.id)
            self.db.log_event(EVENT_EXIT, self._vid(t.id), count,
                              detail=f"duration={t.last_seen - t.first_seen:.1f}s")
        for t in upd.new_confirmed:
            self.db.log_event(EVENT_ENTRY, self._vid(t.id), count, len(upd.new_confirmed))
            if not self._pending:
                self._pending_deadline = now + self.cfg.ring_decision_window
                self._pending_start = now
            self._pending.add(t.id)

        d = self.multi.update(count, now)
        if d.fired:
            self.db.log_event(EVENT_MULTI, "", count, count, f"{count} people in frame")
            if d.play_sound:
                self.sounds.play_multi()

        views, masked_ids, suspects = [], [], set()
        for t in upd.active:
            face = None
            if t.visible:
                inside = [f for f in faces
                          if t.box[0] <= f.box[0] + f.box[2] / 2 <= t.box[0] + t.box[2]
                          and t.box[1] <= f.box[1] + f.box[3] / 2 <= t.box[1] + t.box[3]]
                face = max(inside, key=lambda f: f.score) if inside else None
                if face is None:
                    res = MaskResult(UNKNOWN, 0.0, "person seen, no face found", CAT_PARTIAL)
                elif face.landmarks is not None:
                    res = self.cover.analyze(frame, face)
                else:
                    res = self.mask.classify(frame, face.box)
                status = self.filter.observe(t.id, res.status, res.strong, now)
                if res.status == MASKED:
                    suspects.add(t.id)   # covering evidence is building up
                self._last_mask_info[t.id] = (res.confidence, res.reason, res.category)
            else:
                status = UNKNOWN   # not seen this frame: never assume masked
            conf, reason, cat = self._last_mask_info.get(t.id, (0.0, "", "unknown"))
            if status == UNKNOWN:
                cat = cat if cat == "partial" else "unknown"
            elif status == VISIBLE:
                cat = "normal"
            if status == MASKED and t.visible:
                masked_ids.append(t.id)
            views.append(TrackView(t.id, face.box if face else t.box, status, conf,
                                   reason if t.visible else "not visible", cat))

        ring = self._decide_ring(views, now, suspects)

        d = self.mask_alert.update(len(masked_ids), now)
        if d.fired:
            self.db.log_event(EVENT_MASK, self._vid(masked_ids[0]), count, len(masked_ids),
                              f"face covering detected: {self._last_mask_info[masked_ids[0]][2]}")
            # The mask doorbell already announced this arrival; don't stack the siren on it.
            if d.play_sound and ring != RING_MASK:
                self.sounds.play_security()

        if ring:
            self.last_ring = (ring, now)
        return FrameResult([f.box for f in faces], views, count, self.multi.active, self.mask_alert.active,
                           ring, self.last_ring)

    def _decide_ring(self, views, now: float, suspects=frozenset()) -> Optional[str]:
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
        elif all(s == VISIBLE for s in states):
            kind = RING_NORMAL
        elif now >= self._pending_deadline and not (self._pending & suspects
                                                    and now < self._pending_start + self.cfg.ring_max_wait):
            kind = RING_NORMAL                    # unknown never counts as masked
        if kind:
            self._pending.clear()
        return kind
