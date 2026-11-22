"""Lightweight centroid face tracker used to de-duplicate entries across frames."""
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

Box = Tuple[int, int, int, int]


@dataclass
class Track:
    id: int
    box: Box
    first_seen: float
    last_seen: float
    hits: int = 1
    confirmed: bool = False
    visible: bool = True   # detected in the most recent frame


@dataclass
class TrackUpdate:
    new_confirmed: List[Track] = field(default_factory=list)
    lost: List[Track] = field(default_factory=list)
    active: List[Track] = field(default_factory=list)   # confirmed and not yet expired


def _center(b: Box):
    return b[0] + b[2] / 2.0, b[1] + b[3] / 2.0


class FaceTracker:
    """A face becomes a *confirmed* track after ``min_hits`` consecutive
    detections (filters detector flicker) and survives ``max_age`` seconds of
    missed detections, so one person is one track however long they stay."""

    def __init__(self, min_hits: int = 3, max_age: float = 2.0, max_dist_ratio: float = 0.8):
        self.min_hits = min_hits
        self.max_age = max_age
        self.max_dist_ratio = max_dist_ratio
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1

    def update(self, boxes: Sequence[Box], now: float) -> TrackUpdate:
        boxes = [tuple(int(v) for v in b) for b in boxes]
        pairs = []
        for tid, t in self._tracks.items():
            tc = _center(t.box)
            for i, b in enumerate(boxes):
                bc = _center(b)
                d = ((tc[0] - bc[0]) ** 2 + (tc[1] - bc[1]) ** 2) ** 0.5
                if d <= self.max_dist_ratio * max(t.box[2], b[2]):
                    pairs.append((d, tid, i))
        pairs.sort()
        used_t, used_b = set(), set()
        for _, tid, i in pairs:
            if tid in used_t or i in used_b:
                continue
            used_t.add(tid)
            used_b.add(i)
            t = self._tracks[tid]
            t.box, t.last_seen, t.hits, t.visible = boxes[i], now, t.hits + 1, True

        upd = TrackUpdate()
        for tid in list(self._tracks):
            t = self._tracks[tid]
            if tid in used_t:
                continue
            t.visible = False
            expired = (now - t.last_seen) > self.max_age
            if not t.confirmed:
                del self._tracks[tid]      # unconfirmed flicker is dropped at once
            elif expired:
                del self._tracks[tid]
                upd.lost.append(t)
        for i, b in enumerate(boxes):
            if i not in used_b:
                t = Track(self._next_id, b, now, now)
                self._next_id += 1
                self._tracks[t.id] = t
        for t in self._tracks.values():
            if not t.confirmed and t.hits >= self.min_hits:
                t.confirmed = True
                upd.new_confirmed.append(t)
        upd.active = [t for t in self._tracks.values() if t.confirmed]
        return upd
