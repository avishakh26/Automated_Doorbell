"""Heuristic face-covering classifier built on OpenCV only (no extra model).

Approach: for a detected frontal face, (1) require the face to be large, fully
inside the frame, reasonably lit and to show eyes (so we know it is a real,
frontal, visible face); (2) measure the fraction of skin-coloured pixels in the
forehead band (calibration reference) and in the nose-to-chin band. A face
whose lower band has almost no skin compared with the forehead is "masked".
Anything that cannot be judged reliably returns "unknown" -- never "masked".
"""
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

MASKED, VISIBLE, UNKNOWN = "masked", "visible", "unknown"   # MASKED == any face covering

# Finer-grained category, used for display and logging only. Both COVER_* categories
# raise the same single special alert.
CAT_NORMAL = "normal"
CAT_MASK = "mask/cloth/object"
CAT_HAND = "hand/skin-tone cover"
CAT_PARTIAL = "partial"
CAT_UNKNOWN = "unknown"


@dataclass
class MaskResult:
    status: str
    confidence: float
    reason: str = ""
    category: str = CAT_UNKNOWN
    strong: bool = True   # strong = unambiguous colour evidence; weak = needs longer persistence


def skin_ratio(bgr: np.ndarray) -> float:
    if bgr.size == 0:
        return 0.0
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycc[..., 1], ycc[..., 2]
    skin = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    return float(skin.mean())


def mouth_features_present(band: np.ndarray, ref_cr: float, ref_gray: float) -> bool:
    """True if the mouth region shows lips (redder than the reference skin) or
    dark mouth/shadow pixels. A flat skin-toned hand or object shows neither."""
    if band.size == 0:
        return True
    ycc = cv2.cvtColor(band, cv2.COLOR_BGR2YCrCb)
    lips = float((ycc[..., 1].astype(float) > ref_cr + 10).mean())
    dark = float((ycc[..., 0].astype(float) < ref_gray * 0.55).mean())
    return lips > 0.03 or dark > 0.02


class MaskDetector:
    def __init__(self, eye_finder: Optional[Callable[[np.ndarray], list]] = None,
                 min_face: int = 80, min_ref_skin: float = 0.45,
                 masked_below: float = 0.25, visible_above: float = 0.70):
        self.min_face = min_face
        self.min_ref_skin = min_ref_skin
        self.masked_below = masked_below
        self.visible_above = visible_above
        if eye_finder is None:
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
            eye_finder = lambda g: list(cascade.detectMultiScale(g, 1.1, 6, minSize=(14, 14)))  # noqa: E731
        self._eyes = eye_finder

    def classify(self, frame: np.ndarray, box: Tuple[int, int, int, int]) -> MaskResult:
        x, y, w, h = box
        fh, fw = frame.shape[:2]
        if w < self.min_face or h < self.min_face:
            return MaskResult(UNKNOWN, 0.0, "face too small", CAT_PARTIAL)
        if x <= 0 or y <= 0 or x + w >= fw or y + h >= fh:
            return MaskResult(UNKNOWN, 0.0, "face partly outside frame", CAT_PARTIAL)
        face = frame[y:y + h, x:x + w]
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        if mean < 40 or mean > 215:
            return MaskResult(UNKNOWN, 0.0, "poor lighting", CAT_UNKNOWN)
        if len(self._eyes(gray[: int(h * 0.55)])) < 2:
            return MaskResult(UNKNOWN, 0.0, "eyes not visible (turned away / occluded)", CAT_PARTIAL)
        fore = face[int(h * 0.08):int(h * 0.30), int(w * 0.2):int(w * 0.8)]
        ref = skin_ratio(fore)
        if ref < self.min_ref_skin:
            return MaskResult(UNKNOWN, 0.0, "cannot calibrate skin tone", CAT_UNKNOWN)
        low = skin_ratio(face[int(h * 0.60):int(h * 0.95), int(w * 0.2):int(w * 0.8)])
        r = low / ref
        if r <= self.masked_below:
            return MaskResult(MASKED, float(min(1.0, 1 - r / 0.5)),
                              f"lower/upper skin ratio {r:.2f}", CAT_MASK)
        if r >= self.visible_above:
            ycc = cv2.cvtColor(fore, cv2.COLOR_BGR2YCrCb)
            h_, w_ = face.shape[:2]
            band = face[int(h_ * 0.62):int(h_ * 0.88), int(w_ * 0.3):int(w_ * 0.7)]
            if not mouth_features_present(band, float(np.median(ycc[..., 1])),
                                          float(np.median(ycc[..., 0]))):
                return MaskResult(MASKED, 0.6, "skin-toned lower face with no mouth features",
                                  CAT_HAND)
            return MaskResult(VISIBLE, float(min(1.0, r)), f"lower/upper skin ratio {r:.2f}", CAT_NORMAL)
        return MaskResult(UNKNOWN, 0.0, f"ambiguous skin ratio {r:.2f}", CAT_PARTIAL)


class MaskStatusFilter:
    """Temporal smoothing per track.

    * strong covering evidence (non-skin cloth/mask): ``need_masked`` of the last
      ``window`` observations (about a third of a second of video).
    * weak covering evidence (occluded-face score, flat mouth area): the face has to
      look covered for ``weak_seconds`` continuously (>= ``weak_ratio`` of samples),
      so a head turn, blink of the detector or someone glancing down cannot ring it.
    ``now`` is the frame timestamp; without it a 15 fps clock is assumed."""

    def __init__(self, window: int = 6, need_masked: int = 5, need_visible: int = 3,
                 weak_seconds: float = 1.5, weak_ratio: float = 0.85):
        self.window, self.need_masked, self.need_visible = window, need_masked, need_visible
        self.weak_seconds, self.weak_ratio = weak_seconds, weak_ratio
        self._hist: Dict[int, Deque] = {}
        self._tick = 0.0

    def observe(self, track_id: int, status: str, strong: bool = True, now=None) -> str:
        if now is None:
            self._tick += 1 / 15
            now = self._tick
        h = self._hist.setdefault(track_id, deque())
        code = ("S" if strong else "W") if status == MASKED else "V" if status == VISIBLE else "U"
        h.append((now, code))
        while len(h) > 1 and now - h[0][0] > self.weak_seconds * 1.5:
            h.popleft()
        recent = [c for _, c in list(h)[-self.window:]]
        if recent.count("S") >= self.need_masked:
            return MASKED
        span = [c for t, c in h if now - t <= self.weak_seconds]
        lasted = h[0][0] <= now - self.weak_seconds * 0.8
        if lasted and span and sum(c in "SW" for c in span) / len(span) >= self.weak_ratio:
            return MASKED
        if recent.count("V") >= self.need_visible and sum(c in "SW" for c in recent) <= 1:
            return VISIBLE
        return UNKNOWN

    def forget(self, track_id: int) -> None:
        self._hist.pop(track_id, None)


def _crop(img, x0, y0, x1, y1):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1))
    return img[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else img[0:0, 0:0]


class CoverAnalyzer:
    """Decides whether a detected face is clear, covered, or uncertain, using
    the YuNet landmarks to locate regions (no eye cascade needed).

    Evidence, tuned on real webcam recordings (clear face, palm over face, hand
    over mouth, cloth over nose/mouth, object in front of face, head turned):

    * STRONG - lower face is not skin coloured (mask / cloth / scarf / object):
      ``lower/upper skin ratio <= masked_below``.
    * WEAK   - the face detector's score collapses (<``weak_score``; a clear face
      scores ~0.93, an occluded one ~0.3-0.7) on a large face, or the mouth area
      is as bright/flat as the cheeks (hand over mouth; a real mouth, lips or
      moustache is darker). Weak evidence has to persist (see MaskStatusFilter).
    Anything else that is not clearly a visible face is ``unknown``/``partial``.
    """

    def __init__(self, min_face: int = 60, min_ref_skin: float = 0.35,
                 masked_below: float = 0.25, weak_score: float = 0.75,
                 weak_min_face: int = 90, clear_score: float = 0.85,
                 mouth_bright: float = 0.92, mouth_score_max: float = 0.92):
        self.min_face, self.min_ref_skin = min_face, min_ref_skin
        self.masked_below, self.weak_score = masked_below, weak_score
        self.weak_min_face, self.clear_score = weak_min_face, clear_score
        self.mouth_bright, self.mouth_score_max = mouth_bright, mouth_score_max

    def regions(self, frame, face):
        x, y, w, h = face.box
        re, le, nose, rm, lm = face.landmarks
        eye_y = (re[1] + le[1]) / 2
        mouth_y = (rm[1] + lm[1]) / 2
        left, right = min(rm[0], lm[0]), max(rm[0], lm[0])
        fore = _crop(frame, x + 0.2 * w, y, x + 0.8 * w, eye_y - 0.08 * h)
        cheeks = _crop(frame, x + 0.05 * w, eye_y + 0.05 * h, x + 0.95 * w, nose[1] - 0.02 * h)
        lower = _crop(frame, left - 0.1 * w, nose[1] + 0.08 * h, right + 0.1 * w, y + h)
        mouth = _crop(frame, left, mouth_y - 0.08 * h, right, mouth_y + 0.08 * h)
        cheek_band = _crop(frame, x + 0.1 * w, eye_y + 0.1 * h, x + 0.9 * w, nose[1])
        return fore, cheeks, lower, mouth, cheek_band

    def analyze(self, frame, face) -> MaskResult:
        x, y, w, h = face.box
        if w < self.min_face or h < self.min_face:
            return MaskResult(UNKNOWN, 0.0, "face too small", CAT_PARTIAL)
        fh, fw = frame.shape[:2]
        tol_x, tol_y = 0.03 * w, 0.03 * h
        if x < -tol_x or y < -tol_y or x + w > fw + tol_x or y + h > fh + tol_y:
            return MaskResult(UNKNOWN, 0.0, "face partly outside frame", CAT_PARTIAL)
        gray = cv2.cvtColor(_crop(frame, x, y, x + w, y + h), cv2.COLOR_BGR2GRAY)
        if gray.size == 0 or not (35 <= gray.mean() <= 220):
            return MaskResult(UNKNOWN, 0.0, "poor lighting", CAT_UNKNOWN)
        fore, cheeks, lower, mouth, cheek_band = self.regions(frame, face)
        if lower.size == 0 or mouth.size == 0 or cheek_band.size == 0:
            return MaskResult(UNKNOWN, 0.0, "regions outside frame", CAT_PARTIAL)
        ref = max(skin_ratio(fore) if fore.size else 0.0, skin_ratio(cheeks) if cheeks.size else 0.0)
        if ref < self.min_ref_skin:
            return MaskResult(UNKNOWN, 0.0, "cannot calibrate skin tone", CAT_UNKNOWN)
        r = skin_ratio(lower) / ref
        if r <= self.masked_below:                                   # strong: non-skin cover
            return MaskResult(MASKED, float(min(1.0, 1 - r / 0.5)),
                              f"lower face not skin (ratio {r:.2f})", CAT_MASK, strong=True)
        mouth_ratio = float(cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY).mean()
                            / max(cv2.cvtColor(cheek_band, cv2.COLOR_BGR2GRAY).mean(), 1.0))
        if face.score < self.weak_score and w >= self.weak_min_face:  # weak: face-ness collapsed
            return MaskResult(MASKED, 0.6, f"face score {face.score:.2f} (occluded)", CAT_HAND, strong=False)
        if mouth_ratio >= self.mouth_bright and face.score < self.mouth_score_max:
            return MaskResult(MASKED, 0.6, f"mouth area flat/bright ({mouth_ratio:.2f})",
                              CAT_HAND, strong=False)
        if face.score >= self.clear_score:
            return MaskResult(VISIBLE, float(face.score), f"clear face (score {face.score:.2f})", CAT_NORMAL)
        return MaskResult(UNKNOWN, 0.0, f"uncertain face (score {face.score:.2f})", CAT_PARTIAL)
