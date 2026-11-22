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


def skin_ratio(bgr: np.ndarray) -> float:
    if bgr.size == 0:
        return 0.0
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycc[..., 1], ycc[..., 2]
    skin = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    return float(skin.mean())


def mouth_features_present(face: np.ndarray, ref_cr: float, ref_gray: float) -> bool:
    """True if the mouth band shows lips (redder than the forehead) or dark
    mouth/shadow pixels. A flat skin-toned hand or object shows neither."""
    h, w = face.shape[:2]
    band = face[int(h * 0.62):int(h * 0.88), int(w * 0.3):int(w * 0.7)]
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
            if not mouth_features_present(face, float(np.median(ycc[..., 1])),
                                          float(np.median(ycc[..., 0]))):
                return MaskResult(MASKED, 0.6, "skin-toned lower face with no mouth features",
                                  CAT_HAND)
            return MaskResult(VISIBLE, float(min(1.0, r)), f"lower/upper skin ratio {r:.2f}", CAT_NORMAL)
        return MaskResult(UNKNOWN, 0.0, f"ambiguous skin ratio {r:.2f}", CAT_PARTIAL)


class MaskStatusFilter:
    """Temporal smoothing per track: a face is only reported masked after
    ``need_masked`` of the last ``window`` observations say so."""

    def __init__(self, window: int = 6, need_masked: int = 5, need_visible: int = 3):
        self.window, self.need_masked, self.need_visible = window, need_masked, need_visible
        self._hist: Dict[int, Deque[str]] = {}

    def observe(self, track_id: int, status: str) -> str:
        h = self._hist.setdefault(track_id, deque(maxlen=self.window))
        h.append(status)
        masked, visible = h.count(MASKED), h.count(VISIBLE)
        if masked >= self.need_masked:
            return MASKED
        if visible >= self.need_visible and masked <= 1:
            return VISIBLE
        return UNKNOWN

    def forget(self, track_id: int) -> None:
        self._hist.pop(track_id, None)
