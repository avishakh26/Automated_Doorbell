"""Person and face detectors (OpenCV DNN) used by the pipeline.

* PersonDetector: MobileNet-SSD (COCO/VOC "person"). Finds a person even when the
  face is covered, which a face detector alone cannot do.
* YuNetFaceDetector: YuNet face detector with 5 landmarks (eyes, nose, mouth
  corners) -- works in dim light where the Haar eye cascade finds nothing.
Both are optional: if the model files are missing the pipeline falls back to the
original Haar face detector.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)
Box = Tuple[int, int, int, int]
VOC_PERSON = 15


@dataclass
class Face:
    box: Box
    score: float
    landmarks: np.ndarray   # (5, 2): right eye, left eye, nose tip, right mouth, left mouth


class PersonDetector:
    def __init__(self, prototxt: Path, weights: Path, min_conf: float = 0.5):
        self.net = cv2.dnn.readNetFromCaffe(str(prototxt), str(weights))
        self.min_conf = min_conf

    def __call__(self, frame: np.ndarray) -> List[Box]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        out = self.net.forward()
        boxes, confs = [], []
        for d in out[0, 0]:
            if int(d[1]) == VOC_PERSON and d[2] >= self.min_conf:
                x1, y1, x2, y2 = (d[3:7] * np.array([w, h, w, h])).astype(int)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                if x2 - x1 > 20 and y2 - y1 > 20:
                    boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                    confs.append(float(d[2]))
        if len(boxes) > 1:   # drop duplicate overlapping boxes of the same person
            keep = cv2.dnn.NMSBoxes(boxes, confs, self.min_conf, 0.4)
            boxes = [boxes[int(i)] for i in np.array(keep).flatten()]
        return [tuple(b) for b in boxes]


class YuNetFaceDetector:
    def __init__(self, model: Path, min_score: float = 0.5):
        self.model, self.min_score = str(model), min_score
        self._det = None
        self._size = None

    def __call__(self, frame: np.ndarray) -> List[Face]:
        h, w = frame.shape[:2]
        if self._det is None or self._size != (w, h):
            self._det = cv2.FaceDetectorYN.create(self.model, "", (w, h), self.min_score, 0.3, 50)
            self._size = (w, h)
        _, faces = self._det.detect(frame)
        out = []
        for r in (faces if faces is not None else []):
            x, y, bw, bh = (int(v) for v in r[:4])
            out.append(Face((x, y, bw, bh), float(r[14]), np.array(r[4:14], float).reshape(5, 2)))
        return out


def load_detectors(cfg) -> Tuple[Optional[PersonDetector], Optional[YuNetFaceDetector]]:
    person = face = None
    try:
        person = PersonDetector(cfg.person_proto, cfg.person_weights, cfg.person_min_conf)
    except Exception as exc:
        log.warning("Person detector unavailable (%s)", exc)
    try:
        face = YuNetFaceDetector(cfg.yunet_model, cfg.face_min_score)
    except Exception as exc:
        log.warning("YuNet face detector unavailable (%s)", exc)
    return person, face
