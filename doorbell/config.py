"""Central configuration for the doorbell application."""
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    data_dir: Path = BASE_DIR / "data"
    doorbell_sound: Path = BASE_DIR / "doorbell.wav"
    mask_doorbell_sound: Path = BASE_DIR / "doorbell_mask.wav"
    camera_index: int = 0

    # Optional DNN models (see models/). Missing -> fall back to Haar face detection.
    models_dir: Path = BASE_DIR / "models"
    person_min_conf: float = 0.5
    face_min_score: float = 0.25   # low on purpose: covered faces score 0.3-0.7

    # Doorbell: rings once per new arrival. After a visitor is confirmed we wait
    # up to this long for the mask check to settle; still unsure -> normal bell.
    ring_decision_window: float = 1.5
    ring_max_wait: float = 4.0   # extra wait while covering evidence is building up

    # Tracking / de-duplication.
    min_hits: int = 3            # frames a face must persist before it counts as an entry
    track_max_age: float = 2.0   # seconds a track survives without a detection

    # Multiple-person alert.
    multi_threshold: int = 2
    multi_cooldown: float = 10.0

    # Mask / face-covering alert.
    mask_cooldown: float = 8.0
    mask_clear_after: float = 1.5

    @property
    def person_proto(self) -> Path:
        return self.models_dir / "MobileNetSSD_deploy.prototxt"

    @property
    def person_weights(self) -> Path:
        return self.models_dir / "MobileNetSSD_deploy.caffemodel"

    @property
    def yunet_model(self) -> Path:
        return self.models_dir / "face_detection_yunet_2023mar.onnx"

    @property
    def csv_path(self) -> Path:
        return self.data_dir / "visitor_log.csv"
