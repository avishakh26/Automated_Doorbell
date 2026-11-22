"""Central configuration for the doorbell application."""
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    data_dir: Path = BASE_DIR / "data"
    doorbell_sound: Path = BASE_DIR / "doorbell.wav"
    camera_index: int = 0

    # Original doorbell behaviour (any face -> ring, with cooldown).
    ring_cooldown: float = 5.0

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
    def csv_path(self) -> Path:
        return self.data_dir / "visitor_log.csv"
