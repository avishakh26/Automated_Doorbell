"""Replay a folder of recorded frames (<phase>_<n>.png) through the full pipeline
with the real detectors and print which doorbell would ring per phase.
Usage: python tools/replay_frames.py <folder> [seconds_between_frames]"""
import glob
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from doorbell.alerts import SoundPlayer  # noqa: E402
from doorbell.config import Config  # noqa: E402
from doorbell.database import VisitorDatabase  # noqa: E402
from doorbell.detectors import load_detectors  # noqa: E402
from doorbell.pipeline import DoorbellPipeline  # noqa: E402

folder = sys.argv[1]
dt = float(sys.argv[2]) if len(sys.argv) > 2 else 1 / 15
phases = defaultdict(list)
for f in sorted(glob.glob(os.path.join(folder, "*.png"))):
    phases[os.path.basename(f).rsplit("_", 1)[0]].append(f)
cfg = Config(data_dir=Path(tempfile.mkdtemp()))
person, face = load_detectors(cfg)
for ph, files in sorted(phases.items()):
    db = VisitorDatabase(Path(tempfile.mkdtemp()) / "l.csv")
    pipe = DoorbellPipeline(cfg, db, SoundPlayer(None, None), face_detector=face, person_detector=person)
    t, rings, cats, alerts = 0.0, [], {}, 0
    for f in files[4:]:                      # skip the first frames (transition)
        # each recorded frame is repeated to emulate the live frame rate
        reps = max(1, int(round(0.3 / dt)))
        im = cv2.imread(f)
        for _ in range(reps):
            t += dt
            r = pipe.process(im, t)
            if r.ring:
                rings.append(r.ring)
            for v in r.views:
                cats[v.category] = cats.get(v.category, 0) + 1
            alerts += r.mask_active
    s = db.stats()
    print(f"{ph:22s} rings={rings} entries={s['total_people']} mask_alert_frames={alerts} cats={cats}")
    db.close()
