import logging
import time

import cv2
import pygame

from doorbell.alerts import SoundPlayer
from doorbell.config import Config
from doorbell.dashboard import render
from doorbell.database import VisitorDatabase
from doorbell.pipeline import DoorbellPipeline


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = Config()

    pygame.mixer.init(frequency=44100, size=-16, channels=2)
    pygame.mixer.music.load(str(cfg.doorbell_sound))

    db = VisitorDatabase(cfg.csv_path)
    pipeline = DoorbellPipeline(cfg, db, SoundPlayer.create())

    cap = cv2.VideoCapture(cfg.camera_index)
    last_ring_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            result = pipeline.process(frame, now)

            # Original behaviour: any face rings the doorbell, with a cooldown.
            if len(result.faces) > 0 and now - last_ring_time > cfg.ring_cooldown:
                print("Someone is at the door...")
                pygame.mixer.music.play()
                last_ring_time = now

            cv2.imshow("Face detected automated Doorbell", render(frame, result, db))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        db.close()


if __name__ == "__main__":
    main()
