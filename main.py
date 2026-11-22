import logging
import time

import cv2
import pygame

from doorbell.alerts import SoundPlayer
from doorbell.config import Config
from doorbell.dashboard import render
from doorbell.database import VisitorDatabase
from doorbell.pipeline import DoorbellPipeline
from doorbell.ringer import Ringer



def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = Config()

    pygame.mixer.init(frequency=44100, size=-16, channels=2)
    ringer = Ringer.create(cfg.doorbell_sound, cfg.mask_doorbell_sound)

    db = VisitorDatabase(cfg.csv_path)
    pipeline = DoorbellPipeline(cfg, db, SoundPlayer.create())

    cap = cv2.VideoCapture(cfg.camera_index)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = pipeline.process(frame, time.time())

            # One doorbell per new arrival: normal bell, or the mask bell if masked.
            if result.ring:
                print("Someone is at the door..." if result.ring == "normal"
                      else "Someone wearing a mask is at the door...")
                ringer.ring(result.ring)

            cv2.imshow("Face detected automated Doorbell", render(frame, result, db))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        db.close()


if __name__ == "__main__":
    main()
