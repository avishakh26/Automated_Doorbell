"""Downloads the two optional DNN models into models/ (run once).

* MobileNet-SSD (person detector, ~23 MB)  - github.com/chuanqi305/MobileNet-SSD
* YuNet (face + landmarks, ~0.2 MB)        - github.com/opencv/opencv_zoo
Without them the app falls back to the original Haar face detector.
"""
import hashlib
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parent.parent / "models"
FILES = {
    "MobileNetSSD_deploy.prototxt":
        ("https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt", None),
    "MobileNetSSD_deploy.caffemodel":
        ("https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel", None),
    "face_detection_yunet_2023mar.onnx":
        ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/"
         "face_detection_yunet_2023mar.onnx",
         "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
}

DEST.mkdir(exist_ok=True)
for name, (url, sha) in FILES.items():
    target = DEST / name
    if target.exists():
        print("have", name)
        continue
    print("downloading", name)
    data = urllib.request.urlopen(url, timeout=60).read()
    if sha and hashlib.sha256(data).hexdigest() != sha:
        raise SystemExit(f"checksum mismatch for {name}")
    target.write_bytes(data)
print("done")
