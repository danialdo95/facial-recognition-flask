"""Download the ONNX models used by the recognizer.

Run once locally, and as part of the Render build step:
    python scripts/fetch_models.py
"""

import hashlib
import os
import sys
import urllib.request

BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODEL_DIR = os.environ.get("MODEL_DIR", "models")

MODELS = [
    ("face_detection_yunet_2023mar.onnx", f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx"),
    ("face_recognition_sface_2021dec.onnx", f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx"),
]


def download(name, url):
    target = os.path.join(MODEL_DIR, name)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        print(f"  {name} already present ({os.path.getsize(target) / 1e6:.1f} MB)")
        return

    print(f"  downloading {name} ...", flush=True)
    tmp = target + ".part"
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as fh:
        digest = hashlib.sha256()
        while chunk := response.read(1 << 16):
            fh.write(chunk)
            digest.update(chunk)
    os.replace(tmp, target)
    print(f"  {name} -> {os.path.getsize(target) / 1e6:.1f} MB (sha256 {digest.hexdigest()[:12]})")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Fetching models into {MODEL_DIR}/")
    try:
        for name, url in MODELS:
            download(name, url)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
