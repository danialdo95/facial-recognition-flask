"""Face recognition backed by OpenCV's bundled YuNet detector and SFace recognizer.

Known faces are enrolled once at startup into an in-memory matrix of L2-normalised
embeddings, so a request only has to encode the uploaded image and take a dot
product against that matrix.
"""

import logging
import os
import threading

import cv2
import numpy as np

log = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
KNOWN_DIR = os.environ.get("KNOWN_DIR", os.path.join("static", "known"))

DETECTOR_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
RECOGNIZER_PATH = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")

# SFace ships with a recommended cosine threshold of 0.363; anything at or above
# that is treated as the same person.
MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.363"))
DETECT_SCORE_THRESHOLD = 0.9
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class ModelsMissing(RuntimeError):
    """Raised when the ONNX model files have not been downloaded."""


class Recognizer:
    def __init__(self):
        for path in (DETECTOR_PATH, RECOGNIZER_PATH):
            if not os.path.exists(path):
                raise ModelsMissing(
                    f"Missing model file: {path}. Run `python scripts/fetch_models.py` first."
                )

        self._detector = cv2.FaceDetectorYN.create(
            DETECTOR_PATH, "", (320, 320), DETECT_SCORE_THRESHOLD, 0.3, 5000
        )
        self._recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_PATH, "")
        # The OpenCV detector carries mutable input-size state, so serialise access.
        self._lock = threading.Lock()

        self.names: list[str] = []
        # name -> the actual file on disk, so templates never guess the extension
        self.files: dict[str, str] = {}
        self._embeddings = np.empty((0, 128), dtype=np.float32)
        self._enrol_known_faces()

    # -- enrolment ---------------------------------------------------------

    def _enrol_known_faces(self):
        names, vectors, files = [], [], {}

        if not os.path.isdir(KNOWN_DIR):
            log.warning("Known-faces directory %s does not exist", KNOWN_DIR)
            return

        for filename in sorted(os.listdir(KNOWN_DIR)):
            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue

            path = os.path.join(KNOWN_DIR, filename)
            image = cv2.imread(path)
            if image is None:
                log.warning("Skipping %s: not a readable image", filename)
                continue

            faces = self._detect(image)
            if faces is None:
                log.warning("Skipping %s: no face detected", filename)
                continue

            name = os.path.splitext(filename)[0]
            names.append(name)
            files[name] = filename
            vectors.append(self._encode(image, faces[0]))

        self.names = names
        self.files = files
        if vectors:
            self._embeddings = np.vstack(vectors)
        log.info("Enrolled %d known face(s): %s", len(names), ", ".join(names))

    # -- inference ---------------------------------------------------------

    def _detect(self, image):
        """Return detected faces sorted largest-first, or None if there are none."""
        height, width = image.shape[:2]
        with self._lock:
            self._detector.setInputSize((width, height))
            _, faces = self._detector.detect(image)
        if faces is None or len(faces) == 0:
            return None
        return sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    def _encode(self, image, face):
        """Align a detected face and return its L2-normalised embedding."""
        with self._lock:
            aligned = self._recognizer.alignCrop(image, face)
            feature = self._recognizer.feature(aligned)
        norm = np.linalg.norm(feature)
        return feature / norm if norm else feature

    def identify(self, image_bytes):
        """Identify every face in an image.

        Returns (faces, error). `faces` is a list of dicts, largest face first.
        """
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            return [], "That file could not be read as an image."

        detected = self._detect(image)
        if detected is None:
            return [], None

        results = []
        for face in detected:
            embedding = self._encode(image, face)
            name, score = None, 0.0

            if len(self.names):
                similarities = self._embeddings @ embedding[0]
                best = int(np.argmax(similarities))
                score = float(similarities[best])
                if score >= MATCH_THRESHOLD:
                    name = self.names[best]

            x, y, w, h = (int(v) for v in face[:4])
            results.append(
                {
                    "name": name,
                    "file": self.files.get(name) if name else None,
                    "score": score,
                    "confidence": round(min(score / MATCH_THRESHOLD, 1.0) * 100),
                    "box": {"x": x, "y": y, "width": w, "height": h},
                }
            )

        return results, None
