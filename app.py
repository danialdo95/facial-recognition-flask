"""A small Flask service that recognises faces in uploaded images."""

import sys

if sys.version_info < (3, 12):
    raise SystemExit(
        f"This app needs Python 3.12+ (NumPy 2.x requires it), but this "
        f"interpreter is {sys.version.split()[0]} at {sys.executable}.\n"
        f"Rebuild the virtualenv with an explicit interpreter:\n"
        f"    rm -rf .venv && python3.13 -m venv .venv\n"
        f"    source .venv/bin/activate && pip install -r requirements.txt"
    )

import base64
import logging
import os

from flask import Flask, render_template, request

from recognizer import ModelsMissing, Recognizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

try:
    recognizer = Recognizer()
except ModelsMissing as exc:
    log.error("%s", exc)
    recognizer = None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def render_result(**kwargs):
    """Render the result panel, as a bare fragment for htmx requests."""
    template = "partials/result.html" if request.headers.get("HX-Request") else "result.html"
    return render_template(template, **kwargs)


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        known_names=recognizer.names if recognizer else [],
        known_files=recognizer.files if recognizer else {},
        ready=recognizer is not None,
    )


@app.route("/recognize", methods=["POST"])
def recognize():
    if recognizer is None:
        return render_result(error="The recognition models are not loaded on the server."), 503

    upload = request.files.get("file")
    if upload is None or upload.filename == "":
        return render_result(error="Please choose an image first."), 400

    if not allowed_file(upload.filename):
        return render_result(
            error="Unsupported file type. Use a JPG, PNG or WebP image."
        ), 400

    image_bytes = upload.read()
    if not image_bytes:
        return render_result(error="That file was empty."), 400

    faces, error = recognizer.identify(image_bytes)
    if error:
        return render_result(error=error), 400

    return render_result(
        faces=faces,
        source_image=base64.b64encode(image_bytes).decode("ascii"),
        source_mime=upload.mimetype or "image/jpeg",
    )


@app.errorhandler(413)
def upload_too_large(_):
    limit = MAX_UPLOAD_BYTES // (1024 * 1024)
    return render_result(error=f"That image is larger than the {limit} MB limit."), 413


@app.route("/healthz")
def healthz():
    if recognizer is None:
        return {"status": "degraded", "known_faces": 0}, 503
    return {"status": "ok", "known_faces": len(recognizer.names)}


if __name__ == "__main__":
    # Port 5000 is taken by AirPlay Receiver on macOS, which also binds IPv6 and
    # answers everything else with 403 -- so http://localhost:5000 hits AirPlay
    # rather than this app. Default to 5001, where the unused IPv6 port refuses
    # fast and the browser falls back to IPv4.
    app.run(host="localhost", port=int(os.environ.get("PORT", 5001)), debug=True)
