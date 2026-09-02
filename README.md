# facial-recognition-flask

A small Flask web app that identifies faces in an uploaded photo by matching them
against a folder of known faces.

Detection uses OpenCV's **YuNet** model and recognition uses **SFace** — both run
on the CPU through `opencv-python-headless`, so there is no dlib compile step, no
TensorFlow, and no GPU requirement.

## How it works

1. On startup, every image in `static/known/` is detected, aligned and encoded
   into a 128-d embedding. These are cached in memory as a single normalised matrix.
2. An upload is decoded, and every face in it is encoded the same way.
3. Each face embedding is compared against the known matrix with a cosine
   similarity; the best score above `MATCH_THRESHOLD` (default `0.363`, SFace's
   recommended value) is reported as a match.

Because enrolment happens once at boot, a request only costs one decode plus one
encode per detected face — roughly **10 ms** for a typical photo.

## Running locally

**Requires Python 3.12 or newer** (NumPy 2.x sets the floor). Create the
virtualenv with an explicit interpreter — a bare `python` may point at an older
system or Anaconda install:

Any Python 3.12+ interpreter works; `python3.13` is used here as an example.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -V
pip install -r requirements.txt
python scripts/fetch_models.py
python app.py
```

`python -V` should report 3.12 or newer. `fetch_models.py` downloads ~39 MB of
ONNX models on first run. The app then serves on <http://localhost:5001>.

> **Careful pasting multi-line blocks into zsh:** interactive zsh does not treat
> `#` as a comment unless `setopt interactive_comments` is set, so a trailing
> `# note` becomes an argument. That is why the commands above carry no inline
> comments — `python3.13 -m venv .venv  # or any python3.12+` would create
> virtualenvs called `#`, `or`, `any` and `python3.12+`.

If you see `ModuleNotFoundError: No module named 'flask'` despite an active
`.venv`, the venv was probably built by a different Python than the one that
installed the packages. Check that `cat .venv/pyvenv.cfg` reports 3.12+, and if
not, rebuild it: `rm -rf .venv` and repeat the steps above.

> **macOS note:** port 5000 is used by AirPlay Receiver, which answers other
> requests with `403 Forbidden`. That is why the dev server defaults to 5001.
> If you ever do use 5000, reach it at `http://127.0.0.1:5000` rather than
> `http://localhost:5000` — AirPlay also binds IPv6, and `localhost` resolves
> to `::1` first.

For a production-style run:

```bash
gunicorn app:app --bind 127.0.0.1:5001 --workers 1 --threads 4
```

## Adding a known face

Drop a clear, front-facing photo into `static/known/`. The filename (minus its
extension) becomes the displayed name:

```
static/known/Ada Lovelace.jpg   ->   "Ada Lovelace"
```

`.jpg`, `.jpeg` and `.png` are all accepted. Restart the app to re-enrol. Files
with no detectable face are skipped with a warning rather than crashing.

## Deploying to Render

The repo includes [`render.yaml`](render.yaml), so you can create the service
directly from the dashboard via **New → Blueprint** and point it at this repo.
Render will:

- install `requirements.txt`
- run `scripts/fetch_models.py` during the build, so the 39 MB of models stay out of git
- start gunicorn and health-check `/healthz`

To wire it up manually instead:

| Setting | Value |
|---|---|
| Runtime | Python 3 |
| Build command | `pip install -r requirements.txt && python scripts/fetch_models.py` |
| Start command | `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 60` |
| Health check path | `/healthz` |

The whole stack fits comfortably in Render's free 512 MB instance.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MATCH_THRESHOLD` | `0.363` | Cosine similarity required to call it a match. Raise it to be stricter. |
| `KNOWN_DIR` | `static/known` | Folder of known faces. |
| `MODEL_DIR` | `models` | Where the ONNX models are stored. |
| `PORT` | `5001` | Port for the dev server. |

## Endpoints

| Route | Method | Description |
|---|---|---|
| `/` | GET | Upload form |
| `/recognize` | POST | Accepts `file`; returns the result page, or an HTML fragment for htmx requests |
| `/healthz` | GET | JSON status and enrolled-face count |

## Notes

The front end is server-rendered Jinja with [htmx](https://htmx.org/) for the
async upload, so results swap in without a page reload — and it still works with
JavaScript disabled, falling back to a normal form POST and a full result page.

Uploads are capped at 8 MB (`MAX_CONTENT_LENGTH`) and restricted to JPG, PNG and WebP.
