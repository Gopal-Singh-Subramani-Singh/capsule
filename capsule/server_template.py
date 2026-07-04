from __future__ import annotations
from capsule.models import CapsuleManifest, Framework

# ── Request limits ─────────────────────────────────────────────────────────────
# Max body size (bytes) accepted by the generated server
_MAX_BODY_BYTES = 10 * 1024 * 1024   # 10 MB
# Max batch size (number of rows in inputs array)
_MAX_BATCH = 1024

_SERVER_ONNX = '''
import os
import time
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import uvicorn
import signal
import sys

MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", "{max_body}"))
MAX_BATCH = int(os.environ.get("MAX_BATCH", "{max_batch}"))

app = FastAPI(title="Capsule Model Server")
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model/model.onnx")
PORT = int(os.environ.get("PORT", "{port}"))

REQUEST_COUNT = Counter("model_requests_total", "Total requests", ["status"])
REQUEST_LATENCY = Histogram(
    "model_request_duration_seconds", "Request latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0],
)

session = None
_healthy = False


def _shutdown(sig, frame):
    global _healthy
    _healthy = False
    sys.exit(0)

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)


@app.on_event("startup")
def load_model():
    global session, _healthy
    available = ort.get_available_providers()
    providers = [p for p in ["CoreMLExecutionProvider", "CPUExecutionProvider"] if p in available]
    session = ort.InferenceSession(MODEL_PATH, providers=providers)
    _healthy = True


@app.get("/health")
def health():
    if not _healthy:
        return JSONResponse(status_code=503, content={{"status": "not_ready"}})
    return {{"status": "ok", "model_loaded": session is not None}}


@app.post("/predict")
async def predict(request: Request):
    # ── size guard ────────────────────────────────────────────────────────
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_BODY_BYTES:
        REQUEST_COUNT.labels(status="error").inc()
        return JSONResponse(
            status_code=413,
            content={{"error": f"Request body exceeds {MAX_BODY_BYTES} bytes"}},
        )

    t0 = time.monotonic()
    try:
        body = await request.json()

        # ── input validation ──────────────────────────────────────────────
        if "inputs" not in body:
            return JSONResponse(status_code=422, content={{"error": "'inputs' key required"}})
        if not isinstance(body["inputs"], list):
            return JSONResponse(status_code=422, content={{"error": "'inputs' must be a list"}})
        if len(body["inputs"]) > MAX_BATCH:
            return JSONResponse(
                status_code=422,
                content={{"error": f"Batch size {len(body['inputs'])} exceeds limit {MAX_BATCH}"}},
            )

        inputs = np.array(body["inputs"], dtype=np.float32)
        input_name = session.get_inputs()[0].name
        output = session.run(None, {{input_name: inputs}})
        latency = time.monotonic() - t0
        REQUEST_COUNT.labels(status="ok").inc()
        REQUEST_LATENCY.observe(latency)
        return {{"output": output[0].tolist(), "latency_ms": round(latency * 1000, 2)}}

    except (ValueError, TypeError) as e:
        REQUEST_COUNT.labels(status="error").inc()
        return JSONResponse(status_code=422, content={{"error": f"Invalid input: {{e}}"}})
    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        return JSONResponse(status_code=500, content={{"error": "Internal server error"}})


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_config=None)
'''

_SERVER_PYTORCH = '''
import os
import signal
import sys
import time
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import uvicorn

MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", "{max_body}"))
MAX_BATCH = int(os.environ.get("MAX_BATCH", "{max_batch}"))

app = FastAPI(title="Capsule Model Server")
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model/model.pt")
PORT = int(os.environ.get("PORT", "{port}"))

REQUEST_COUNT = Counter("model_requests_total", "Total requests", ["status"])
REQUEST_LATENCY = Histogram(
    "model_request_duration_seconds", "Request latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0],
)

model = None
_healthy = False


def _shutdown(sig, frame):
    global _healthy
    _healthy = False
    sys.exit(0)

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)


@app.on_event("startup")
def load_model():
    global model, _healthy
    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()
    _healthy = True


@app.get("/health")
def health():
    if not _healthy:
        return JSONResponse(status_code=503, content={{"status": "not_ready"}})
    return {{"status": "ok", "model_loaded": model is not None}}


@app.post("/predict")
async def predict(request: Request):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_BODY_BYTES:
        REQUEST_COUNT.labels(status="error").inc()
        return JSONResponse(status_code=413, content={{"error": "Request too large"}})

    t0 = time.monotonic()
    try:
        body = await request.json()
        if "inputs" not in body or not isinstance(body["inputs"], list):
            return JSONResponse(status_code=422, content={{"error": "'inputs' list required"}})
        if len(body["inputs"]) > MAX_BATCH:
            return JSONResponse(status_code=422, content={{"error": "Batch too large"}})

        inputs = torch.tensor(body["inputs"], dtype=torch.float32)
        with torch.no_grad():
            output = model(inputs)
        latency = time.monotonic() - t0
        REQUEST_COUNT.labels(status="ok").inc()
        REQUEST_LATENCY.observe(latency)
        return {{"output": output.tolist(), "latency_ms": round(latency * 1000, 2)}}

    except (ValueError, TypeError) as e:
        REQUEST_COUNT.labels(status="error").inc()
        return JSONResponse(status_code=422, content={{"error": f"Invalid input: {{e}}"}})
    except Exception:
        REQUEST_COUNT.labels(status="error").inc()
        return JSONResponse(status_code=500, content={{"error": "Internal server error"}})


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_config=None)
'''

_SERVER_SKLEARN = '''
import os
import pickle
import signal
import sys
import time
import hashlib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import uvicorn

MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", "{max_body}"))
MAX_BATCH = int(os.environ.get("MAX_BATCH", "{max_batch}"))
MODEL_DIGEST = os.environ.get("MODEL_DIGEST", "")

app = FastAPI(title="Capsule Model Server")
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model/model.pkl")
PORT = int(os.environ.get("PORT", "{port}"))

REQUEST_COUNT = Counter("model_requests_total", "Total requests", ["status"])
REQUEST_LATENCY = Histogram(
    "model_request_duration_seconds", "Request latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0],
)

model = None
_healthy = False


def _shutdown(sig, frame):
    global _healthy
    _healthy = False
    sys.exit(0)

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)


@app.on_event("startup")
def load_model():
    global model, _healthy
    # Integrity check: verify file hash matches the digest baked in at package time
    if MODEL_DIGEST:
        with open(MODEL_PATH, "rb") as f:
            actual = "sha256:" + hashlib.sha256(f.read()).hexdigest()
        if actual != MODEL_DIGEST:
            raise RuntimeError(
                f"Model integrity check failed: expected {{MODEL_DIGEST}}, got {{actual}}"
            )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    _healthy = True


@app.get("/health")
def health():
    if not _healthy:
        return JSONResponse(status_code=503, content={{"status": "not_ready"}})
    return {{"status": "ok", "model_loaded": model is not None}}


@app.post("/predict")
async def predict(request: Request):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_BODY_BYTES:
        REQUEST_COUNT.labels(status="error").inc()
        return JSONResponse(status_code=413, content={{"error": "Request too large"}})

    t0 = time.monotonic()
    try:
        body = await request.json()
        if "inputs" not in body or not isinstance(body["inputs"], list):
            return JSONResponse(status_code=422, content={{"error": "'inputs' list required"}})
        if len(body["inputs"]) > MAX_BATCH:
            return JSONResponse(status_code=422, content={{"error": "Batch too large"}})

        inputs = np.array(body["inputs"])
        output = model.predict(inputs)
        latency = time.monotonic() - t0
        REQUEST_COUNT.labels(status="ok").inc()
        REQUEST_LATENCY.observe(latency)
        return {{"output": output.tolist(), "latency_ms": round(latency * 1000, 2)}}

    except (ValueError, TypeError) as e:
        REQUEST_COUNT.labels(status="error").inc()
        return JSONResponse(status_code=422, content={{"error": f"Invalid input: {{e}}"}})
    except Exception:
        REQUEST_COUNT.labels(status="error").inc()
        return JSONResponse(status_code=500, content={{"error": "Internal server error"}})


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_config=None)
'''


def generate_server_code(framework: Framework, manifest: CapsuleManifest) -> str:
    """Generate a hardened FastAPI server for the given framework."""
    def _render(tmpl: str) -> str:
        return (
            tmpl
            .replace("{port}", str(manifest.port))
            .replace("{max_body}", str(_MAX_BODY_BYTES))
            .replace("{max_batch}", str(_MAX_BATCH))
        )

    if framework == Framework.PYTORCH:
        return _render(_SERVER_PYTORCH).strip()
    elif framework == Framework.ONNX:
        return _render(_SERVER_ONNX).strip()
    elif framework == Framework.SKLEARN:
        return _render(_SERVER_SKLEARN).strip()
    else:
        return _render(_SERVER_ONNX).strip()
