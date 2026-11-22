from __future__ import annotations

import io
import json
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS / "mobilenetv2.keras"
CLASSES_PATH = ARTIFACTS / "classes.json"
METRICS_PATH = ARTIFACTS / "metrics.json"
RELEASE_BASE = "https://github.com/zipgirl/butterfly-classifier-api/releases/download/model-v1"
MODEL_URL = f"{RELEASE_BASE}/mobilenetv2.zip"
CLASSES_URL = f"{RELEASE_BASE}/classes.json"
METRICS_URL = f"{RELEASE_BASE}/metrics.json"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
IMAGE_SIZE = (128, 128)

app = FastAPI(title="Butterfly Classifier API", version="1.0.0")
origins = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "https://zipgirl.github.io").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

_model = None
_classes: list[str] | None = None
_metrics: dict[str, Any] | None = None
_load_error: str | None = None


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ButterflyClassifierAPI/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def install_model_archive(archive_bytes: bytes) -> None:
    """Turn the release ZIP into a root-level .keras archive, validating its contents."""
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        keras_files = [name for name in names if name.lower().endswith(".keras")]
        if len(keras_files) == 1:
            MODEL_PATH.write_bytes(archive.read(keras_files[0]))
            return

        required = {"config.json", "metadata.json", "model.weights.h5"}
        roots = set()
        for name in names:
            parts = Path(name).parts
            if parts:
                roots.add(parts[0])
        prefix = ""
        if not required.issubset(set(names)) and len(roots) == 1:
            candidate = next(iter(roots)) + "/"
            if required.issubset({name[len(candidate):] for name in names if name.startswith(candidate)}):
                prefix = candidate

        normalized = io.BytesIO()
        with zipfile.ZipFile(normalized, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for name in names:
                if prefix and name.startswith(prefix):
                    output.writestr(name[len(prefix):], archive.read(name))
                elif not prefix:
                    output.writestr(name, archive.read(name))
        with zipfile.ZipFile(io.BytesIO(normalized.getvalue())) as check:
            if not required.issubset(set(check.namelist())):
                raise ValueError("mobilenetv2.zip does not contain a valid Keras model archive")
        MODEL_PATH.write_bytes(normalized.getvalue())


def ensure_artifacts() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.is_file():
        install_model_archive(download(MODEL_URL))
    if not CLASSES_PATH.is_file():
        CLASSES_PATH.write_bytes(download(CLASSES_URL))
    if not METRICS_PATH.is_file():
        METRICS_PATH.write_bytes(download(METRICS_URL))


def load_artifacts() -> None:
    global _model, _classes, _metrics, _load_error
    if _model is not None:
        return
    try:
        ensure_artifacts()
        import tensorflow as tf

        with CLASSES_PATH.open("r", encoding="utf-8") as f:
            classes = json.load(f)
        with METRICS_PATH.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        if not isinstance(classes, list) or not classes or not all(isinstance(c, str) for c in classes):
            raise ValueError("classes.json must contain a non-empty JSON list of class names")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        output_count = int(model.output_shape[-1])
        if output_count != len(classes):
            raise ValueError(f"Model outputs {output_count} classes but classes.json has {len(classes)}")
        _model, _classes, _metrics, _load_error = model, classes, metrics, None
    except Exception as exc:  # surfaced via health, never replaced with fake output
        _load_error = f"Could not load model artifacts: {type(exc).__name__}: {exc}"


def require_model() -> None:
    load_artifacts()
    if _model is None or _classes is None:
        raise HTTPException(status_code=503, detail=_load_error or "Model is not ready")


@app.get("/health")
def health() -> dict[str, Any]:
    load_artifacts()
    return {"status": "ok", "model_ready": _model is not None, "detail": _load_error}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    require_model()
    return _metrics or {}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    require_model()
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, or WEBP image")
    raw = await file.read(MAX_IMAGE_BYTES + 1)
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller")
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB").resize(IMAGE_SIZE)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image")
    array = np.asarray(image, dtype=np.float32)
    # Preprocessing is embedded in the exported model (Rescaling to [-1, 1]).
    # Send RGB pixel values in [0, 255] to avoid applying MobileNetV2 scaling twice.
    probabilities = np.asarray(_model.predict(np.expand_dims(array, axis=0), verbose=0))[0]
    if probabilities.ndim != 1 or len(probabilities) != len(_classes):
        raise HTTPException(status_code=500, detail="Model returned an unexpected output shape")
    top_indices = np.argsort(probabilities)[::-1][:5]
    top = [{"class": _classes[int(i)], "confidence": float(probabilities[i])} for i in top_indices]
    best = top[0]
    return {
        "prediction": best["class"],
        "predicted_class": best["class"],
        "confidence": best["confidence"],
        "model_name": "MobileNetV2",
        "top_predictions": [
            {"class": item["class"], "class_name": item["class"], "confidence": item["confidence"]}
            for item in top
        ],
    }
