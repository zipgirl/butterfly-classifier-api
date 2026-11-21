from __future__ import annotations

import io
import json
import os
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


def load_artifacts() -> None:
    global _model, _classes, _metrics, _load_error
    if _model is not None:
        return
    if not (MODEL_PATH.is_file() and CLASSES_PATH.is_file() and METRICS_PATH.is_file()):
        _load_error = "Trained model artifacts have not been installed."
        return
    try:
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
    # MobileNetV2 preprocessing: map RGB values from [0, 255] to [-1, 1].
    array = (array / 127.5) - 1.0
    probabilities = np.asarray(_model.predict(np.expand_dims(array, axis=0), verbose=0))[0]
    if probabilities.ndim != 1 or len(probabilities) != len(_classes):
        raise HTTPException(status_code=500, detail="Model returned an unexpected output shape")
    top_indices = np.argsort(probabilities)[::-1][:5]
    top = [{"class": _classes[int(i)], "confidence": float(probabilities[i])} for i in top_indices]
    best = top[0]
    return {"prediction": best["class"], "confidence": best["confidence"], "top_predictions": top}
