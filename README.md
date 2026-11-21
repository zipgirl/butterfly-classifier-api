# Butterfly Classifier API

FastAPI inference service for the Zipgirl butterfly classifier. The service intentionally reports that the model is unavailable until the trained model artifacts are added.

## Endpoints

- `GET /health` — service and artifact readiness
- `GET /metrics` — validation metrics from `api/artifacts/metrics.json`
- `POST /predict` — upload an image as multipart field `file`; returns the predicted class, confidence, and top five classes

## Required artifacts

Place these files in `api/artifacts/` after training on the official Kaggle train/validation folders:

- `mobilenetv2.keras` — exported trained Keras model
- `classes.json` — ordered list of class names matching model output indices
- `metrics.json` — validation accuracy, macro precision, macro recall, macro F1, and class count

Do not commit fabricated metrics or predictions. If artifacts are absent, `/health` reports `model_ready: false`, `/metrics` and `/predict` return HTTP 503.

## Run locally

```bash
pip install -r requirements-api.txt
uvicorn api.app:app --host 0.0.0.0 --port 7860
```

The production container uses the same command. Configure CORS with `ALLOWED_ORIGINS` if the frontend origin changes.

## Response contracts

`POST /predict` returns `predicted_class`, `confidence` (0–1), `model_name`, and `top_predictions` entries with `class_name` and `confidence`.

`GET /metrics` expects `accuracy`, `precision`, `recall`, and `f1` as fractions from 0 to 1, plus optional `split`, `samples`, and `model_name` fields. These must come from evaluation on the official labeled validation split.
