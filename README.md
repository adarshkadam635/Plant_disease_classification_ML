# LeafLens — Plant Disease Classification

A responsive dashboard for classifying leaf images across **38 conditions and 14 plant varieties**. Upload a photo, drag it into the dashboard, or capture one with your camera. The included trained ResNet-9 model runs on your server through ONNX Runtime, without an external AI API.

## Features

- Image upload, drag and drop, live camera preview/capture, and mobile camera support.
- Preview and validation for JPEG, PNG, and WebP files up to 10 MB. Large photos are optimized before upload to fit Vercel's request limit.
- Top three predictions with confidence scores and downloadable JSON reports.
- Searchable plant/condition library and browser-local history for 50 scans.
- Images are not saved by the server or in scan history.
- Bounded input decoding, CPU inference, security headers, and API/inference tests.

## Run locally

Use **Python 3.12**. The trained ONNX checkpoint is included, so retraining is not required.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. Alternatively, run `.\start.ps1` after installing dependencies. On Linux/macOS, use `python3.12` and `.venv/bin/python`.

Camera access requires localhost or HTTPS plus browser permission. When live camera access is unavailable, use the device camera option or upload a photo. No Node build step is needed.

## Trained model

Dataset: [New Plant Diseases Dataset, version 2](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).

| Item | Value |
| --- | --- |
| Architecture | ResNet-9, matching the original notebook |
| Training images | 70,295 |
| Validation images | 17,572 |
| Epochs | 2 |
| Best validation accuracy | 98.87% |
| Input | RGB, 256 × 256, float32, pixel values divided by 255 |
| Training | Adam, OneCycle LR (maximum 0.01), gradient clipping 0.1, weight decay 1e-4 |

`models/plant-disease-model.onnx` is the deployment checkpoint. `models/plant-disease-model.pth` is its original PyTorch state dictionary. Label order is defined in `labels.py` and recorded in `models/classes.json`. The validation report includes per-epoch metrics and the confusion matrix. `models/training-summary.json` records completion.

The validation split contains augmented dataset images; its accuracy does not establish independent field-image accuracy. The classifier can assign labels to unsupported plants or non-leaf images. Confidence is a model score, not a confirmed diagnosis. Confirm findings with a plant specialist.

## Train or resume

```powershell
.\.venv\Scripts\python -m pip install -r requirements-train.txt
.\.venv\Scripts\python -u train.py --epochs 2 --batch-size 16 --threads 6 --workers 2 --resume
```

This downloads the dataset to `data/.cache` (several GB). Set `--data PATH` to use an existing extracted copy. The CPU run takes several hours. Checkpoints save every 250 batches and at epoch boundaries, including optimizer/scheduler state. Use the same command to resume; configuration changes are rejected. Progress appears in the dashboard and at `/api/training`.

To refresh the ONNX deployment checkpoint after retraining:

```powershell
.\.venv\Scripts\python export_model.py
```

The exporter checks the ONNX graph and verifies numerical parity against PyTorch. Keep the computer awake while training. Training never runs inside a Vercel function. To serve a different checkpoint locally, set `MODEL_PATH` to an absolute `.onnx` or trusted `.pth` file path. PyTorch is required only for training, export, or loading `.pth` files. Full pickled model objects are not supported.

## Deploy to Vercel

The project uses the FastAPI framework preset and Python 3.12. Runtime dependencies are in `requirements.txt`; PyTorch, dataset files, resumable checkpoints, and development files are excluded from the deployed bundle.

```powershell
vercel login
vercel --prod --yes --project plant-disease-classification
```

Import this repository into Vercel for automatic Git deployments. Use the project name **plant-disease-classification**, FastAPI framework, and repository root directory. No environment variables are required. The exported model is included with the source; there is no external model download at startup.

Vercel accepts request bodies up to 4.5 MB; the dashboard compresses uploads above 4 MB before submitting them. Direct API clients must respect the platform limit. `/api/health` must return `model_ready: true` after deployment.

For a container deployment:

```sh
docker build -t leaflens .
docker run --rm -p 8000:8000 -v /absolute/path/to/models:/app/models:ro leaflens
```

Docker runs as a non-root user; mount the models directory read-only. For remote standalone hosting, use an HTTPS reverse proxy and configure request limits, timeouts, and rate limiting. The app has no account system; restrict access at the hosting layer for private deployments.

## Checks

```powershell
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pytest -q
node --check static/app.js
.\.venv\Scripts\python -m pip check
```

Tests cover the deployment model, ONNX/PyTorch prediction parity, startup without importing PyTorch, security headers, input validation, concurrency rejection, checkpoint loading, training status, and Windows file-lock retries. Synthetic inputs test compatibility, not diagnostic accuracy.

## API

- `GET /api/health`: liveness and model availability.
- `GET /api/classes`: canonical class labels.
- `GET /api/training`: live progress or the completed training summary.
- `POST /api/predict`: raw JPEG, PNG, or WebP body with the corresponding `Content-Type`; returns the top three predictions. Multipart form data is not used.

Implementation references: [Vercel FastAPI](https://vercel.com/docs/frameworks/backend/fastapi), [ONNX Runtime](https://onnxruntime.ai/docs/api/python/api_summary.html), [PyTorch ONNX](https://docs.pytorch.org/docs/stable/onnx.html).
