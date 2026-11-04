"""LeafLens: same-origin dashboard and bounded, CPU-only ResNet inference."""
import io
import asyncio
import json
import logging
import os
import threading
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from inference import CLASS_NAMES, load_model, predict

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    app.state.model = None
    app.state.inference_lock = threading.Lock()
    default_checkpoint = ROOT / 'models/plant-disease-model.onnx'
    if not default_checkpoint.is_file():
        default_checkpoint = ROOT / 'models/plant-disease-model.pth'
    checkpoint = Path(os.getenv('MODEL_PATH', str(default_checkpoint)))
    if checkpoint.is_file():
        try:
            app.state.model = load_model(checkpoint)
        except Exception:
            logger.exception('Model could not be loaded; predictions are disabled')
    async def watch_training():
        """Activate published validation checkpoints without interrupting the dashboard."""
        loaded_stamp = checkpoint.stat().st_mtime_ns if app.state.model is not None else None
        while True:
            await asyncio.sleep(15)
            try:
                if not checkpoint.is_file():
                    continue
                stamp = checkpoint.stat().st_mtime_ns
                if stamp == loaded_stamp:
                    continue
                model = await run_in_threadpool(load_model, checkpoint)
                app.state.model = model
                loaded_stamp = stamp
                logger.info('Activated published model checkpoint')
            except Exception:
                logger.exception('Could not activate training checkpoint')
    watcher = asyncio.create_task(watch_training())
    try:
        yield
    finally:
        watcher.cancel()
        try:
            await watcher
        except asyncio.CancelledError:
            pass


app = FastAPI(title='LeafLens', lifespan=lifespan, docs_url=None, redoc_url=None)


@app.middleware('http')
async def guard_requests(request: Request, call_next):
    if request.method == 'POST':
        try:
            length = int(request.headers.get('content-length', '0'))
        except ValueError:
            return JSONResponse({'detail': 'Invalid content length.'}, status_code=400)
        if length > MAX_UPLOAD:
            return JSONResponse({'detail': 'Image must be smaller than 10 MB.'}, status_code=413)
    response = await call_next(request)
    response.headers.update({
        'X-Content-Type-Options': 'nosniff',
        'Referrer-Policy': 'same-origin',
        'Permissions-Policy': 'camera=(self), microphone=()',
        'Content-Security-Policy': "default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
        'Cache-Control': 'no-store' if request.url.path.startswith('/api') else 'no-cache',
    })
    return response


@app.get('/api/health')
def health():
    ready = app.state.model is not None
    return {'status': 'ok', 'model_ready': ready, 'classes': len(CLASS_NAMES),
            'message': 'Model ready' if ready else 'Trained model required. Add models/plant-disease-model.pth and restart.'}


@app.get('/api/classes')
def classes():
    return {'classes': CLASS_NAMES}


@app.get('/api/training')
def training_status():
    path = ROOT / 'models/training-status.json'
    if not path.is_file():
        path = ROOT / 'models/training-summary.json'
    if not path.is_file():
        return {'state': 'not_started'}
    try:
        status = json.loads(path.read_text(encoding='utf-8'))
        # Return progress fields only; local paths and exception details stay in logs.
        allowed = {'state', 'epoch', 'epochs', 'train_images', 'validation_images',
                   'completed_batches', 'batches_per_epoch', 'train_loss', 'train_accuracy',
                   'best_validation_accuracy', 'estimated_training_seconds_remaining',
                   'validated_images', 'updated_at', 'finished_at', 'metrics', 'device'}
        return {key: value for key, value in status.items() if key in allowed}
    except (OSError, ValueError):
        return {'state': 'unavailable'}


def analyze(data):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in {'JPEG', 'PNG', 'WEBP'}:
                    raise HTTPException(415, 'Use a JPEG, PNG, or WebP image.')
                source.load()
                if min(source.size) < 32:
                    raise HTTPException(422, 'Image is too small. Use an image at least 32 pixels on each side.')
                image = ImageOps.exif_transpose(source).convert('RGB')
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(422, 'This image is damaged or too large to decode. Try a different photo.')
    if app.state.model is None:
        raise HTTPException(503, 'The trained model is not installed. Add models/plant-disease-model.pth and restart the server.')
    if not app.state.inference_lock.acquire(blocking=False):
        raise HTTPException(429, 'Another image is being analyzed. Please try again shortly.')
    try:
        start = time.perf_counter()
        results = predict(app.state.model, image)
        return {'predictions': results, 'elapsed_ms': round((time.perf_counter() - start) * 1000),
                'model': 'ResNet-9', 'notice': 'Model confidence is not a confirmed diagnosis. Confirm findings with a local plant specialist.'}
    finally:
        app.state.inference_lock.release()


@app.post('/api/predict')
async def prediction(request: Request):
    # Raw bytes avoid unbounded multipart parsing and temporary upload files.
    if request.headers.get('content-type', '').split(';')[0] not in {'image/jpeg', 'image/png', 'image/webp'}:
        raise HTTPException(415, 'Use a JPEG, PNG, or WebP image.')
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, 'Image must be smaller than 10 MB.')
    if not data:
        raise HTTPException(422, 'Choose an image first.')
    return await run_in_threadpool(analyze, bytes(data))


@app.get('/')
def dashboard():
    return FileResponse(ROOT / 'static/index.html')


app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app:app', host='127.0.0.1', port=int(os.getenv('PORT', '8000')))
