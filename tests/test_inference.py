from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from app import app
from inference import load_model, predict
from model import load_model as load_torch_model, predict as predict_torch

ROOT = Path(__file__).resolve().parents[1]


def test_deployment_import_does_not_require_pytorch():
    subprocess.run([sys.executable, '-c', "import app,sys; assert 'torch' not in sys.modules"],
                   cwd=ROOT, check=True)


def test_onnx_checkpoint_matches_pytorch():
    image = Image.fromarray(np.random.default_rng(7).integers(0, 256, (320, 400, 3), dtype=np.uint8))
    runtime = load_model(ROOT / 'models/plant-disease-model.onnx')
    torch_model = load_torch_model(ROOT / 'models/plant-disease-model.pth')
    actual, expected = predict(runtime, image), predict_torch(torch_model, image)
    assert [p['label'] for p in actual] == [p['label'] for p in expected]
    for a, e in zip(actual, expected):
        assert abs(a['confidence'] - e['confidence']) <= 0.02


def test_deployed_model_api(monkeypatch):
    import io
    monkeypatch.setenv('MODEL_PATH', str(ROOT / 'models/plant-disease-model.onnx'))
    image = io.BytesIO()
    Image.new('RGB', (256, 256), (40, 120, 60)).save(image, format='PNG')
    with TestClient(app) as client:
        assert client.get('/api/health').json()['model_ready'] is True
        result = client.post('/api/predict', content=image.getvalue(), headers={'content-type': 'image/png'})
        assert result.status_code == 200
        assert len(result.json()['predictions']) == 3
