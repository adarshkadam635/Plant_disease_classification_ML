import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app import MAX_UPLOAD, app
from model import CLASS_NAMES, ResNet9, load_model, predict


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv('MODEL_PATH', str(tmp_path / 'missing.pth'))
    with TestClient(app) as client:
        yield client


def photo(size=(256, 256), format='PNG'):
    stream = io.BytesIO()
    Image.new('RGB', size, (40, 120, 60)).save(stream, format=format)
    return stream.getvalue()


def test_dashboard_and_security(client):
    response = client.get('/')
    assert response.status_code == 200
    assert 'Take a photo' in response.text
    assert "frame-ancestors 'none'" in response.headers['content-security-policy']
    assert client.get('/static/app.js').status_code == 200
    assert client.get('/static/styles.css').status_code == 200
    assert client.get('/api/health').json()['model_ready'] is False
    assert len(client.get('/api/classes').json()['classes']) == 38
    assert len({c.split('___')[0] for c in CLASS_NAMES}) == 14


@pytest.mark.parametrize('body,kind,code', [
    (b'', 'image/png', 422), (b'not an image', 'image/jpeg', 422),
    (b'<svg/>', 'image/svg+xml', 415), (photo((8, 8)), 'image/png', 422),
    (photo(format='GIF'), 'image/png', 415),
    (photo(), 'image/png', 503),
])
def test_input_validation(client, body, kind, code):
    response = client.post('/api/predict', content=body, headers={'content-type': kind})
    assert response.status_code == code
    assert response.json()['detail']


def test_oversized_upload(client):
    response = client.post('/api/predict', content=b'x', headers={'content-type':'image/png','content-length':str(MAX_UPLOAD + 1)})
    assert response.status_code == 413


def test_chunked_upload_limit(client):
    def chunks():
        for _ in range(11):
            yield b'x' * 1024 * 1024
    assert client.post('/api/predict', content=chunks(), headers={'content-type':'image/png'}).status_code == 413


def test_inference_and_roundtrip(tmp_path):
    torch.set_num_threads(2)
    model = ResNet9().eval()
    path = tmp_path / 'weights.pth'
    torch.save(model.state_dict(), path)
    loaded = load_model(path)
    results = predict(loaded, Image.new('RGB', (640, 480)))
    assert len(results) == 3
    assert results[0]['confidence'] >= results[1]['confidence']
    assert all(0 <= r['confidence'] <= 100 for r in results)
    # Synthetic weights test compatibility only, never diagnostic accuracy.


def test_prediction_route_and_busy(client):
    class FixedModel:
        def __call__(self, tensor):
            logits = torch.zeros(1, 38)
            logits[0, CLASS_NAMES.index('Apple___healthy')] = 10
            return logits
    app.state.model = FixedModel()
    response = client.post('/api/predict', content=photo(), headers={'content-type':'image/png'})
    assert response.status_code == 200
    result = response.json()
    assert result['predictions'][0]['disease'] == 'healthy'
    assert result['predictions'][0]['confidence'] > 99
    app.state.inference_lock.acquire()
    try:
        assert client.post('/api/predict', content=photo(), headers={'content-type':'image/png'}).status_code == 429
    finally:
        app.state.inference_lock.release()


def test_corrupt_checkpoint(monkeypatch, tmp_path):
    path = tmp_path / 'bad.pth'
    path.write_bytes(b'broken checkpoint')
    monkeypatch.setenv('MODEL_PATH', str(path))
    with TestClient(app) as client:
        assert not client.get('/api/health').json()['model_ready']


def test_training_status_filters_private_fields(client, monkeypatch, tmp_path):
    import app as app_module
    import json
    monkeypatch.setattr(app_module, 'ROOT', tmp_path)
    assert client.get('/api/training').json() == {'state': 'not_started'}
    (tmp_path / 'models').mkdir()
    path = tmp_path / 'models/training-status.json'
    path.write_text(json.dumps({'state': 'training', 'epoch': 1, 'epochs': 2,
                               'message': 'private filesystem details'}))
    assert client.get('/api/training').json() == {'state': 'training', 'epoch': 1, 'epochs': 2}
    path.write_text('invalid JSON')
    assert client.get('/api/training').json() == {'state': 'unavailable'}
