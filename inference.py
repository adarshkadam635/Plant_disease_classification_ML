"""Lightweight ONNX inference for deployment, with local PyTorch compatibility."""
import numpy as np
from PIL import Image

from labels import CLASS_NAMES


def load_model(path):
    if path.suffix.lower() != '.onnx':
        from model import load_model as load_torch_model
        return load_torch_model(path)
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    output = session.run(None, {'leaf': np.zeros((1, 3, 256, 256), dtype=np.float32)})[0]
    if output.shape != (1, len(CLASS_NAMES)) or not np.isfinite(output).all():
        raise ValueError('Invalid ONNX model output')
    return session


def predict(model, image):
    if not hasattr(model, 'run'):
        from model import predict as predict_torch
        return predict_torch(model, image)
    image = image.resize((256, 256), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = np.ascontiguousarray(array.transpose(2, 0, 1)[None])
    logits = model.run(None, {'leaf': tensor})[0][0]
    if not np.isfinite(logits).all():
        raise RuntimeError('Invalid model output')
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    results = []
    for index in np.argsort(-probabilities, kind='stable')[:3]:
        label = CLASS_NAMES[index]
        plant, disease = label.split('___')
        results.append({'label': label, 'plant': plant.replace('_', ' '),
                        'disease': disease.replace('_', ' ').strip(),
                        'confidence': round(float(probabilities[index]) * 100, 2),
                        'healthy': disease == 'healthy'})
    return results
