"""Export a trusted trained state dictionary and verify ONNX output parity."""
import json
from pathlib import Path

import numpy as np
import torch

from model import load_model


def export(checkpoint=Path('models/plant-disease-model.pth'), output=Path('models/plant-disease-model.onnx')):
    import onnx
    import onnxruntime as ort
    model = load_model(checkpoint)
    sample = torch.rand(1, 3, 256, 256, generator=torch.Generator().manual_seed(7))
    torch.onnx.export(model, sample, str(output), input_names=['leaf'], output_names=['logits'],
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(str(output))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(output), sess_options=options, providers=['CPUExecutionProvider'])
    with torch.inference_mode():
        expected = model(sample).numpy()
    actual = session.run(None, {'leaf': sample.numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
    print(json.dumps({'model': str(output), 'bytes': output.stat().st_size,
                      'max_absolute_logit_error': float(np.abs(actual - expected).max())}))


if __name__ == '__main__':
    export()
