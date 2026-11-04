"""Architecture and class order matching the notebook's ImageFolder training."""
import numpy as np
import torch
from torch import nn
from PIL import Image

from labels import CLASS_NAMES


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, in_channels=3, num_diseases=38):
        super().__init__()
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, num_diseases))

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = self.res1(x) + x
        x = self.conv4(self.conv3(x))
        return self.classifier(self.res2(x) + x)


def load_model(path):
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model = ResNet9()
    # Only load state dictionaries; never deserialize arbitrary model objects.
    weights = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 256, 256))
        if not torch.isfinite(output).all():
            raise ValueError('Checkpoint produces non-finite output')
    return model


def predict(model, image):
    image = image.resize((256, 256), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0]
        if not torch.isfinite(probabilities).all():
            raise RuntimeError('Invalid model output')
        values, indices = probabilities.topk(3)
    results = []
    for value, index in zip(values.tolist(), indices.tolist()):
        label = CLASS_NAMES[index]
        plant, disease = label.split('___')
        results.append({'label': label, 'plant': plant.replace('_', ' '),
                        'disease': disease.replace('_', ' ').strip(),
                        'confidence': round(value * 100, 2), 'healthy': disease == 'healthy'})
    return results
