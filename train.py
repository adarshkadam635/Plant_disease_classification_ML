"""Train the notebook's ResNet-9 on the full Kaggle train/valid split.

Checkpoints include optimizer/scheduler state and the next batch position so
interrupted runs can resume without silently restarting training.
"""
import argparse
import json
import math
import os
import random
import time
import traceback
import io
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from filelock import FileLock, Timeout
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from model import CLASS_NAMES, ResNet9, load_model, predict

ROOT = Path(__file__).resolve().parent
DATASET = 'vipoooool/new-plant-diseases-dataset/versions/2'


def replace_with_retry(source, destination):
    """Windows readers/scanners can briefly lock files against replacement."""
    for attempt in range(10):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(0.05 * 2 ** attempt, 0.5))


def atomic_json(path, value):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    replace_with_retry(temp, path)


def atomic_save(value, path):
    temp = path.with_suffix('.pth.tmp')
    torch.save(value, temp)
    replace_with_retry(temp, path)


class LeafDataset(Dataset):
    def __init__(self, path):
        path = Path(path)
        classes = sorted(p.name for p in path.iterdir() if p.is_dir())
        if classes != CLASS_NAMES:
            raise ValueError(f'Training label order does not match the app: {classes}')
        self.samples = []
        for label, name in enumerate(classes):
            files = sorted(p for p in (path / name).iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
            if not files:
                raise ValueError(f'No images found for {name}')
            self.samples.extend((str(p), label) for p in files)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array.transpose(2, 0, 1).copy()), label


def find_split(root):
    candidates = [p for p in Path(root).rglob('train') if p.is_dir() and (p.parent / 'valid').is_dir()]
    if not candidates:
        raise FileNotFoundError('No train/valid folders found in downloaded dataset')
    return sorted(candidates, key=lambda p: len(p.parts))[0].parent


def train(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'training-status.json'
    status = {'state': 'preparing', 'dataset': DATASET, 'epochs': args.epochs,
              'device': 'cpu', 'started_at': datetime.now(timezone.utc).isoformat()}
    def report(**updates):
        status.update(updates, updated_at=datetime.now(timezone.utc).isoformat())
        try:
            atomic_json(status_path, status)
        except OSError as exc:
            # Dashboard telemetry must never abort a multi-hour training run.
            print(f'Warning: could not update progress file: {exc}', flush=True)
        print(json.dumps(status), flush=True)
    report()
    try:
        torch.set_num_threads(args.threads)
        torch.manual_seed(7)
        random.seed(7)
        np.random.seed(7)
        if args.data:
            root = Path(args.data)
        else:
            os.environ.setdefault('KAGGLEHUB_CACHE', str(ROOT / 'data/.cache'))
            import kagglehub
            report(state='downloading')
            root = Path(kagglehub.dataset_download(DATASET))
        split = find_split(root)
        train_data, valid_data = LeafDataset(split / 'train'), LeafDataset(split / 'valid')
        steps_per_epoch = math.ceil(len(train_data) / args.batch_size)
        model = ResNet9().to(memory_format=torch.channels_last)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate,
            total_steps=args.epochs * steps_per_epoch)
        criterion = nn.CrossEntropyLoss()
        checkpoint_path = output / 'training-checkpoint.pth'
        epoch_start, batch_start, best_accuracy = 0, 0, -1.0
        epoch_loss = epoch_correct = epoch_seen = 0
        metrics = []
        if args.resume and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            expected = {'epochs': args.epochs, 'batch_size': args.batch_size,
                        'steps_per_epoch': steps_per_epoch, 'classes': CLASS_NAMES,
                        'learning_rate': args.learning_rate, 'dataset': DATASET}
            if checkpoint['config'] != expected:
                raise ValueError('Resume configuration differs from checkpoint; use the original settings')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            epoch_start, batch_start = checkpoint['epoch'], checkpoint['next_batch']
            best_accuracy, metrics = checkpoint['best_accuracy'], checkpoint['metrics']
            epoch_loss, epoch_correct, epoch_seen = checkpoint['epoch_totals']
            torch.set_rng_state(checkpoint['torch_rng'])
        atomic_json(output / 'classes.json', CLASS_NAMES)
        report(state='training', train_images=len(train_data), validation_images=len(valid_data),
               batch_size=args.batch_size, batches_per_epoch=steps_per_epoch,
               epoch=epoch_start + 1, completed_batches=batch_start, metrics=metrics,
               resumed=bool(args.resume and checkpoint_path.exists()))
        def save(epoch, next_batch):
            atomic_save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'epoch': epoch, 'next_batch': next_batch,
                'best_accuracy': best_accuracy, 'metrics': metrics,
                'epoch_totals': [epoch_loss, epoch_correct, epoch_seen],
                'torch_rng': torch.get_rng_state(),
                'config': {'epochs': args.epochs, 'batch_size': args.batch_size,
                           'steps_per_epoch': steps_per_epoch, 'classes': CLASS_NAMES,
                           'learning_rate': args.learning_rate, 'dataset': DATASET}}, checkpoint_path)
        started = time.monotonic()
        completed_this_run = 0
        next_epoch, next_batch = epoch_start, batch_start
        try:
            for epoch in range(epoch_start, args.epochs):
                model.train()
                order = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(7 + epoch)).tolist()
                first_batch = batch_start if epoch == epoch_start else 0
                remaining = Subset(train_data, order[first_batch * args.batch_size:])
                loader = DataLoader(remaining, batch_size=args.batch_size, num_workers=args.workers,
                                    persistent_workers=args.workers > 0)
                for batch, (images, labels) in enumerate(loader, start=first_batch):
                    images = images.to(memory_format=torch.channels_last)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(images)
                    loss = criterion(logits, labels)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Training loss is non-finite')
                    loss.backward()
                    nn.utils.clip_grad_value_(model.parameters(), 0.1)
                    optimizer.step()
                    scheduler.step()
                    epoch_loss += loss.item() * len(labels)
                    epoch_correct += int((logits.argmax(1) == labels).sum())
                    epoch_seen += len(labels)
                    completed_this_run += 1
                    next_epoch, next_batch = epoch, batch + 1
                    if (batch + 1) % 250 == 0:
                        save(next_epoch, next_batch)
                    if (batch + 1) % 25 == 0 or batch == first_batch:
                        rate = completed_this_run / (time.monotonic() - started)
                        left = (args.epochs - epoch - 1) * steps_per_epoch + steps_per_epoch - batch - 1
                        report(state='training', epoch=epoch + 1, completed_batches=batch + 1,
                               train_loss=round(epoch_loss / epoch_seen, 5),
                               train_accuracy=round(epoch_correct / epoch_seen, 5),
                               estimated_training_seconds_remaining=round(left / rate),
                               learning_rate=optimizer.param_groups[0]['lr'])
                save(epoch, steps_per_epoch)
                report(state='validating', epoch=epoch + 1, validated_images=0,
                       completed_batches=steps_per_epoch, train_loss=epoch_loss / epoch_seen,
                       train_accuracy=epoch_correct / epoch_seen)
                model.eval()
                valid_loss = valid_correct = valid_seen = 0
                confusion = torch.zeros(38, 38, dtype=torch.int64)
                with torch.inference_mode():
                    valid_loader = DataLoader(valid_data, batch_size=args.batch_size,
                                              num_workers=args.workers)
                    for batch, (images, labels) in enumerate(valid_loader):
                        logits = model(images.to(memory_format=torch.channels_last))
                        loss = criterion(logits, labels)
                        if not torch.isfinite(loss):
                            raise RuntimeError('Validation loss is non-finite')
                        predicted = logits.argmax(1)
                        valid_loss += loss.item() * len(labels)
                        valid_correct += int((predicted == labels).sum())
                        valid_seen += len(labels)
                        confusion += torch.bincount(labels * 38 + predicted, minlength=38*38).reshape(38,38)
                        if (batch + 1) % 100 == 0:
                            report(validated_images=valid_seen)
                accuracy = valid_correct / valid_seen
                metric = {'epoch': epoch + 1, 'train_loss': epoch_loss / epoch_seen,
                          'train_accuracy': epoch_correct / epoch_seen,
                          'validation_loss': valid_loss / valid_seen,
                          'validation_accuracy': accuracy}
                metrics.append(metric)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    atomic_save(model.state_dict(), output / 'plant-disease-model.pth')
                    atomic_json(output / 'validation-report.json', {
                        'dataset': DATASET, 'split': 'valid', 'images': valid_seen,
                        'best_epoch': epoch + 1, 'accuracy': accuracy, 'classes': CLASS_NAMES,
                        'confusion_matrix': confusion.tolist(), 'metrics': metrics,
                        'note': 'Validation split is from an augmented dataset; this is not independent field-image accuracy.'})
                epoch_loss = epoch_correct = epoch_seen = 0
                next_epoch, next_batch = epoch + 1, 0
                save(next_epoch, next_batch)
                report(metrics=metrics, best_validation_accuracy=best_accuracy, validated_images=valid_seen)
            best_model = load_model(output / 'plant-disease-model.pth')
            sample_path, sample_label = valid_data.samples[0]
            with Image.open(sample_path) as image:
                sample_image = image.convert('RGB')
                smoke_predictions = predict(best_model, sample_image)
                payload = io.BytesIO()
                sample_image.save(payload, format='JPEG')
            atomic_json(output / 'inference-check.json', {
                'expected_label': CLASS_NAMES[sample_label], 'predictions': smoke_predictions,
                'note': 'Single-image integration check; use validation-report.json for aggregate quality.'})
            report(state='complete', epoch=args.epochs, completed_batches=steps_per_epoch,
                   best_validation_accuracy=best_accuracy, metrics=metrics,
                   finished_at=datetime.now(timezone.utc).isoformat())
            if args.verify_app:
                # Wait for the dashboard's checkpoint watcher, then exercise its real API.
                verified = False
                for _ in range(30):
                    try:
                        with urllib.request.urlopen(args.verify_app.rstrip('/') + '/api/health', timeout=5) as response:
                            ready = json.load(response)['model_ready']
                        if ready:
                            request = urllib.request.Request(args.verify_app.rstrip('/') + '/api/predict',
                                data=payload.getvalue(), headers={'Content-Type': 'image/jpeg'}, method='POST')
                            with urllib.request.urlopen(request, timeout=60) as response:
                                api_result = json.load(response)
                            if len(api_result.get('predictions', [])) != 3:
                                raise ValueError('Unexpected prediction response')
                            atomic_json(output / 'app-inference-check.json', api_result)
                            verified = True
                            break
                    except (OSError, ValueError, KeyError):
                        pass
                    time.sleep(2)
                report(application_verified=verified)
        except KeyboardInterrupt:
            save(next_epoch, next_batch)
            report(state='interrupted', message='Resume with the same command and --resume.')
            raise
    except Exception as exc:
        report(state='failed', message=str(exc))
        traceback.print_exc()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', help='Existing extracted Kaggle dataset directory')
    parser.add_argument('--output', default=str(ROOT / 'models'))
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verify-app', help='After training, verify the running app at this base URL')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.workers < 0 or args.learning_rate <= 0:
        parser.error('Epochs, batch size, threads and learning rate must be positive; workers must be non-negative')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(output / 'training.lock'), timeout=0):
            train(args)
    except Timeout:
        raise SystemExit('Another training process is already using this output directory.')
