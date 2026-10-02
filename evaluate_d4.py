#!/usr/bin/env python3
"""Evaluate a ResNet34 U-Net checkpoint with 8-way D4 test-time augmentation."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from mimu.data import FullResolutionDataset, PALETTE, _patch_origins
from mimu.metrics import class_iou, confusion_matrix, summarize
from mimu.resnet34_unet import ResNet34UNet


@torch.inference_mode()
def predict_image(model, image, size, stride, device, batch_size):
    h, w = image.shape
    origins = _patch_origins(h, w, size, stride)
    probability_sum = torch.zeros((4, h, w), dtype=torch.float32)
    coverage = torch.zeros((h, w), dtype=torch.float32)
    for start in range(0, len(origins), batch_size):
        batch_origins = origins[start:start + batch_size]
        patches = torch.stack([
            torch.from_numpy(image[y:y+size, x:x+size].copy()).float().unsqueeze(0) / 255.0
            for y, x in batch_origins
        ]).to(device)
        tile_probabilities = torch.zeros((len(batch_origins), 4, size, size), dtype=torch.float32)
        for k in range(4):
            for flip in (False, True):
                view = torch.rot90(patches, k, dims=(-2, -1))
                if flip:
                    view = view.flip(-1)
                probs = model(view).softmax(1)
                if flip:
                    probs = probs.flip(-1)
                probs = torch.rot90(probs, -k, dims=(-2, -1))
                tile_probabilities += probs.cpu() / 8.0
        for (y, x), probs in zip(batch_origins, tile_probabilities):
            probability_sum[:, y:y+size, x:x+size] += probs
            coverage[y:y+size, x:x+size] += 1
    return probability_sum / coverage.clamp_min(1)[None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/resnet34_unet_scse_ce_dice/best.pt'))
    parser.add_argument('--data', type=Path, default=Path('Data'))
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='mps')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--output', type=Path, default=Path('runs/resnet34_unet_scse_ce_dice/d4_tta'))
    args = parser.parse_args()
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable')
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    if config['architecture'] != 'resnet34-unet-scse-decoder':
        raise ValueError(f"Expected scSE ResNet34 U-Net checkpoint, got {config['architecture']}")
    size, stride = config['size'], config['tile_stride']
    model = ResNet34UNet(pretrained=False, use_scse=True)
    model.load_state_dict(checkpoint['model'])
    model.pretrained = not config['no_imagenet_pretrained']
    model.to(device).eval()
    dataset = FullResolutionDataset(args.data, 'valid')
    matrices, predictions = [], []
    for index, image in enumerate(dataset.images):
        probs = predict_image(model, image, size, stride, device, args.batch_size)
        prediction = probs.argmax(0).numpy().astype(np.uint8)
        predictions.append(prediction)
        matrices.append(confusion_matrix(dataset.original_labels[index], prediction))
        print(f"{index+1}/{len(dataset.paths)} {dataset.paths[index].name}", flush=True)
    result = summarize(matrices, 'one')
    result['per_image_class_iou'] = [class_iou(cm, 'one').tolist() for cm in matrices]
    result['filenames'] = [p.name for p in dataset.paths]
    result['tta'] = '8-view D4; inverse-transform and average probabilities per tile; overlap-average tiles'
    args.output.mkdir(parents=True, exist_ok=True)
    for path, pred in zip(dataset.paths, predictions):
        Image.fromarray(PALETTE[pred]).save(args.output / path.name.replace('_image', '_mask'))
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({'miou': result['miou'], 'class_iou': result['class_iou'],
                      'output': str(args.output.resolve())}, indent=2), flush=True)


if __name__ == '__main__':
    main()
