#!/usr/bin/env python3
"""Train the published MIMU-Net graph on the fixed KoMaP split."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from mimu.data import CLASS_NAMES, PALETTE, PhaseDataset, NativePatchDataset, FullResolutionDataset, _patch_origins
from mimu.loss import HybridLoss
from mimu.metrics import confusion_matrix, summarize
from mimu.model import MIMUNet
from mimu.resnet34_unet import ResNet34UNet


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def save_checkpoint(path, data):
    tmp = path.with_suffix('.tmp')
    torch.save(data, tmp)
    tmp.replace(path)


def choose_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable. On macOS the sandbox may hide the GPU; run outside it or use --device cpu.')
    return torch.device(name)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def data_manifest(root):
    records = []
    for path in sorted(root.glob('*/*/*.png')):
        records.append({'path': str(path.relative_to(root)),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return records


@torch.inference_mode()
def tiled_probabilities(model, image, size, stride, device, batch_size):
    """Overlap-tile a uint8 HxW source and average class probabilities."""
    h, w = image.shape
    origins = _patch_origins(h, w, size, stride)
    sums = torch.zeros((4, h, w), dtype=torch.float32)
    counts = torch.zeros((h, w), dtype=torch.float32)
    for start in range(0, len(origins), batch_size):
        batch_origins = origins[start:start + batch_size]
        crops = torch.stack([
            torch.from_numpy(image[y:y+size, x:x+size].copy()).float().unsqueeze(0) / 255.0
            for y, x in batch_origins
        ]).to(device)
        probs = model(crops).softmax(1).cpu()
        for (y, x), prob in zip(batch_origins, probs):
            sums[:, y:y+size, x:x+size] += prob
            counts[y:y+size, x:x+size] += 1
    return sums / counts.clamp_min(1)[None]


@torch.inference_mode()
def evaluate(model, dataset, device, batch_size, criterion, absent, export=None):
    model.eval()
    matrices, resized_matrices, predictions = [], [], []
    total_loss = 0.0
    if getattr(dataset, 'full_resolution', False):
        tile_size = dataset.size
        stride = dataset.stride
        for index, image in enumerate(dataset.images):
            probs = tiled_probabilities(model, image, tile_size, stride, device, batch_size)
            pred = probs.argmax(0).numpy().astype(np.uint8)
            gt = dataset.original_labels[index]
            matrices.append(confusion_matrix(gt, pred))
            small_gt = np.asarray(Image.fromarray(gt).resize((tile_size, tile_size), Image.Resampling.NEAREST))
            small_pred = np.asarray(Image.fromarray(pred).resize((tile_size, tile_size), Image.Resampling.NEAREST))
            resized_matrices.append(confusion_matrix(small_gt, small_pred))
            if export is not None:
                export.mkdir(parents=True, exist_ok=True)
                Image.fromarray(PALETTE[pred]).save(export / dataset.paths[index].name.replace('_image', '_mask'))
                predictions.append(pred)
        result = summarize(matrices, absent)
        result['loss'] = None
        result['resized_miou'] = summarize(resized_matrices, absent)['miou']
        result['filenames'] = [p.name for p in dataset.paths]
        if export is not None:
            from mimu.metrics import class_iou
            result['per_image_class_iou'] = [class_iou(cm, 'one').tolist() for cm in matrices]
            result['per_image_class_iou_policy'] = 'one'
            write_json(export.parent / 'validation_metrics.json', result)
            make_preview(dataset, predictions, result, export.parent / 'validation_preview.jpg')
        return result
    for start in range(0, len(dataset.paths), batch_size):
        images = torch.stack(dataset.images[start:start + batch_size]).to(device)
        labels = torch.stack(dataset.labels[start:start + batch_size]).to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * len(images)
        probs = logits.softmax(1).cpu()
        for j in range(len(images)):
            index = start + j
            gt = dataset.original_labels[index]
            pred = F.interpolate(probs[j:j+1], size=gt.shape, mode='bilinear',
                                 align_corners=False).argmax(1)[0].numpy().astype(np.uint8)
            matrices.append(confusion_matrix(gt, pred))
            small_pred = probs[j].argmax(0).numpy().astype(np.uint8)
            resized_matrices.append(confusion_matrix(dataset.labels[index].numpy(), small_pred))
            if export is not None:
                export.mkdir(parents=True, exist_ok=True)
                Image.fromarray(PALETTE[pred]).save(export / dataset.paths[index].name.replace('_image', '_mask'))
                predictions.append(pred)
    result = summarize(matrices, absent)
    result['loss'] = total_loss / len(dataset.paths)
    result['resized_miou'] = summarize(resized_matrices, absent)['miou']
    result['filenames'] = [p.name for p in dataset.paths]
    if export is not None:
        result['per_image_class_iou'] = []
        from mimu.metrics import class_iou
        for cm in matrices:
            result['per_image_class_iou'].append(class_iou(cm, 'one').tolist())
        result['per_image_class_iou_policy'] = 'one'
        write_json(export.parent / 'validation_metrics.json', result)
        make_preview(dataset, predictions, result, export.parent / 'validation_preview.jpg')
    return result


def make_preview(dataset, predictions, result, path):
    # Show both the best and worst examples, not a handpicked success case.
    order = np.argsort(result['per_image_miou'])
    indices = list(order[:3]) + list(order[-3:])
    tile_w, tile_h = 320, 235
    canvas = Image.new('RGB', (3 * tile_w, len(indices) * (tile_h + 28) + 25), 'white')
    draw = ImageDraw.Draw(canvas)
    for col, title in enumerate(['Input', 'Ground truth', 'Prediction']):
        draw.text((col * tile_w + 5, 5), title, fill='black')
    for row, i in enumerate(indices):
        with Image.open(dataset.paths[i]) as im:
            input_image = im.convert('RGB')
        ims = [input_image, Image.fromarray(PALETTE[dataset.original_labels[i]]),
               Image.fromarray(PALETTE[predictions[i]])]
        y = 25 + row * (tile_h + 28)
        draw.text((5, y), f'{dataset.paths[i].stem}  mIoU={result["per_image_miou"][i]:.4f}', fill='black')
        for col, im in enumerate(ims):
            im.thumbnail((tile_w, tile_h), Image.Resampling.NEAREST if col else Image.Resampling.BILINEAR)
            canvas.paste(im, (col * tile_w, y + 20))
    canvas.save(path)


@torch.inference_mode()
def export_test(model, root, size, device, output):
    dataset = PhaseDataset(root, 'test', size)
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    for path, tensor in zip(dataset.paths, dataset.images):
        with Image.open(path) as im:
            w, h = im.size
        probs = model(tensor[None].to(device)).softmax(1).cpu()
        labels = F.interpolate(probs, size=(h, w), mode='bilinear', align_corners=False).argmax(1)[0].numpy()
        Image.fromarray(PALETTE[labels]).save(output / path.name.replace('_image', '_mask'))


@torch.inference_mode()
def export_test_tiled(model, dataset, device, batch_size, output):
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    for path, image in zip(dataset.paths, dataset.images):
        probs = tiled_probabilities(model, image, dataset.size, dataset.stride, device, batch_size)
        Image.fromarray(PALETTE[probs.argmax(0).numpy().astype(np.uint8)]).save(
            output / path.name.replace('_image', '_mask'))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('Data'))
    p.add_argument('--output', type=Path, default=Path('runs/mimu_paper'))
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--no-augment', action='store_true')
    p.add_argument('--patch-train', action='store_true')
    p.add_argument('--eutectic-dice-weight', type=float, default=1.0)
    p.add_argument('--eutectic-patch-threshold', type=float, default=0.10)
    p.add_argument('--eutectic-patch-probability', type=float, default=0.50)
    p.add_argument('--tile-stride', type=int, default=112)
    p.add_argument('--pixel-loss', choices=['focal', 'ce'], default='focal')
    p.add_argument('--pixel-loss-weight', type=float, default=1.0)
    p.add_argument('--dice-loss-weight', type=float, default=2.0)
    p.add_argument('--architecture', choices=['mimunet', 'resnet34_unet', 'resnet34_unet_scse'], default='mimunet')
    p.add_argument('--no-imagenet-pretrained', action='store_true')
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--optimizer', choices=['adam', 'adamw'], default='adam')
    p.add_argument('--cosine-scheduler', action='store_true')
    p.add_argument('--absent-policy', choices=['one', 'zero', 'ignore'], default='one')
    p.add_argument('--resume', type=Path)
    p.add_argument('--evaluate', type=Path, help='Evaluate/export an existing checkpoint without training')
    return p.parse_args()


def main():
    args = parse_args()
    if args.size < 32 or args.size % 16 or args.batch_size < 1 or args.epochs < 1:
        raise ValueError('size must be >=32 and divisible by 16; batch size and epochs positive')
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'history.json').exists() and not (args.resume or args.evaluate):
        raise FileExistsError('Output already has a run. Use --resume or a new --output.')
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    architecture_name = {
        'mimunet': 'author-connected-graph-b2d21ef',
        'resnet34_unet': 'resnet34-unet',
        'resnet34_unet_scse': 'resnet34-unet-scse-decoder',
    }[args.architecture]
    config.update({'device_used': str(device), 'torch_version': str(torch.__version__),
                   'classes': CLASS_NAMES, 'palette': PALETTE.tolist(),
                   'architecture': architecture_name,
                   'loss': f'{args.pixel_loss}(weight={args.pixel_loss_weight:g})+weighted_dice(weight={args.dice_loss_weight:g},eutectic={args.eutectic_dice_weight:g})',
                   'metric': 'mean_of_per_image_4class_iou_at_original_resolution'})
    if args.architecture == 'mimunet':
        model = MIMUNet()
    else:
        model = ResNet34UNet(pretrained=not args.no_imagenet_pretrained,
                             use_scse=args.architecture == 'resnet34_unet_scse')
    model = model.to(device)
    criterion = HybridLoss(class_weights=(1.0, 1.0, args.eutectic_dice_weight, 1.0),
                           pixel_loss=args.pixel_loss, pixel_weight=args.pixel_loss_weight,
                           dice_weight=args.dice_loss_weight).to(device)
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-7, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.cosine_scheduler else None
    config['parameters'] = sum(p.numel() for p in model.parameters())
    if args.patch_train:
        valid = FullResolutionDataset(args.data, 'valid')
        valid.size, valid.stride = args.size, args.tile_stride
    else:
        valid = PhaseDataset(args.data, 'valid', args.size)
    if args.evaluate:
        ckpt = torch.load(args.evaluate, map_location='cpu', weights_only=True)
        if args.size != ckpt['config']['size']:
            raise ValueError('Evaluation size must match checkpoint preprocessing')
        model.load_state_dict(ckpt['model'])
        metrics = evaluate(model, valid, device, args.batch_size, criterion, args.absent_policy,
                           args.output / 'valid_masks')
        if ckpt['config'].get('patch_train'):
            test = FullResolutionDataset(args.data, 'test')
            test.size, test.stride = args.size, args.tile_stride
            export_test_tiled(model, test, device, args.batch_size, args.output / 'test_masks')
        else:
            export_test(model, args.data, args.size, device, args.output / 'test_masks')
        print(json.dumps(metrics, indent=2), flush=True)
        return
    if args.patch_train:
        train = NativePatchDataset(args.data, patch=args.size, stride=args.tile_stride,
                                   eutectic_fraction=args.eutectic_patch_threshold,
                                   enriched_probability=args.eutectic_patch_probability)
    else:
        train = PhaseDataset(args.data, 'train', args.size, augment=not args.no_augment)
    config.update({'train_images': len(train.paths), 'train_samples_per_epoch': len(train),
                   'valid_images': len(valid.paths)})
    manifest = data_manifest(args.data)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        generator=generator)
    start_epoch, best, best_epoch, history = 0, -1.0, 0, []
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=True)
        for key in ['size', 'batch_size', 'lr', 'seed', 'no_augment', 'absent_policy', 'architecture', 'loss', 'patch_train', 'eutectic_dice_weight', 'eutectic_patch_threshold', 'eutectic_patch_probability', 'tile_stride', 'pixel_loss', 'pixel_loss_weight', 'dice_loss_weight', 'no_imagenet_pretrained', 'optimizer', 'weight_decay', 'cosine_scheduler']:
            if ckpt['config'][key] != config[key]:
                raise ValueError(f'Resume configuration differs: {key}')
        if ckpt['manifest'] != manifest:
            raise ValueError('Dataset content has changed since checkpoint')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if scheduler is not None and ckpt.get('scheduler') is not None:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch, best, best_epoch = ckpt['epoch'], ckpt['best_miou'], ckpt['best_epoch']
        history = ckpt['history']
        generator.set_state(ckpt['loader_rng'])
        torch.set_rng_state(ckpt['torch_rng'])
        # Best checkpoint must remain available when resuming into another folder.
        best_source = args.resume.parent / 'best.pt'
        if not (args.output / 'best.pt').exists() and best_source.exists():
            import shutil
            shutil.copy2(best_source, args.output / 'best.pt')
    write_json(args.output / 'config.json', config)
    write_json(args.output / 'data_manifest.json', manifest)
    write_json(args.output / 'status.json', {'state': 'training', 'pid': os.getpid(),
               'completed_epochs': start_epoch, 'target_epochs': args.epochs,
               'best_epoch': best_epoch, 'best_miou': best})
    print(json.dumps(config, indent=2), flush=True)
    print(f'PID={os.getpid()} | Starting at epoch {start_epoch + 1}', flush=True)
    try:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            begun = time.monotonic()
            model.train()
            loss_sum, count = 0.0, 0
            for step, (images, labels, _) in enumerate(loader, 1):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), labels)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f'Nonfinite loss at epoch {epoch}, step {step}')
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(images)
                count += len(images)
                if step == 1 or step % 20 == 0 or step == len(loader):
                    print(f'epoch {epoch}/{args.epochs} step {step}/{len(loader)} '
                          f'loss={loss_sum/count:.5f} elapsed={time.monotonic()-begun:.1f}s', flush=True)
            metrics = evaluate(model, valid, device, args.batch_size, criterion, args.absent_policy)
            if scheduler is not None:
                scheduler.step()
            improved = metrics['miou'] > best
            if improved:
                best, best_epoch = metrics['miou'], epoch
            row = {'epoch': epoch, 'train_loss': loss_sum / count, 'val_loss': metrics['loss'],
                   'val_miou': metrics['miou'], 'val_resized_miou': metrics['resized_miou'],
                   'class_iou': metrics['class_iou'],
                   'absent_one': metrics['miou_absent_one'], 'absent_zero': metrics['miou_absent_zero'],
                   'absent_ignore': metrics['miou_absent_ignore'], 'seconds': time.monotonic() - begun}
            history.append(row)
            checkpoint = {'epoch': epoch, 'model': cpu_tree(model.state_dict()),
                          'optimizer': cpu_tree(optimizer.state_dict()), 'config': config,
                          'scheduler': cpu_tree(scheduler.state_dict()) if scheduler is not None else None,
                          'best_miou': best, 'best_epoch': best_epoch, 'history': history,
                          'manifest': manifest, 'loader_rng': generator.get_state(),
                          'torch_rng': torch.get_rng_state()}
            save_checkpoint(args.output / 'last.pt', checkpoint)
            if improved:
                save_checkpoint(args.output / 'best.pt', checkpoint)
            write_json(args.output / 'history.json', history)
            with (args.output / 'history.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)
            write_json(args.output / 'status.json', {'state': 'training', 'pid': os.getpid(),
                       'completed_epochs': epoch, 'target_epochs': args.epochs,
                       'best_epoch': best_epoch, 'best_miou': best,
                       'latest': row})
            print(f'EPOCH {epoch} DONE | native mIoU={metrics["miou"]:.5f} '
                  f'224 mIoU={metrics["resized_miou"]:.5f} | best={best:.5f} '
                  f'@{best_epoch} | seconds={row["seconds"]:.1f}', flush=True)
        ckpt = torch.load(args.output / 'best.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(ckpt['model'])
        metrics = evaluate(model, valid, device, args.batch_size, criterion, args.absent_policy,
                           args.output / 'valid_masks')
        if args.patch_train:
            test = FullResolutionDataset(args.data, 'test')
            test.size, test.stride = args.size, args.tile_stride
            export_test_tiled(model, test, device, args.batch_size, args.output / 'test_masks')
        else:
            export_test(model, args.data, args.size, device, args.output / 'test_masks')
        write_json(args.output / 'status.json', {'state': 'completed', 'completed_epochs': args.epochs,
                   'best_epoch': best_epoch, 'best_miou': best, 'test_masks': 10,
                   'validation_metrics': metrics})
        print(f'COMPLETE: best epoch={best_epoch}, native mIoU={best:.5f}. Masks exported.', flush=True)
    except BaseException as exc:
        write_json(args.output / 'status.json', {'state': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                   'error': repr(exc), 'completed_epochs': history[-1]['epoch'] if history else start_epoch})
        raise


if __name__ == '__main__':
    main()
