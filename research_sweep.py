#!/usr/bin/env python3
"""Sequential, resumable paper-inspired experiments and paired result report."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from mimu.data import NativePatchDataset, FullResolutionDataset, PALETTE, decode_mask
from mimu.loss import HybridLoss
from mimu.metrics import confusion_matrix, summarize, class_iou
from mimu.research_variants import ResearchUNet, pixel_contrast, boundary_loss, cldice_loss
from evaluate_d4 import predict_image
from train import evaluate, save_checkpoint, write_json, cpu_tree, data_manifest

ROOT = Path(__file__).resolve().parent
METHODS = ['contrast', 'snake', 'boundary', 'cldice']
WEIGHTS = {'contrast': .05, 'snake': 0., 'boundary': .1, 'cldice': .05, 'baseline': 0.}


def audit_topology(output):
    """Describe GT morphology and baseline centerline coverage, without tuning."""
    from scipy import ndimage as ndi
    output.mkdir(parents=True, exist_ok=True)
    def skeleton(mask):
        cross = ndi.generate_binary_structure(2, 1)
        skel = np.zeros_like(mask)
        for _ in range(64):
            eroded = ndi.binary_erosion(mask, structure=cross)
            opened = ndi.binary_dilation(eroded, structure=cross)
            skel |= mask & ~opened
            mask = eroded
            if not mask.any():
                break
        return skel
    training, validation = [], []
    for path in sorted((ROOT/'Data/train/masks').glob('*.png')):
        mask = decode_mask(np.asarray(Image.open(path).convert('RGB'))) == 2
        dist = ndi.distance_transform_edt(mask)
        _, components = ndi.label(mask, structure=np.ones((3, 3)))
        training.append({'file': path.name, 'eutectic_pixels': int(mask.sum()),
                         'fraction_within_2px_of_background': float((dist[mask] <= 2).mean()) if mask.any() else None,
                         'components_8_connected': int(components)})
    prediction_root = ROOT/'runs/resnet34_unet_scse_ce_dice/valid_masks'
    for path in sorted((ROOT/'Data/valid/masks').glob('*.png')):
        gt = decode_mask(np.asarray(Image.open(path).convert('RGB'))) == 2
        pr = decode_mask(np.asarray(Image.open(prediction_root/path.name).convert('RGB'))) == 2
        gs, ps = skeleton(gt), skeleton(pr)
        recall = float((gs & pr).sum() / max(gs.sum(), 1))
        precision = float((ps & gt).sum() / max(ps.sum(), 1))
        labels, n = ndi.label(gt, structure=np.ones((3, 3)))
        fragmented, eligible = 0, 0
        for index, slices in enumerate(ndi.find_objects(labels), 1):
            if slices is None:
                continue
            component = labels[slices] == index
            if component.sum() < 16:
                continue
            eligible += 1
            _, pieces = ndi.label(component & pr[slices], structure=np.ones((3, 3)))
            fragmented += int(pieces > 1)
        validation.append({'file': path.name, 'skeleton_recall': recall,
                           'skeleton_precision': precision, 'eligible_gt_components': eligible,
                           'fragmented_gt_components': fragmented})
    result = {'train': training, 'valid_baseline': validation,
              'notes': 'Morphological binary skeleton diagnostic; intersection fragmentation is a proxy, not proof of biological connectivity.',
              'cldice_decision': 'Run as exploratory fourth ablation, matching disconnected GT masks; no constraint that all Eutectic must connect.'}
    write_json(output/'topology_audit.json', result)
    print('TOPOLOGY AUDIT', json.dumps({'skeleton_recall_mean': np.mean([x['skeleton_recall'] for x in validation]),
                                      'fragmented_gt_components': sum(x['fragmented_gt_components'] for x in validation)}), flush=True)


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if any(key in name for key in ['projection', 'boundary_head', '.snake.']):
            continue
        name = name.replace('decoder1.ordinary.', 'decoder1.')
        digest.update(name.encode())
        digest.update(tensor.cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def d4_eval(model, dataset, output, device):
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    matrices = []
    for i, image in enumerate(dataset.images):
        probs = predict_image(model, image, 224, 112, device, 4)
        pred = probs.argmax(0).numpy().astype(np.uint8)
        matrices.append(confusion_matrix(dataset.original_labels[i], pred))
        Image.fromarray(PALETTE[pred]).save(output/dataset.paths[i].name.replace('_image', '_mask'))
        print(f'D4 {i+1}/{len(dataset.paths)}', flush=True)
    metrics = summarize(matrices, 'one')
    metrics['filenames'] = [p.name for p in dataset.paths]
    metrics['per_image_class_iou'] = [class_iou(cm, 'one').tolist() for cm in matrices]
    metrics['confusion_matrix_sum'] = np.sum(matrices, axis=0).tolist()
    write_json(output/'metrics.json', metrics)
    return metrics


def run_method(method, args):
    output = args.output/method
    output.mkdir(parents=True, exist_ok=True)
    status_path = output/'status.json'
    if status_path.exists() and json.loads(status_path.read_text()).get('state') == 'completed':
        print('ALREADY COMPLETED', method, flush=True)
        return
    seed = getattr(args, 'seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    model = ResearchUNet(method, pretrained=True)
    initial_digest = model_digest(model)
    model.to(device)
    criterion = HybridLoss(pixel_loss='ce', pixel_weight=.5, dice_weight=.5,
                           class_weights=(1, 1, 2, 1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train = NativePatchDataset(ROOT/'Data', patch=224, stride=112,
                               eutectic_fraction=.1, enriched_probability=.5)
    valid = FullResolutionDataset(ROOT/'Data', 'valid')
    valid.size, valid.stride = 224, 112
    loader_rng = torch.Generator().manual_seed(seed)
    aux_rng = torch.Generator().manual_seed(314159)
    loader = DataLoader(train, batch_size=4, shuffle=True, num_workers=0, generator=loader_rng)
    manifest = data_manifest(ROOT/'Data')
    config = {'method': method, 'epochs': args.epochs, 'seed': seed, 'size': 224, 'tile_stride': 112,
              'batch_size': 4, 'samples_per_epoch': 420, 'lr': 3e-4, 'optimizer': 'AdamW',
              'weight_decay': 1e-4, 'scheduler': 'cosine', 'dice_class_weights': [1,1,2,1],
              'loss': '.5 CE + .5 weighted Dice + ramp * auxiliary_weight * auxiliary',
              'auxiliary_weight': WEIGHTS[method], 'auxiliary_warmup': '0 through epoch 10; linear ramp to full at 20',
              'initial_common_weights_sha256': initial_digest,
              'parameters': sum(p.numel() for p in model.parameters()),
              'architecture': getattr(args, 'architecture', 'resnet34-unet-scse-decoder'), 'no_imagenet_pretrained': False,
              'sampling': '50% rich (eutectic >=10%), 50% all; source-uniform within pool',
              'evaluator': 'original-resolution per-image four-class mean IoU, absent=one'}
    config.update(getattr(args, 'extra_config', {}))
    write_json(output/'config.json', config)
    history, best, best_epoch, start = [], -1., 0, 0
    if (output/'last.pt').exists():
        checkpoint = torch.load(output/'last.pt', map_location='cpu', weights_only=True)
        if checkpoint['config'] != config or checkpoint['manifest'] != manifest:
            raise ValueError('Resume config or dataset mismatch')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        history, start = checkpoint['history'], checkpoint['epoch']
        best, best_epoch = checkpoint['best_miou'], checkpoint['best_epoch']
        loader_rng.set_state(checkpoint['loader_rng'])
        aux_rng.set_state(checkpoint['aux_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
    write_json(output/'data_manifest.json', manifest)
    print('START', method, 'epoch', start+1, 'parameters', config['parameters'], flush=True)
    try:
        for epoch in range(start+1, args.epochs+1):
            begun = time.monotonic()
            model.train()
            sums = np.zeros(3)
            count = 0
            factor = min(1., max(0., (epoch-10)/10))
            epoch_lr = optimizer.param_groups[0]['lr']
            for images, labels, sources in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits, features = model(images, auxiliary=True)
                primary = criterion(logits, labels)
                auxiliary = logits.sum() * 0
                if factor > 0:
                    if method == 'contrast':
                        auxiliary = pixel_contrast(model, features, logits, labels, sources, aux_rng)
                    elif method == 'boundary':
                        auxiliary = boundary_loss(model, features, labels)
                    elif method == 'cldice':
                        auxiliary = cldice_loss(logits, labels)
                loss = primary + factor * WEIGHTS[method] * auxiliary
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                optimizer.step()
                sums += np.array([loss.item(), primary.item(), auxiliary.item()]) * len(images)
                count += len(images)
            metrics = evaluate(model, valid, device, 4, criterion, 'one')
            scheduler.step()
            improved = metrics['miou'] > best
            if improved:
                best, best_epoch = metrics['miou'], epoch
            row = {'epoch': epoch, 'train_loss': float(sums[0]/count), 'seg_loss': float(sums[1]/count),
                   'aux_loss': float(sums[2]/count), 'aux_factor': factor, 'lr': epoch_lr,
                   'val_miou': metrics['miou'], 'class_iou': metrics['class_iou'],
                   'per_image_miou': metrics['per_image_miou'], 'seconds': time.monotonic()-begun}
            history.append(row)
            checkpoint = {'epoch': epoch, 'method': method, 'model': cpu_tree(model.state_dict()),
                          'optimizer': cpu_tree(optimizer.state_dict()), 'scheduler': scheduler.state_dict(),
                          'config': config, 'manifest': manifest, 'best_miou': best, 'best_epoch': best_epoch,
                          'history': history, 'loader_rng': loader_rng.get_state(),
                          'aux_rng': aux_rng.get_state(), 'torch_rng': torch.get_rng_state()}
            save_checkpoint(output/'last.pt', checkpoint)
            if improved:
                save_checkpoint(output/'best.pt', checkpoint)
            write_json(output/'history.json', history)
            write_json(status_path, {'state': 'training', 'method': method, 'pid': os.getpid(),
                                    'completed_epochs': epoch, 'target_epochs': args.epochs,
                                    'best_epoch': best_epoch, 'best_miou': best, 'latest': row})
            print(f'{method} {epoch}/{args.epochs} loss={row["train_loss"]:.4f} '
                  f'mIoU={metrics["miou"]:.5f} best={best:.5f}@{best_epoch} '
                  f'seconds={row["seconds"]:.1f}', flush=True)
        checkpoint = torch.load(output/'best.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model'])
        single = evaluate(model, valid, device, 4, criterion, 'one', output/'valid_masks')
        write_json(status_path, {'state': 'd4_evaluating', 'completed_epochs': args.epochs,
                                'best_epoch': best_epoch, 'best_miou': best})
        tta = d4_eval(model, valid, output/'d4_tta', device)
        write_json(status_path, {'state': 'completed', 'completed_epochs': args.epochs,
                                'best_epoch': best_epoch, 'best_miou': best,
                                'single': single, 'd4': tta})
    except BaseException as exc:
        write_json(status_path, {'state': 'failed', 'error': repr(exc),
                                'completed_epochs': history[-1]['epoch'] if history else start})
        raise


def create_report(output):
    baseline = ROOT/'runs/resnet34_unet_scse_ce_dice'
    base_single = json.loads((baseline/'validation_metrics.json').read_text())
    base_d4 = json.loads((baseline/'d4_tta/metrics.json').read_text())
    rows = [('existing scSE baseline', 78, base_single, base_d4)]
    for method in METHODS:
        path = output/method/'status.json'
        if path.exists():
            item = json.loads(path.read_text())
            if item.get('state') == 'completed':
                rows.append((method, item['best_epoch'], item['single'], item['d4']))
    lines = ['**Paper-inspired sequential experiment results**', '',
             'Single seed 42, 100 epochs, same 224-patch pipeline. Paper-inspired components; not full-paper reproductions.', '',
             '| Method | Best epoch | Single mIoU % | D4 mIoU % | D4 Al3Ni IoU % | D4 Eutectic IoU % | Δ D4 (pp) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    summary = []
    rng = np.random.default_rng(42)
    for name, epoch, single, d4 in rows:
        lines.append(f'| {name} | {epoch} | {100*single["miou"]:.4f} | {100*d4["miou"]:.4f} | '
                     f'{100*d4["class_iou"][1]:.4f} | {100*d4["class_iou"][2]:.4f} | '
                     f'{100*(d4["miou"]-base_d4["miou"]):+.4f} |')
        if d4['filenames'] != base_d4['filenames']:
            raise ValueError('Validation image order differs')
        delta = np.array(d4['per_image_miou']) - np.array(base_d4['per_image_miou'])
        draws = rng.integers(0, len(delta), size=(10000, len(delta)))
        interval = np.quantile(delta[draws].mean(1), [.025, .975])
        summary.append({'method': name, 'single_miou': single['miou'], 'd4_miou': d4['miou'],
                        'd4_delta_pp': 100*float(delta.mean()), 'paired_image_bootstrap_95ci_pp': (100*interval).tolist(),
                        'improved_images': int((delta>0).sum())})
    lines += ['', '**Retrospective early-stopping simulation**', '',
              'Minimum epoch 40; min_delta .001; selection still retains the best actual mIoU checkpoint.', '',
              '| Method | Patience | Stop epoch | Saved epochs | Lost single mIoU (pp) |',
              '|---|---:|---:|---:|---:|']
    early = []
    for method in METHODS:
        path = output/method/'history.json'
        if not path.exists():
            continue
        history = json.loads(path.read_text())
        for patience in (15, 20, 30):
            meaningful, last_improvement, stop = -1., 0, len(history)
            for row in history:
                if row['val_miou'] > meaningful + .001:
                    meaningful, last_improvement = row['val_miou'], row['epoch']
                if row['epoch'] >= 40 and row['epoch']-last_improvement >= patience:
                    stop = row['epoch']
                    break
            missed = max(x['val_miou'] for x in history)-max(x['val_miou'] for x in history[:stop])
            early.append({'method': method, 'patience': patience, 'stop_epoch': stop,
                          'saved_epochs': len(history)-stop, 'lost_miou_pp': 100*missed})
            lines.append(f'| {method} | {patience} | {stop} | {len(history)-stop} | {100*missed:.4f} |')
    write_json(output/'early_stopping_review.json', early)
    lines += ['', 'Bootstrap intervals measure image resampling only; they do not capture seed variation or validation model-selection bias.',
              'Checkpoint chosen using single-view validation, then evaluated once with D4. No hyperparameter search on validation.',
              'See experiment configs, per-epoch per-image histories, masks, topology_audit.json, and summary.json for details.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    write_json(output/'summary.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/research_sweep_v1')
    parser.add_argument('--device', default='mps', choices=['cpu', 'mps'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS+['baseline'])
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.hub.set_dir(str(ROOT/'.torch_cache/hub'))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if not (args.output/'topology_audit.json').exists():
        audit_topology(args.output)
    if args.audit_only:
        return
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; launch outside sandbox')
    sources = args.output/'source_snapshot'
    sources.mkdir(exist_ok=True)
    for path in ['research_sweep.py', 'mimu/research_variants.py', 'mimu/resnet34_unet.py',
                 'mimu/data.py', 'mimu/loss.py', 'mimu/metrics.py', 'train.py', 'evaluate_d4.py']:
        destination = sources/path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(ROOT/path, destination)
    for method in args.methods:
        write_json(args.output/'status.json', {'state': 'running', 'active_method': method,
                                              'sequence': args.methods, 'pid': os.getpid()})
        run_method(method, args)
        create_report(args.output)
    write_json(args.output/'status.json', {'state': 'completed', 'sequence': args.methods})
    print((args.output/'RESULTS.md').read_text(), flush=True)


if __name__ == '__main__':
    main()
