#!/usr/bin/env python3
"""Explicit commands only: invoking without a command never starts training."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

from komap_lab.config import load_config
from komap_lab.data import FullResolutionDataset, manifest
from komap_lab.evaluation import evaluate
from komap_lab.models import SegmentationModel
from komap_lab.runtime import device_for, environment, seed_everything, write_json
from komap_lab.training import train, pretrain_ssl

ROOT = Path(__file__).resolve().parent


def check(args):
    paths = [args.config] if args.config else sorted((ROOT / 'configs').glob('*.json'))
    device = device_for(args.device)
    report = {'environment': environment(device), 'note': 'Synthetic forward/backward only; no dataset training or downloads',
              'checks': []}
    for path in paths:
        config = load_config(path)
        architecture = config['model']
        status = {'id': config['id'], 'config': Path(path).name}
        if architecture['mamba_blocks'] and device.type != 'cuda':
            status.update(status='skipped', reason='C10 requires Linux CUDA + mamba-ssm')
        else:
            try:
                seed_everything(config['train']['seed'])
                model = SegmentationModel(config, pretrained=False).to(device)
                model.train(args.backward)
                shape = config['train']['patch'] if args.backward or architecture['hybrid'] != 'none' else args.size
                image = torch.rand(1, 1, shape, shape, device=device)
                started = time.perf_counter()
                if args.backward:
                    output = model(image)
                    output.square().mean().backward()
                    gradients = [p.grad for p in model.parameters() if p.grad is not None]
                    if not gradients or any(not g.isfinite().all() for g in gradients):
                        raise RuntimeError('Nonfinite/missing backward gradients')
                else:
                    with torch.inference_mode():
                        output = model(image)
                expected_size = config['train']['target'] if args.backward else shape
                if output.shape != (1, 4, expected_size, expected_size) or not output.isfinite().all():
                    raise RuntimeError(f'Bad output: {tuple(output.shape)}')
                status.update(status='passed', input=list(image.shape), output=list(output.shape),
                              backward=args.backward, parameters=sum(p.numel() for p in model.parameters()),
                              seconds=time.perf_counter() - started)
                del model, image, output
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            except RuntimeError as exc:
                if 'requires: pip install' in str(exc) or 'requires a compatible Linux CUDA' in str(exc):
                    status.update(status='skipped', reason=str(exc))
                else:
                    status.update(status='failed', reason=str(exc))
            except Exception as exc:
                status.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
        report['checks'].append(status)
        print(json.dumps(status), flush=True)
    if args.data:
        dataset_manifest = manifest(args.data, patch=load_config(paths[0])['train']['target'])
        report['dataset_counts'] = {k: len(v) for k, v in dataset_manifest.items()}
        print(json.dumps({'dataset_counts': report['dataset_counts']}), flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.report, report)
    failed = any(s['status'] == 'failed' for s in report['checks'])
    skipped = any(s['status'] == 'skipped' for s in report['checks'])
    if failed or (args.require_all and skipped):
        raise SystemExit(1)


def evaluate_checkpoint(args):
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if state.get('phase') != 'segmentation':
        raise ValueError('Expected this package\'s segmentation checkpoint, not an older local checkpoint')
    config = state['config']
    device = device_for(args.device)
    model = SegmentationModel(config, pretrained=False).to(device)
    model.load_state_dict(state['model'], strict=True)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Evaluation output must be new/empty')
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = FullResolutionDataset(args.data, args.split)
    if args.split == 'valid':
        # Detect changed validation data; Train need not be mounted during evaluation.
        current = []
        import hashlib
        for path, image in zip(dataset.paths, dataset.images):
            mask_path = args.data / 'valid/masks' / path.name.replace('_image', '_mask')
            current.append({'image': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'shape': list(image.shape), 'mask_sha256': hashlib.sha256(mask_path.read_bytes()).hexdigest()})
        if current != state['data_manifest']['valid']:
            raise ValueError('Validation images/masks differ from checkpoint manifest')
    result = evaluate(model, dataset, config, device, d4=args.d4, export=args.output / 'masks', diagnostic=True)
    result.update(id=config['id'], checkpoint_epoch=state['epoch'], split=args.split, environment=environment(device))
    write_json(args.output / 'metrics.json', result)
    print(json.dumps({'id': config['id'], 'split': args.split, 'miou': result.get('miou'), 'output': str(args.output)}))


def compare(args):
    summaries = [json.loads((p / 'summary.json').read_text()) for p in args.runs]
    configs = [load_config(p / 'config.json') for p in args.runs]
    manifests = [json.loads((p / 'data_manifest.json').read_text()) for p in args.runs]
    import copy
    def protocol(c):
        selected = copy.deepcopy(c)
        for key in ('id', 'description', 'model', 'ssl'):
            selected.pop(key)
        return selected
    if any(m != manifests[0] for m in manifests[1:]) or any(protocol(c) != protocol(configs[0]) for c in configs[1:]):
        raise ValueError('Different data/normalization/training/evaluation protocols; comparison rejected')
    environments = [json.loads((p / 'environment.json').read_text()) for p in args.runs]
    if any(e['code_sha256'] != environments[0]['code_sha256'] for e in environments[1:]):
        raise ValueError('Different model/training code versions; comparison rejected')
    filenames = summaries[0]['single']['filenames']
    if any(s['single']['filenames'] != filenames for s in summaries):
        raise ValueError('Validation ordering differs')
    rows = []
    for path, summary in zip(args.runs, summaries):
        single, d4 = summary['single'], summary.get('d4', {})
        rows.append({'run': str(path), 'id': summary['id'], 'best_epoch': summary['best_epoch'],
                     'single_miou': single['miou'], 'd4_miou': d4.get('miou'),
                     'eutectic_iou': single['class_iou'][2], **single['eutectic'],
                     'seconds_per_image': single['seconds_per_image'], 'peak_cuda_memory_bytes': summary['peak_cuda_memory_bytes']})
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Comparison output must be new/empty')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'comparison.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pairs = [{'filename': name, 'baseline_single_miou': summaries[0]['single']['per_image_miou'][i],
              'deltas_vs_first_run': {str(path): summary['single']['per_image_miou'][i]
                                     - summaries[0]['single']['per_image_miou'][i]
                                     for path, summary in zip(args.runs[1:], summaries[1:])}}
             for i, name in enumerate(filenames)]
    write_json(args.output / 'paired_deltas.json', pairs)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=4, help='CPU threads; place before subcommand')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list', help='List configurations, no execution')
    check_parser = sub.add_parser('check', help='Offline synthetic model checks, no training')
    check_parser.add_argument('--config', type=Path)
    check_parser.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'mps'), default='cpu')
    check_parser.add_argument('--size', type=int, default=64, help='C09 always uses configured patch size')
    check_parser.add_argument('--backward', action='store_true')
    check_parser.add_argument('--require-all', action='store_true', help='Return failure for skipped optional architectures')
    check_parser.add_argument('--data', type=Path, help='Optional dataset pair/palette/overlap checks')
    check_parser.add_argument('--report', type=Path)
    for command in ('train', 'ssl'):
        child = sub.add_parser(command)
        child.add_argument('--config', type=Path, required=True)
        child.add_argument('--data', type=Path, required=True)
        child.add_argument('--output', type=Path, required=True)
        child.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'mps'), default='cuda')
        child.add_argument('--resume', action='store_true')
        if command == 'train':
            child.add_argument('--encoder-weights', type=Path)
            child.add_argument('--encoder-source', help='MicroNet version/variant/download source')
    ev = sub.add_parser('evaluate')
    ev.add_argument('--checkpoint', type=Path, required=True)
    ev.add_argument('--data', type=Path, required=True)
    ev.add_argument('--output', type=Path, required=True)
    ev.add_argument('--split', choices=('valid', 'test'), default='valid')
    ev.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'mps'), default='cuda')
    ev.add_argument('--d4', action='store_true')
    cmp = sub.add_parser('compare')
    cmp.add_argument('--runs', type=Path, nargs='+', required=True, help='First run is baseline for paired deltas')
    cmp.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    if args.command == 'list':
        for path in sorted((ROOT / 'configs').glob('*.json')):
            config = load_config(path)
            print(f'{config["id"]:24s} {path.name:28s} {config["description"]}')
    elif args.command == 'check':
        check(args)
    elif args.command == 'train':
        train(load_config(args.config), args.data, args.output, device_for(args.device), args.resume,
              args.encoder_weights, args.encoder_source)
    elif args.command == 'ssl':
        pretrain_ssl(load_config(args.config), args.data, args.output, device_for(args.device), args.resume)
    elif args.command == 'evaluate':
        evaluate_checkpoint(args)
    elif args.command == 'compare':
        compare(args)


if __name__ == '__main__':
    main()
