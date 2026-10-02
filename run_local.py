#!/usr/bin/env python3
"""Start a detached local training job with a persistent log and PID."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['start', 'status'])
    p.add_argument('--output', type=Path, default=Path('runs/mimu_paper150'))
    p.add_argument('--resume', type=Path)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--optimizer', choices=['adam', 'adamw'], default='adam')
    p.add_argument('--cosine-scheduler', action='store_true')
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
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    output = (root / args.output).resolve()
    pidfile = output / 'pid.txt'
    alive = False
    if pidfile.exists():
        try:
            os.kill(int(pidfile.read_text()), 0)
            alive = True
        except ProcessLookupError:
            pass
        except PermissionError:
            alive = True
    if args.action == 'status':
        print('Process alive:', alive)
        status = output / 'status.json'
        print(status.read_text() if status.exists() else 'Status not yet written')
        log = output / 'training.log'
        if log.exists():
            print('\n'.join(log.read_text().splitlines()[-8:]))
        return
    if alive:
        raise SystemExit('A process is already running for this output directory.')
    if args.resume and not (root / args.resume).exists():
        raise FileNotFoundError(args.resume)
    # Fail before detaching if GPU is hidden by the sandbox.
    import torch
    if not torch.backends.mps.is_available():
        raise SystemExit('Apple MPS GPU unavailable. Run this command outside the sandbox.')
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['TORCH_HOME'] = str(root / '.torch_cache')
    command = [sys.executable, '-u', str(root / 'train.py'), '--device', 'mps',
               '--epochs', str(args.epochs), '--output', str(output),
               '--batch-size', str(args.batch_size), '--size', str(args.size), '--lr', str(args.lr),
               '--weight-decay', str(args.weight_decay), '--optimizer', args.optimizer]
    if args.cosine_scheduler:
        command.append('--cosine-scheduler')
    if args.patch_train:
        command.extend(['--patch-train', '--eutectic-dice-weight', str(args.eutectic_dice_weight),
                        '--eutectic-patch-threshold', str(args.eutectic_patch_threshold),
                        '--eutectic-patch-probability', str(args.eutectic_patch_probability),
                        '--tile-stride', str(args.tile_stride), '--pixel-loss', args.pixel_loss,
                        '--pixel-loss-weight', str(args.pixel_loss_weight),
                        '--dice-loss-weight', str(args.dice_loss_weight),
                        '--architecture', args.architecture])
    elif args.architecture != 'mimunet':
        command.extend(['--architecture', args.architecture])
    if args.no_imagenet_pretrained:
        command.append('--no-imagenet-pretrained')
    if args.resume:
        command.extend(['--resume', str((root / args.resume).resolve())])
    with (output / 'training.log').open('a') as log:
        proc = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True, env=env)
    pidfile.write_text(str(proc.pid))
    (output / 'launch.json').write_text(json.dumps({'pid': proc.pid, 'command': command}, indent=2))
    print(f'Started PID {proc.pid}\nLog: {output / "training.log"}')


if __name__ == '__main__':
    main()
