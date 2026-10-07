#!/usr/bin/env python3
"""Locked sequential three-seed queue, with source/data/environment resume guards."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from komap_phase.config import ROOT, load_config
from komap_phase.data import manifest
from komap_phase.runtime import device_for, environment, write_json


def read(path):
    return json.loads(path.read_text())


def guard_equal(path, value):
    if path.exists() and read(path) != value:
        raise ValueError('Persisted identity differs: ' + str(path))
    if not path.exists():
        write_json(path, value)


def completed(directory, config, identity, env):
    if not (directory/'status.json').exists() or read(directory/'status.json').get('state') != 'completed':
        return False
    for name, value in [('config.json', config), ('data_manifest.json', identity)]:
        if read(directory/name) != value:
            raise ValueError('Completed run identity changed: ' + str(directory/name))
    prior = read(directory/'environment.json')
    for key in ('code_sha256', 'device', 'torch', 'torchvision', 'cuda_build', 'gpu', 'dependencies'):
        if prior.get(key) != env.get(key):
            raise ValueError('Completed environment differs: ' + key)
    history = read(directory/'history.json')
    if len(history) != config['train']['epochs'] or history[-1]['epoch'] != config['train']['epochs']:
        raise ValueError('Completed history incomplete')
    if not all((directory/name).is_file() for name in ('best.pt','last.pt')) or 'd4' not in read(directory/'summary.json'):
        raise ValueError('Completed checkpoints/evaluations missing')
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('cuda','cpu'), default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--batch-size', type=int, choices=(4,8,16), default=4)
    p.add_argument('--phase', choices=('primary','followup','all'), default='primary')
    p.add_argument('--skip-baseline', action='store_true', help='Require verified B results already in this same output')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.threads < 1:
        p.error('threads must be positive')
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()) and not args.resume:
        p.error('Nonempty output; use --resume or a new directory')
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'.suite.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name in ('logs','resolved_configs'):
        (root/name).mkdir(exist_ok=True)
    identity = manifest(args.data)
    if {k:len(v) for k,v in identity.items()} != {'train':70,'valid':20}:
        p.error('Full queue requires Train70/Valid20')
    env = environment(device_for(args.device))
    recipe = load_config(ROOT/'configs/B.json')
    recipe['train']['batch_size'] = args.batch_size
    recipe['evaluation']['batch_size'] = args.batch_size
    guard_equal(root/'suite_identity.json', dict(data=identity,
        environment={k:env.get(k) for k in ('code_sha256','device','torch','torchvision','cuda_build','gpu','dependencies')},
        seeds=[42,43,44], recipe=recipe))
    queue = []
    for seed in (42,43,44):
        ids = ([] if args.skip_baseline or args.phase == 'followup' else ['B'])
        if args.phase in ('primary','all'):
            ids += ['HT_BOT','HT_PYR']
        if args.phase in ('followup','all'):
            ids += ['HT_DEC16']
        if args.skip_baseline or args.phase == 'followup':
            base = copy.deepcopy(recipe); base['train']['seed'] = seed
            if not completed(root/f'seed{seed}/B', base, identity, env):
                p.error('Need completed same-package B for seed' + str(seed))
        for ident in ids:
            c = copy.deepcopy(load_config(ROOT/'configs'/f'{ident}.json'))
            c['train']['seed'] = seed
            c['train']['batch_size'] = args.batch_size
            c['evaluation']['batch_size'] = args.batch_size
            queue.append(c)
    worker = None
    done = []
    def status(state, **extra):
        write_json(root/'suite_status.json', dict(state=state, supervisor_pid=os.getpid(),
            phase=args.phase, queue_size=len(queue), completed=len(done), completed_runs=done, **extra))
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt('Signal; resume from the last committed epoch')
    signal.signal(signal.SIGTERM, interrupt)
    try:
        for c in queue:
            key = f"seed{c['train']['seed']}/{c['id']}"
            out = root/key
            config_path = root/'resolved_configs'/(key.replace('/','_')+'.json')
            guard_equal(config_path, c)
            if not completed(out, c, identity, env):
                if out.exists() and any(out.iterdir()) and not (out/'last.pt').exists():
                    # Preserve a failure before epoch1 without accepting changed inputs.
                    if (out/'config.json').exists() and read(out/'config.json') != c:
                        raise ValueError('Pre-epoch config changed')
                    if (out/'data_manifest.json').exists() and read(out/'data_manifest.json') != identity:
                        raise ValueError('Pre-epoch data changed')
                    from datetime import datetime, timezone
                    archives = root/'pre_epoch_archives'; archives.mkdir(exist_ok=True)
                    out.rename(archives/(key.replace('/','_')+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')))
                command = [sys.executable,'-u',str(ROOT/'run.py'),'--threads',str(args.threads),
                    'train','--config',str(config_path),'--data',str(args.data.resolve()),
                    '--output',str(out),'--device',args.device]
                if (out/'last.pt').exists():
                    if not args.resume:
                        raise ValueError('Existing checkpoint requires --resume')
                    command.append('--resume')
                status('running', current=key)
                with (root/'logs'/(key.replace('/','_')+'.log')).open('a') as log:
                    worker = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    status('running', current=key, training_pid=worker.pid)
                    code = worker.wait(); worker = None
                if code or not completed(out, c, identity, env):
                    raise RuntimeError('Worker failed; inspect logs/' + key.replace('/','_')+'.log')
            done.append(key)
            status('running', current=key)
            print(json.dumps(dict(completed=key)), flush=True)
        from summarize import summarize
        summarize(root)
        status('completed')
    except (Exception, KeyboardInterrupt) as exc:
        if worker is not None and worker.poll() is None:
            os.killpg(worker.pid, signal.SIGTERM)
            worker.wait()
        status('interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
