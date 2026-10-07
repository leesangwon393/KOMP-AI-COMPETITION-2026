"""Resume the existing RRCU suite, then run thirteen matched follow-ups."""
import argparse
import copy
import fcntl
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from komap_phase.config import ROOT, load_config
from komap_phase.data import manifest
from komap_phase.runtime import code_hash, device_for, write_json
from make_configs import generate

SCREEN = ['B', 'D03', 'X01_FREQ', 'X02_CONN', 'X03_PIXEL', 'L01_DICE35', 'L02_LOVASZ35', 'O01_LR2', 'O02_LR4']


def plan():
    return [{'seed':43,'ids':['B','D03']}, {'seed':44,'ids':['B','D03']}, {'seed':42,'ids':SCREEN}]


def differences(left, right, prefix=''):
    result = {}
    for key in sorted(set(left)|set(right)):
        name = f'{prefix}.{key}' if prefix else key
        if key not in left or key not in right:
            raise ValueError(f'Configuration schema differs at {name}')
        if isinstance(left[key], dict):
            result.update(differences(left[key], right[key], name))
        elif left[key] != right[key]:
            result[name] = [left[key], right[key]]
    return result


def validate_contrasts():
    baseline = load_config(ROOT/'configs/B.json')
    allowed = {'D03':{'aux.contact_weight'}, 'X01_FREQ':{'model.skip_fusion'},
               'X02_CONN':{'model.connectivity','aux.connectivity_weight'},
               'X03_PIXEL':{'model.upsampling'},
               'L01_DICE35':{'loss.dice_weight','loss.lovasz_weight'},
               'L02_LOVASZ35':{'loss.dice_weight','loss.lovasz_weight'},
               'O01_LR2':{'train.lr'}, 'O02_LR4':{'train.lr'}}
    result = {}
    for name in SCREEN[1:]:
        delta = differences(baseline, load_config(ROOT/f'configs/{name}.json'))
        delta.pop('id'); delta.pop('description')
        if set(delta) != allowed[name]:
            raise ValueError(f'Unexpected changes in {name}: {delta}')
        result[name] = delta
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--prerequisite', type=Path)
    parser.add_argument('--device', choices=('mps','cuda','cpu'), default='mps')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    contrasts = validate_contrasts()
    if args.plan_only:
        print(json.dumps({'groups':plan(),'new_runs':13,'epochs_each':150,'contrasts':contrasts},indent=2))
        return
    device_for(args.device)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error('Output exists; use a new directory or --resume')
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'.pipeline.lock').open('a')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('This pipeline already has a supervisor')
    identity = {'groups':plan(),'epochs_each':150,'device':args.device,'data':str(args.data.resolve()),
                'manifest':manifest(args.data),'code_sha256':code_hash(),'contrasts':contrasts,
                'prerequisite':str(args.prerequisite.resolve()) if args.prerequisite else None}
    saved = output/'plan.json'
    if saved.exists() and json.loads(saved.read_text()) != identity:
        parser.error('Code/data/plan changed; resume rejected')
    write_json(saved, identity)
    child = None
    completed = []

    def stop(_signal, _frame):
        raise KeyboardInterrupt('Cancel queue; keep completed-epoch checkpoints')

    signal.signal(signal.SIGTERM, stop)

    def run(command, cwd, stage):
        nonlocal child
        with (output/f'{stage}.log').open('a') as log:
            child = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
            write_json(output/'status.json', {'state':'running','stage':stage,'supervisor_pid':os.getpid(),
                                             'child_pid':child.pid,'completed_seeds':completed})
            code = child.wait()
        child = None
        if code:
            raise RuntimeError(f'{stage} exited {code}; queue cancelled')

    try:
        if args.prerequisite:
            parent = args.prerequisite.resolve()
            trials = parent/'trials'
            status = json.loads((trials/'suite_status.json').read_text())
            if status['state'] != 'completed':
                # run_suite's flock refuses a duplicate live supervisor. Its resume
                # guard validates each completed and partial run, code/data/env/RNG.
                previous_launch = json.loads((parent/'launch.json').read_text())
                if previous_launch['device'] != args.device:
                    raise ValueError('Prerequisite device mismatch')
                snapshot = parent/'source_snapshot'
                run([sys.executable,'-u',str(snapshot/'run_suite.py'),'--data',str(args.data.resolve()),
                     '--output',str(trials),'--ids','R01','R02','R03','B','--device',args.device,'--resume'],
                    snapshot,'prerequisite_rrcu')
            status = json.loads((trials/'suite_status.json').read_text())
            if status.get('completed') != ['R01','R02','R03','B'] or status['state'] != 'completed':
                raise ValueError('Prerequisite incomplete; do not start follow-ups')
            comparison = output/'rrcu_comparison'
            if not (comparison/'comparison.json').exists():
                snapshot = parent/'source_snapshot'
                run([sys.executable,str(snapshot/'run.py'),'compare','--runs',
                     *[str(trials/name) for name in ('B','R01','R02','R03')], '--output',str(comparison)],
                    snapshot,'compare_rrcu')
        for group in plan():
            seed, ids = group['seed'], group['ids']
            configs = generate(output/'configs'/f'seed{seed}', seed, 150)
            command = [sys.executable,'-u',str(ROOT/'run_suite.py'),'--data',str(args.data.resolve()),
                       '--configs',str(configs),'--output',str(output/f'seed{seed}'),
                       '--ids',*ids,'--device',args.device]
            if args.resume: command.append('--resume')
            run(command, ROOT, f'seed{seed}')
            comparison = output/'comparisons'/f'seed{seed}'
            if not (comparison/'comparison.json').exists():
                run([sys.executable,str(ROOT/'run.py'),'compare','--runs',
                     *[str(output/f'seed{seed}'/name) for name in ids],'--output',str(comparison)],
                    ROOT,f'compare_seed{seed}')
            completed.append(seed)
        # Reuse the three B/D03 pairs in a compact seed-summary plan.
        summary_plan = output/'d03_reproduction'
        summary_plan.mkdir(exist_ok=True)
        write_json(summary_plan/'plan.json', {'ids':['B','D03'],'seeds':[42,43,44],'epochs':150})
        (summary_plan/'comparisons').mkdir(exist_ok=True)
        for seed in (42,43,44):
            for target, source in ((summary_plan/f'seed{seed}',output/f'seed{seed}'),
                                   (summary_plan/'comparisons'/f'seed{seed}',output/'comparisons'/f'seed{seed}')):
                if not target.exists(): target.symlink_to(source, target_is_directory=True)
        run([sys.executable,str(ROOT/'summarize_seeds.py'),'--plan',str(summary_plan)], ROOT,'summarize_d03')
        write_json(output/'status.json', {'state':'completed','completed_seeds':completed,'new_runs':13})
    except (Exception, KeyboardInterrupt) as exc:
        if child and child.poll() is None:
            child.terminate(); child.wait()
        write_json(output/'status.json', {'state':'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                   'error':str(exc),'completed_seeds':completed,'remaining_queue_cancelled':True})
        if isinstance(exc, KeyboardInterrupt): raise SystemExit(130)
        raise


if __name__ == '__main__':
    main()
