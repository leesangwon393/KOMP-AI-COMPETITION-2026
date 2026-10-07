#!/usr/bin/env python3
"""Sequential, staged layer ablation with durable choices and strict resume guards."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from datetime import datetime, timezone

from komap_phase.config import ROOT, protocol
from komap_phase.data import manifest
from komap_phase.runtime import device_for, environment, write_json
from plan import reproduction, positions, mechanisms, combinations, select_positions, select_final, spatial_candidates


def read(path):
    return json.loads(path.read_text())


def guard_equal(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError('Persisted identity differs: ' + str(path))
    else:
        write_json(path, value)


def completed(directory, config, identity, env):
    status = read(directory/'status.json') if (directory/'status.json').exists() else {}
    if status.get('state') != 'completed':
        return False
    if read(directory/'config.json') != config or read(directory/'data_manifest.json') != identity:
        raise ValueError('Completed run config/data changed: ' + str(directory))
    previous = read(directory/'environment.json')
    for key in ('code_sha256','device','torch','torchvision','cuda_build','gpu','dependencies'):
        if previous.get(key) != env.get(key):
            raise ValueError('Completed run environment changed at '+key)
    history, summary = read(directory/'history.json'), read(directory/'summary.json')
    if len(history) != config['train']['epochs'] or history[-1]['epoch'] != config['train']['epochs']:
        raise ValueError('Incomplete history in completed run')
    if not all((directory/name).is_file() for name in ('best.pt','last.pt')) or 'd4' not in summary:
        raise ValueError('Completed run lacks checkpoints or D4 evaluation')
    return True


class Suite:
    def __init__(self, args):
        self.args, self.root = args, args.output.resolve()
        self.device = device_for(args.device)
        self.identity = manifest(args.data, patch=224 if args.profile=='full' else 32)
        if args.profile=='full' and {k:len(v) for k,v in self.identity.items()} != {'train':70,'valid':20}:
            raise ValueError('Full protocol requires Train70/Valid20; use the bundled Data')
        self.env = environment(self.device)
        self.rows, self.process = [], None

    def status(self, state, **extra):
        write_json(self.root/'suite_status.json', dict(state=state, supervisor_pid=os.getpid(),
                   completed=len(self.rows), completed_runs=self.rows,
                   max_unique_runs=20 if self.args.skip_spatial else 32, profile=self.args.profile, **extra))

    def run(self, config):
        seed, ident = config['train']['seed'], config['id']
        directory = self.root/f'seed{seed}'/ident
        key = f'seed{seed}/{ident}'
        config_path = self.root/'resolved_configs'/f'seed{seed}_{ident}.json'
        guard_equal(config_path, config)
        if completed(directory, config, self.identity, self.env):
            print(json.dumps(dict(skipping_completed=key)), flush=True)
        else:
            if directory.exists() and any(directory.iterdir()) and not (directory/'last.pt').exists():
                if not self.args.resume:
                    raise ValueError('Incomplete output without last.pt: '+key)
                # Preserve a failure before the first completed epoch, then fresh-start.
                if (directory/'config.json').exists() and read(directory/'config.json') != config:
                    raise ValueError('Pre-epoch output config changed')
                archive = self.root/'pre_epoch_archives'
                archive.mkdir(exist_ok=True)
                destination = archive/(key.replace('/','_')+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
                directory.rename(destination)
            command = [sys.executable, '-u', str(ROOT/'run.py'), '--threads', str(self.args.threads),
                'train', '--config', str(config_path), '--data', str(self.args.data.resolve()),
                '--output', str(directory), '--device', self.args.device]
            if (directory/'last.pt').exists():
                if not self.args.resume:
                    raise ValueError('Use --resume for an existing run')
                command.append('--resume')
            self.status('running', current=key)
            print(json.dumps(dict(starting=key, units=config['model']['units'])), flush=True)
            with (self.root/'logs'/(key.replace('/','_')+'.log')).open('a') as log:
                self.process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                                start_new_session=True)
                self.status('running', current=key, training_pid=self.process.pid)
                code = self.process.wait()
                self.process = None
            if code:
                self.status('failed', current=key, exit_code=code, message='Queue stopped; inspect the run log')
                raise RuntimeError('Training failed: '+key)
            if not completed(directory, config, self.identity, self.env):
                raise RuntimeError('Worker returned without a verified completed result: '+key)
        if key not in self.rows:
            self.rows.append(key)
        return dict(config=config, d4=read(directory/'summary.json')['d4']['miou'])

    def choose(self, filename, value):
        guard_equal(self.root/filename, value)
        return value

    def execute(self):
        for c in reproduction(self.args.profile):
            self.run(c)
        self.status('stage_completed', stage='reproduce')
        if self.args.stop_after=='reproduce':
            return
        screened = [self.run(c) for c in positions(self.args.profile)]
        screened.append(dict(config=reproduction(self.args.profile)[1],
            d4=read(self.root/'seed42/C3/summary.json')['d4']['miou']))
        scores = {r['config']['id']:r['d4'] for r in screened}
        top2 = self.choose('position_selection.json', dict(selection_metric='seed42 D4 at Single-selected checkpoint',
                       scores=scores, top2=select_positions(scores)))['top2']
        self.status('stage_completed', stage='layers', top2=top2)
        if self.args.stop_after=='layers':
            return
        others = [self.run(c) for c in mechanisms(int(top2[0][1:]), self.args.profile)]
        others += [self.run(c) for c in combinations(top2, self.args.profile)]
        candidates = screened + others
        chosen = select_final(candidates)
        selection = self.choose('final_selection.json', dict(selection_metric='seed42 D4; B excluded',
            candidates=[dict(id=r['config']['id'],d4=r['d4']) for r in candidates],
            chosen=[r['config'] for r in chosen],
            note='Exploratory selection on the same Valid20; no independent Test claim'))
        for c in selection['chosen']:
            for seed in (43,44):
                extra = copy.deepcopy(c)
                extra['train']['seed'] = seed
                self.run(extra)
        if not self.args.skip_spatial:
            channel_reference=next(r['config'] for r in screened if r['config']['id']==top2[0])
            for seed in (43,44):
                extra=copy.deepcopy(channel_reference);extra['train']['seed']=seed;self.run(extra)
            spatial = [self.run(c) for c in spatial_candidates(int(top2[0][1:]),self.args.profile)]
            best_joint=select_final([r for r in spatial if r['config']['id'].startswith('CS_')])[0]
            method=best_joint['config']['id'].removeprefix('CS_')
            pair=[r['config'] for r in spatial if r['config']['id'] in ('S_'+method,'CS_'+method)]
            selected = self.choose('spatial_selection.json',dict(selection_metric='seed42 CS D4; verify matched S/CS pair',
                stage=int(top2[0][1:]), candidates=[dict(id=r['config']['id'],d4=r['d4']) for r in spatial],
                chosen=pair))
            for c in selected['chosen']:
                for seed in (43,44):
                    extra=copy.deepcopy(c);extra['train']['seed']=seed;self.run(extra)
        self.status('completed', final_selection=[c['id'] for c in selection['chosen']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'Data')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/layer_search')
    parser.add_argument('--device', choices=('cuda','cpu','mps'), default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--profile', choices=('full','smoke'), default='full')
    parser.add_argument('--stop-after', choices=('reproduce','layers','all'), default='all')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--skip-spatial', action='store_true', help='Only the original layer ablation (up to20 runs)')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()) and not args.resume:
        parser.error('Output is nonempty; use a new folder or --resume')
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'.suite.lock').open('a')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('Another supervisor owns this output directory')
    for subdir in ('logs','resolved_configs'):
        (root/subdir).mkdir(exist_ok=True)
    suite = Suite(args)
    identity = dict(profile=args.profile, seeds=[42,43,44], skip_spatial=args.skip_spatial, data=suite.identity,
                    environment={k:suite.env.get(k) for k in ('code_sha256','device','torch','torchvision','cuda_build','gpu','dependencies')},
                    protocol=protocol(reproduction(args.profile)[0]))
    guard_equal(root/'suite_identity.json', identity)
    def stop(_signum,_frame):
        raise KeyboardInterrupt('User signal; cancel remaining queue')
    signal.signal(signal.SIGTERM, stop)
    try:
        suite.execute()
        from summarize import summarize
        summarize(root)
    except KeyboardInterrupt:
        if suite.process and suite.process.poll() is None:
            os.killpg(suite.process.pid, signal.SIGTERM)
            suite.process.wait()
        suite.status('interrupted', message='Resume from the last completed epoch with --resume')
        raise SystemExit(130)
    except Exception as exc:
        suite.status('failed', error=str(exc))
        raise


if __name__=='__main__':
    main()
