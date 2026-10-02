"""Run C13 then a matched B baseline in detached processes with frozen source."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def worker(output):
    launch = json.loads((output / 'launch.json').read_text())
    completed = []
    for job in launch['jobs']:
        write(output / 'pipeline_status.json', {'state': 'running', 'current': job['id'], 'completed': completed})
        with (output / f"{job['id']}.training.log").open('a') as log:
            process = subprocess.Popen(job['command'], cwd=launch['source_snapshot'],
                                       stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            (output / f"{job['id']}.pid").write_text(str(process.pid))
            code = process.wait()
        if code:
            write(output / 'pipeline_status.json', {'state': 'failed', 'current': job['id'],
                                                   'completed': completed, 'exit_code': code})
            return code
        completed.append(job['id'])
    write(output / 'pipeline_status.json', {'state': 'completed', 'completed': completed})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'status', 'worker'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--torch-cache', type=Path)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--device', choices=('mps', 'cuda'), default='mps')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == 'worker':
        raise SystemExit(worker(output))
    if args.action == 'status':
        for name in ('pipeline_status.json', 'C13/status.json', 'B/status.json'):
            path = output / name
            if path.exists():
                print(name, path.read_text(), flush=True)
        return
    if not args.data or not args.preflight or not args.torch_cache:
        parser.error('start requires --data, --preflight, --torch-cache')
    preflight = json.loads(args.preflight.read_text())
    if not preflight.get('passed') or preflight.get('device') != args.device:
        raise ValueError('A passing preflight for the requested device is required')
    if output.exists():
        raise ValueError('Choose a new output directory; existing results are preserved')
    from komap_lab.runtime import device_for
    device_for(args.device)
    root = Path(__file__).resolve().parent
    output.mkdir(parents=True)
    snapshot = output / 'source_snapshot'
    snapshot.mkdir()
    for folder in ('komap_lab', 'configs'):
        shutil.copytree(root / folder, snapshot / folder, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('run.py', 'launch_rrcu.py', 'requirements.txt', 'NOTICE.md'):
        shutil.copy2(root / name, snapshot / name)
    jobs = []
    for name in ('C13', 'B'):
        command = [sys.executable, '-u', str(snapshot / 'run.py'), '--threads', '4', 'train',
                   '--config', str(snapshot / 'configs' / f'{name}.json'),
                   '--data', str(args.data.resolve()), '--output', str(output / name), '--device', args.device]
        jobs.append({'id': name, 'command': command})
    write(output / 'launch.json', {'source_snapshot': str(snapshot), 'jobs': jobs,
                                 'device': args.device, 'preflight': preflight})
    write(output / 'pipeline_status.json', {'state': 'queued', 'completed': []})
    env = os.environ.copy()
    env['TORCH_HOME'] = str(args.torch_cache.resolve())
    with (output / 'supervisor.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(snapshot / 'launch_rrcu.py'), 'worker',
                                    '--output', str(output)], cwd=snapshot, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    (output / 'supervisor.pid').write_text(str(process.pid))
    if sys.platform == 'darwin' and shutil.which('caffeinate'):
        keep_awake = subprocess.Popen(['caffeinate', '-i', '-w', str(process.pid)],
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, start_new_session=True)
        (output / 'caffeinate.pid').write_text(str(keep_awake.pid))
    print(json.dumps({'supervisor_pid': process.pid, 'output': str(output),
                      'sequence': ['C13', 'B'], 'epochs_each': 150, 'device': args.device}), flush=True)


if __name__ == '__main__':
    main()
