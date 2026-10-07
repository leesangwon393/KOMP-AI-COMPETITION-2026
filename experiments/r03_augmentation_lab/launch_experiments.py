"""Freeze source, launch R01/R02/R03/B sequentially, and report or stop only this run."""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from komap_phase.config import ROOT, load_config, protocol
from komap_phase.data import manifest
from komap_phase.runtime import code_hash, device_for, write_json


IDS=['R01','R02','R03','B']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('start','status','stop'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path)
    p.add_argument('--cache',type=Path)
    p.add_argument('--preflight',type=Path)
    p.add_argument('--device',choices=('mps','cuda'),default='mps')
    a=p.parse_args()
    output=a.output.resolve()
    if a.action=='status':
        for name in ['suite_status.json']+[f'{job}/status.json' for job in IDS]:
            path=output/'trials'/name
            if path.exists(): print(name,path.read_text())
        return
    if a.action=='stop':
        launch=json.loads((output/'launch.json').read_text())
        pid=launch['supervisor_pid']
        command=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True).stdout.strip()
        if str(output/'source_snapshot'/'run_suite.py') not in command or str(output) not in command:
            raise ValueError('Supervisor already exited or PID identity changed; no signal sent')
        os.kill(pid,signal.SIGTERM)
        for _ in range(40):
            result=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True)
            if not result.stdout.strip() or result.stdout.strip().startswith('Z'): break
            time.sleep(.5)
        else: raise RuntimeError('Stop sent but supervisor has not exited yet')
        print((output/'trials'/'suite_status.json').read_text())
        return
    if not a.data or not a.cache or not a.preflight:
        p.error('start requires data, cache and preflight')
    if output.exists(): raise ValueError('Output already exists; existing run is preserved')
    preflight=json.loads(a.preflight.read_text())
    if not preflight.get('passed') or preflight['device']!=a.device or preflight['environment']['code_sha256']!=code_hash():
        raise ValueError('A passing preflight from current code/device is required')
    if preflight['data_manifest']!=manifest(a.data): raise ValueError('Data changed after preflight')
    if {v['id'] for v in preflight['checks'] if v['passed']}!=set(IDS): raise ValueError('Missing trial preflight')
    configurations=[load_config(ROOT/f'configs/{name}.json') for name in IDS]
    if any(protocol(c)!=protocol(configurations[0]) for c in configurations[1:]): raise ValueError('Unmatched protocols')
    if any((c['train']['epochs'],c['train']['seed'],c['train']['patch'],c['train']['batch_size'],c['model']['backbone'])!=(150,42,448,4,'resnet101') for c in configurations):
        raise ValueError('Launch requires documented150/seed42/448/batch4/ResNet101 settings')
    device_for(a.device)
    output.mkdir(parents=True)
    snapshot=output/'source_snapshot'
    snapshot.mkdir()
    for folder in ('komap_phase','configs'):
        shutil.copytree(ROOT/folder,snapshot/folder,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('run.py','run_suite.py','launch_experiments.py','README_KO.md','requirements.txt','requirements-cuda.txt'):
        shutil.copy2(ROOT/name,snapshot/name)
    # run_suite requires a new directory; logs/launch are held outside its output.
    runs=output/'trials'
    env=os.environ.copy();env['TORCH_HOME']=str(a.cache.resolve())
    command=[sys.executable,'-u',str(snapshot/'run_suite.py'),'--data',str(a.data.resolve()),
             '--output',str(runs),'--ids',*IDS,'--device',a.device]
    with (output/'supervisor.log').open('a') as log:
        process=subprocess.Popen(command,cwd=snapshot,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    launch={'supervisor_pid':process.pid,'started_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'sequence':IDS,'epochs_each':150,'seed':42,'device':a.device,'trials':str(runs),
            'source_snapshot':str(snapshot),'preflight':preflight}
    write_json(output/'launch.json',launch)
    if sys.platform=='darwin' and shutil.which('caffeinate'):
        awake=subprocess.Popen(['caffeinate','-i','-w',str(process.pid)],stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        (output/'caffeinate.pid').write_text(str(awake.pid))
    print(json.dumps({k:launch[k] for k in ('supervisor_pid','sequence','epochs_each','device','trials')},indent=2))


if __name__=='__main__': main()
