#!/usr/bin/env python3
"""Run matched experiments sequentially on one GPU; stop the queue on failure/signal."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import torch

from komap_phase.config import ROOT, load_config, protocol
from komap_phase.runtime import write_json, resume_guard, device_for
from komap_phase.data import manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'Data')
    p.add_argument('--configs',type=Path,default=ROOT/'configs')
    p.add_argument('--output',type=Path,required=True)
    available=[path.stem for path in sorted((ROOT/'configs').glob('*.json'))]
    p.add_argument('--ids',nargs='+',choices=available,default=['B','D03'])
    p.add_argument('--device',choices=('cpu','cuda','mps'),default='cuda')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    a = p.parse_args()
    if len(set(a.ids))!=len(a.ids):
        p.error('Duplicate experiment IDs')
    configs = [load_config(a.configs/f'{name}.json') for name in a.ids]
    if any(protocol(c)!=protocol(configs[0]) for c in configs[1:]):
        p.error('Suite configs must share data-independent protocol, seed and epochs')
    output = a.output.resolve()
    if output.exists() and any(output.iterdir()) and not a.resume:
        p.error('Output already exists; choose a new folder or --resume')
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.suite.lock').open('a')
    try:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        p.error('Another supervisor is using this output directory')
    (output/'logs').mkdir(exist_ok=True)
    process = None
    completed = []
    def stop(_signum,_frame):
        raise KeyboardInterrupt('User signal: cancel remaining queue')
    signal.signal(signal.SIGTERM,stop)
    try:
        for name in a.ids:
            status_path=output/name/'status.json'
            if a.resume and status_path.exists():
                status=json.loads(status_path.read_text())
                if status.get('state')=='completed':
                    saved=load_config(output/name/'config.json')
                    expected=load_config(a.configs/f'{name}.json')
                    if saved!=expected:
                        raise ValueError(f'Completed run config differs: {name}')
                    if not (output/name/'summary.json').is_file():
                        raise ValueError(f'Completed run lacks summary: {name}')
                    state=torch.load(output/name/'best.pt',map_location='cpu',weights_only=True)
                    resume_guard(state,expected,manifest(a.data,patch=expected['train']['target']),
                                 'phase_segmentation',device_for(a.device))
                    del state
                    completed.append(name)
                    continue
            cmd = [sys.executable,'-u',str(ROOT/'run.py'),'--threads',str(a.threads),'train',
                '--config',str((a.configs/f'{name}.json').resolve()),'--data',str(a.data.resolve()),
                '--output',str(output/name),'--device',a.device]
            if a.resume and (output/name/'last.pt').exists():
                cmd.append('--resume')
            write_json(output/'suite_status.json',{'state':'running','supervisor_pid':os.getpid(),'current':name,'completed':completed,'queue':a.ids})
            print(json.dumps({'starting':name,'log':str(output/'logs'/f'{name}.log')}),flush=True)
            with (output/'logs'/f'{name}.log').open('a') as log:
                process = subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write_json(output/'suite_status.json',{'state':'running','supervisor_pid':os.getpid(),'current':name,'pid':process.pid,'completed':completed,'queue':a.ids})
                code = process.wait()
            process = None
            if code:
                write_json(output/'suite_status.json',{'state':'failed','current':name,'exit_code':code,'completed':completed,'remaining_queue_cancelled':True})
                raise SystemExit(code if code>0 else 1)
            completed.append(name)
        write_json(output/'suite_status.json',{'state':'completed','completed':completed})
    except KeyboardInterrupt:
        if process and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            process.wait()
        write_json(output/'suite_status.json',{'state':'interrupted','completed':completed,'remaining_queue_cancelled':True})
        raise SystemExit(130)


if __name__=='__main__':
    main()
