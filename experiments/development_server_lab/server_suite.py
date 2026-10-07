#!/usr/bin/env python3
"""One-GPU staged server experiments, matching seeds and automatic comparisons."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from komap_phase.config import ROOT
from komap_phase.data import manifest
from komap_phase.runtime import code_hash,device_for,write_json
from make_configs import generate

ARCHITECTURES=['B','R01','R02','R03','A01_CNX','A02_OS16','A03_R50']


def selection(stage,requested):
    if stage=='validation':
        if requested and requested!=['B','D03']: raise ValueError('Validation stage requires B/D03')
        return ['B','D03']
    if stage=='architectures':
        if requested and any(v.endswith('_D03') or v=='D03' for v in requested):
            raise ValueError('Use combinations stage for contact-loss combinations')
        return list(dict.fromkeys(['B']+(requested or ARCHITECTURES)))
    if not requested or any(not v.endswith('_D03') for v in requested):
        raise ValueError('Combinations require explicit --ids, e.g. R02_D03')
    parents=[v.removesuffix('_D03') for v in requested]
    return list(dict.fromkeys(['B','D03']+parents+requested))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('validation','architectures','combinations'),default='validation')
    p.add_argument('--ids',nargs='+',choices=[v.stem for v in (ROOT/'configs').glob('*.json')])
    p.add_argument('--seeds',type=int,nargs='+')
    p.add_argument('--epochs',type=int,default=150)
    p.add_argument('--data',type=Path,default=ROOT/'Data')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cuda','cpu','mps'),default='cuda')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    try: ids=selection(a.stage,a.ids)
    except ValueError as exc: p.error(str(exc))
    seeds=a.seeds or ([43,44] if a.stage=='validation' else [42])
    if len(set(seeds))!=len(seeds) or any(v<0 for v in seeds) or a.epochs<1 or a.threads<1:
        p.error('Invalid seed list/epochs/threads')
    device_for(a.device)
    output=a.output.resolve()
    identity={'stage':a.stage,'ids':ids,'seeds':seeds,'epochs':a.epochs,'device':a.device,
              'data':str(a.data.resolve()),'manifest':manifest(a.data),'code_sha256':code_hash()}
    if output.exists() and any(output.iterdir()) and not a.resume:
        p.error('Choose a new output folder or --resume')
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.plan.lock').open('a')
    try: fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: p.error('Another plan uses this output folder')
    planfile=output/'plan.json'
    if planfile.exists():
        if json.loads(planfile.read_text())!=identity: p.error('Resume plan/code/data differs')
    else: write_json(planfile,identity)
    process=None
    completed=[]
    def stop(_sig,_frame): raise KeyboardInterrupt('Stop current trial and cancel queue')
    signal.signal(signal.SIGTERM,stop)
    try:
        for seed in seeds:
            configs=generate(output/'configs'/f'seed{seed}',seed,a.epochs)
            trials=output/f'seed{seed}'
            command=[sys.executable,'-u',str(ROOT/'run_suite.py'),'--data',str(a.data.resolve()),
                     '--configs',str(configs),'--output',str(trials),'--ids',*ids,
                     '--device',a.device,'--threads',str(a.threads)]
            if a.resume: command.append('--resume')
            print(json.dumps({'seed':seed,'ids':ids,'output':str(trials)}),flush=True)
            process=subprocess.Popen(command,cwd=ROOT)
            write_json(output/'status.json',{'state':'running','supervisor_pid':os.getpid(),
                       'child_pid':process.pid,'seed':seed,'ids':ids,'completed_seeds':completed})
            result=process.wait();process=None
            if result: raise RuntimeError(f'Seed{seed} failed with exit{result}; remaining queue cancelled')
            comparison=output/'comparisons'/f'seed{seed}'
            if not (comparison/'comparison.json').exists():
                subprocess.run([sys.executable,str(ROOT/'run.py'),'compare','--runs',
                                *[str(trials/name) for name in ids],'--output',str(comparison)],cwd=ROOT,check=True)
            completed.append(seed)
        subprocess.run([sys.executable,str(ROOT/'summarize_seeds.py'),'--plan',str(output)],cwd=ROOT,check=True)
        write_json(output/'status.json',{'state':'completed','completed_seeds':completed,'ids':ids})
    except KeyboardInterrupt:
        if process and process.poll() is None:
            process.terminate();process.wait()
        write_json(output/'status.json',{'state':'interrupted','completed_seeds':completed,'remaining_queue_cancelled':True})
        raise SystemExit(130)
    except Exception as exc:
        if process and process.poll() is None: process.terminate();process.wait()
        write_json(output/'status.json',{'state':'failed','error':str(exc),'completed_seeds':completed,'remaining_queue_cancelled':True})
        raise


if __name__=='__main__': main()
