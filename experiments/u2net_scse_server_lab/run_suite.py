"""One GPU, no batch-size fallbacks; preserve every failure and continue candidates."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import torch
from komap_phase.config import ROOT,load_config,protocol
from komap_phase.data import manifest
from komap_phase.runtime import device_for,resume_guard,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--configs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--ids',nargs='+',required=True)
    p.add_argument('--device',choices=('cpu','cuda','mps'),default='cuda');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-on-error',action='store_true');a=p.parse_args()
    if len(set(a.ids))!=len(a.ids):p.error('Duplicate IDs')
    configs={n:load_config(a.configs/(n+'.json')) for n in a.ids}
    if any(protocol(c)!=protocol(configs[a.ids[0]]) for c in configs.values()):p.error('Incompatible training regime')
    output=a.output.resolve()
    if output.exists() and any(output.iterdir()) and not a.resume:p.error('Use new output or --resume')
    output.mkdir(parents=True,exist_ok=True);(output/'logs').mkdir(exist_ok=True)
    lock=(output/'.suite.lock').open('a')
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:p.error('Another suite owns output')
    completed=[];failures=[];process=None
    def stop(_sig,_frame):raise KeyboardInterrupt('Stop active run and queue')
    signal.signal(signal.SIGTERM,stop)
    try:
        for n in a.ids:
            destination=output/n;c=configs[n];status=destination/'status.json'
            if a.resume and status.exists() and json.loads(status.read_text()).get('state')=='completed':
                if load_config(destination/'config.json')!=c or not (destination/'summary.json').exists():raise ValueError('Completed output/config mismatch: '+n)
                state=torch.load(destination/'best.pt',map_location='cpu',weights_only=True)
                resume_guard(state,c,manifest(a.data,patch=c['train']['target']),'phase_segmentation',device_for(a.device));del state
                completed.append(n);print(json.dumps({'skipped_completed':n}),flush=True);continue
            if a.resume and destination.exists() and not (destination/'last.pt').exists():
                # Preserve failed initialization/epoch0 instead of deleting or silently skipping.
                if (destination/'config.json').exists() and load_config(destination/'config.json')!=c:raise ValueError('Failed run config differs')
                archive=output/'failed_attempts';archive.mkdir(exist_ok=True)
                destination.rename(archive/(n+'_'+str(time.time_ns())))
            cmd=[sys.executable,'-u',str(ROOT/'run.py'),'--threads',str(a.threads),'train','--config',str((a.configs/(n+'.json')).resolve()),'--data',str(a.data.resolve()),'--output',str(destination),'--device',a.device]
            if a.resume and (destination/'last.pt').exists():cmd.append('--resume')
            print(json.dumps({'starting':n,'log':str(output/'logs'/(n+'.log'))}),flush=True)
            with (output/'logs'/(n+'.log')).open('a') as log:
                process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write_json(output/'suite_status.json',{'state':'running','current':n,'pid':process.pid,'completed':completed,'failures':failures})
                code=process.wait();process=None
            if code:
                failures.append({'id':n,'exit_code':code,'log':str(output/'logs'/(n+'.log'))})
                destination.mkdir(exist_ok=True)
                if not status.exists():write_json(status,{'state':'failed','exit_code':code,'error':'Initialization or first-epoch failure; inspect log','log':str(output/'logs'/(n+'.log'))})
                write_json(output/'suite_status.json',{'state':'running_with_failures','completed':completed,'failures':failures})
                if n=='B' or a.stop_on_error:break
            else:completed.append(n)
        write_json(output/'suite_status.json',{'state':'completed_with_failures' if failures else 'completed','completed':completed,'failures':failures})
        return 1 if failures else 0
    except KeyboardInterrupt:
        if process and process.poll() is None:os.killpg(process.pid,signal.SIGTERM);process.wait()
        write_json(output/'suite_status.json',{'state':'interrupted','completed':completed,'failures':failures});return 130

if __name__=='__main__':raise SystemExit(main())
