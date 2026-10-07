"""Matched-seed architecture plans, fresh initial weights, resume and automatic tables."""
import argparse
import fcntl
import json
import signal
import subprocess
import sys
from pathlib import Path
from komap_phase.config import ROOT
from komap_phase.data import manifest
from komap_phase.runtime import code_hash,device_for,environment,write_json
from make_configs import catalogue,generate

RECORDS=catalogue()
ALL=[c['id'] for c in RECORDS]
CORE=['B','B_SCRATCH']+[c['id'] for c in RECORDS if c['model']['deep_supervision']]
PROFILES={'all':ALL, 'core':CORE,
          'scse':['B','B_SCRATCH','U2_PLAIN_DEEP','U2_SCSE_ALL_DEEP','U2_SCSE_LAST2_DEEP']}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',choices=tuple(PROFILES),default='all')
    p.add_argument('--ids',nargs='+',choices=ALL);p.add_argument('--seeds',type=int,nargs='+',default=[42])
    p.add_argument('--epochs',type=int,default=150);p.add_argument('--batch-size',type=int,choices=(2,4),default=4)
    p.add_argument('--data',type=Path,default=ROOT/'Data');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda','mps'),default='cuda');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-on-error',action='store_true')
    p.add_argument('--smoke',action='store_true',help='Synthetic fixture checks only: B and U2_SCSE_ALL_DEEP,64px,scratch,4 draws')
    a=p.parse_args()
    ids=list(dict.fromkeys(['B']+(a.ids or PROFILES[a.profile])))
    if a.smoke and (not a.ids or any(n not in ('B','U2_SCSE_ALL_DEEP') for n in ids) or a.batch_size!=2):p.error('Smoke requires --ids B U2_SCSE_ALL_DEEP --batch-size2')
    if len(set(a.seeds))!=len(a.seeds) or any(s<0 for s in a.seeds) or a.epochs<1 or a.threads<1:p.error('Invalid seeds/epochs/threads')
    device=device_for(a.device);output=a.output.resolve()
    identity={'profile':a.profile,'ids':ids,'seeds':a.seeds,'epochs':a.epochs,'batch_size':a.batch_size,
              'data_manifest':manifest(a.data,patch=32 if a.smoke else 224),'device':a.device,'code_sha256':code_hash(),'environment':environment(device),
              'stop_on_error':a.stop_on_error,'synthetic_smoke':a.smoke}
    if output.exists() and any(output.iterdir()) and not a.resume:p.error('Use new output or --resume')
    output.mkdir(parents=True,exist_ok=True);lock=(output/'.plan.lock').open('a')
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:p.error('Another supervisor owns output')
    if (output/'plan.json').exists():
        if json.loads((output/'plan.json').read_text())!=identity:p.error('Resume code/data/settings/environment changed')
    else:write_json(output/'plan.json',identity)
    process=None;completed=[];failed=[]
    def stop(_sig,_frame):raise KeyboardInterrupt('Stop active suite')
    signal.signal(signal.SIGTERM,stop)
    try:
        for seed in a.seeds:
            configs=generate(output/'configs'/('seed'+str(seed)),seed,a.epochs,a.batch_size,a.smoke);trials=output/('seed'+str(seed))
            cmd=[sys.executable,'-u',str(ROOT/'run_suite.py'),'--data',str(a.data.resolve()),'--configs',str(configs),
                 '--output',str(trials),'--ids',*ids,'--device',a.device,'--threads',str(a.threads)]
            if a.resume:cmd.append('--resume')
            if a.stop_on_error:cmd.append('--stop-on-error')
            print(json.dumps({'seed':seed,'experiments':len(ids),'epoch_budget':len(ids)*a.epochs,'output':str(trials)}),flush=True)
            process=subprocess.Popen(cmd,cwd=ROOT)
            write_json(output/'status.json',{'state':'running','current_seed':seed,'child_pid':process.pid,'completed_seeds':completed,'failed_seeds':failed})
            code=process.wait();process=None
            if code==130:raise KeyboardInterrupt('Child interrupted')
            if code:failed.append(seed)
            else:completed.append(seed)
            status=json.loads((trials/'suite_status.json').read_text());available=status.get('completed',[])
            if 'B' not in available:raise RuntimeError('Baseline failed; matched comparisons unavailable. Inspect logs and resume.')
            comparison=output/'comparisons'/('seed'+str(seed))
            # Always regenerate after retries; preserve older tables for audit.
            if comparison.exists():
                import time
                comparison.rename(comparison.with_name(comparison.name+'_'+str(time.time_ns())))
            subprocess.run([sys.executable,str(ROOT/'run.py'),'compare','--runs',*[str(trials/n) for n in available],
                            '--output',str(comparison)],cwd=ROOT,check=True)
            if code and a.stop_on_error:break
        subprocess.run([sys.executable,str(ROOT/'summarize_seeds.py'),'--plan',str(output)],cwd=ROOT,check=True)
        write_json(output/'status.json',{'state':'completed_with_failures' if failed else 'completed','completed_seeds':completed,'failed_seeds':failed})
        return 1 if failed else 0
    except KeyboardInterrupt:
        if process and process.poll() is None:process.terminate();process.wait()
        write_json(output/'status.json',{'state':'interrupted','completed_seeds':completed,'failed_seeds':failed});return 130
    except Exception as exc:
        if process and process.poll() is None:process.terminate();process.wait()
        write_json(output/'status.json',{'state':'failed','error':str(exc),'completed_seeds':completed,'failed_seeds':failed});raise

if __name__=='__main__':raise SystemExit(main())
