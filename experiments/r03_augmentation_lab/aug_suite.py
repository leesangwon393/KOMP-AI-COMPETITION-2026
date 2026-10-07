"""One process per GPU; independent fresh R03 runs, authoritative resume checkpoints."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from komap_phase.config import ROOT,load_config
from komap_phase.runtime import code_hash,write_json

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--session',choices=('gpu-l4-aug1','gpu-l4-aug2'),required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.queue.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    experiment=json.loads((ROOT/'experiment_plan.json').read_text())
    ids=experiment['assignments'][a.session]
    # Both seeds start with their assigned first condition; aug1 controls each seed.
    jobs=[(seed,name) for seed in experiment['seeds'] for name in ids]
    plan={'session':a.session,'ids':ids,'seeds':experiment['seeds'],'jobs':[[s,n] for s,n in jobs],
          'epochs':150,'batch_size':4,'model':'R03','code_sha256':code_hash()}
    if (output/'plan.json').exists() and json.loads((output/'plan.json').read_text())!=plan:
        raise ValueError('Queue assignment/code differs')
    write_json(output/'plan.json',plan)
    (output/'logs').mkdir(exist_ok=True)
    completed=[];process=None
    def interrupted(_signum,_frame):raise KeyboardInterrupt('Explicit interrupt; cancel remaining jobs')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    try:
        for seed,name in jobs:
            run=output/f'seed{seed}'/name
            previous=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
            if previous.get('state')=='completed' and (run/'summary.json').exists():
                completed.append(f'seed{seed}/{name}');continue
            if previous.get('state')=='interrupted':
                raise KeyboardInterrupt('Respect previously interrupted job')
            config=load_config(ROOT/'configs'/f'{name}.json');config['train']['seed']=seed
            configs=output/'generated_configs';configs.mkdir(exist_ok=True)
            config_path=configs/f'{name}_seed{seed}.json';write_json(config_path,config)
            cmd=[sys.executable,'-u',str(ROOT/'run.py'),'--threads',str(a.threads),'train',
                '--config',str(config_path),'--data',str(a.data.resolve()),'--output',str(run),'--device','cuda']
            if (run/'last.pt').exists():cmd.append('--resume')
            elif run.exists() and any(p.name!='.run.lock' for p in run.iterdir()):
                # Retain evidence from a failed first epoch before a truly fresh attempt.
                run.rename(run.with_name(run.name+'_failed_before_checkpoint_'+str(time.time_ns())))
            write_json(output/'status.json',{'state':'running','supervisor_pid':os.getpid(),
                'current':name,'seed':seed,'completed':completed,'total_runs':len(jobs)})
            with (output/'logs'/f'{name}_seed{seed}.log').open('a') as log:
                process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write_json(output/'status.json',{'state':'running','supervisor_pid':os.getpid(),
                    'child_pid':process.pid,'current':name,'seed':seed,'completed':completed,'total_runs':len(jobs)})
                code=process.wait()
            process=None
            if code:
                write_json(output/'status.json',{'state':'failed','current':name,'seed':seed,
                    'exit_code':code,'completed':completed,'total_runs':len(jobs)})
                return code if code>0 else 1
            completed.append(f'seed{seed}/{name}')
        write_json(output/'status.json',{'state':'completed','completed':completed,'total_runs':len(jobs)})
        return 0
    except KeyboardInterrupt:
        if process and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM);process.wait()
        write_json(output/'status.json',{'state':'interrupted','completed':completed,'total_runs':len(jobs)})
        return 130

if __name__=='__main__':sys.exit(main())
