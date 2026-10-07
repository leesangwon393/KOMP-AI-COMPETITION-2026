"""Synthetic fixture suite fresh/resume; no KoMaP training or external downloads."""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from komap_phase.config import ROOT
from komap_phase.runtime import code_hash, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'verification/server_plan_smoke.json')
    a=p.parse_args()
    sys.path.insert(0,str(ROOT/'tests'))
    from test_resume import fixture
    with tempfile.TemporaryDirectory(prefix='u2_scse_supervisor_') as t:
        root=Path(t);fixture(root/'Data');plan=root/'plan'
        command=[sys.executable,str(ROOT/'server_suite.py'),'--ids','B','U2_SCSE_ALL_DEEP',
                 '--seeds','43','44','--epochs','1','--batch-size','2','--device','cpu',
                 '--smoke','--data',str(root/'Data'),'--output',str(plan)]
        fresh=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        if fresh.returncode:raise RuntimeError(fresh.stdout+'\n'+fresh.stderr)
        paths=sorted(plan.glob('seed*/*/best.pt'))
        checks={str(path.relative_to(plan)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        resumed=subprocess.run(command+['--resume'],cwd=ROOT,capture_output=True,text=True)
        if resumed.returncode:raise RuntimeError(resumed.stdout+'\n'+resumed.stderr)
        after={str(path.relative_to(plan)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        if len(paths)!=4 or after!=checks:raise ValueError('Completed run resume changed checkpoints')
        if resumed.stdout.count('skipped_completed')!=4:raise ValueError('Completed runs not skipped')
        board=json.loads((plan/'leaderboard.json').read_text())
        if len(board)!=4 or any(r['completed_seed_count']!=2 for r in board):raise ValueError('Incorrect paired-seed ranking')
        meta={'code_sha256':code_hash(),'passed':True,'device':'cpu','synthetic_fixture':True,
              'fresh_runs':4,'seeds':[43,44],'epochs_per_run':1,
              'resume_completed_runs_skipped':True,'best_checkpoint_sha256_before_after_equal':True,
              'training_on_komap_data':False,'stdout_fresh':fresh.stdout,'stdout_resume':resumed.stdout}
        a.output.parent.mkdir(parents=True,exist_ok=True);write_json(a.output,meta)
        print(json.dumps({k:v for k,v in meta.items() if not k.startswith('stdout')},indent=2))


if __name__=='__main__':main()
