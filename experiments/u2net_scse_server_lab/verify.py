"""Synthetic two-step optimizer checks in separate processes, plus dataset identity."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from komap_phase.config import ROOT
from komap_phase.data import manifest
from komap_phase.runtime import code_hash,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--ids',nargs='+');p.add_argument('--device',default='cpu',choices=('cpu','cuda','mps'))
    p.add_argument('--size',type=int,default=64);p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--data',type=Path);p.add_argument('--output',type=Path,default=ROOT/'verification/synthetic_cpu')
    p.add_argument('--resume',action='store_true');a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    ids=a.ids or [p.stem for p in sorted((ROOT/'configs').glob('*.json'))];passed=[];failed=[];sha=code_hash()
    for n in ids:
        report=a.output/(n+'.json');log=a.output/(n+'.log')
        if a.resume and report.exists():
            old=json.loads(report.read_text())
            if old['environment']['code_sha256']==sha and old['environment']['device']==a.device and old['checks'][0]['synthetic_size']>=a.size and old['checks'][0]['synthetic_batch_size']==a.batch_size:
                passed.append(n);continue
        with log.open('w') as f:
            result=subprocess.run([sys.executable,str(ROOT/'run.py'),'check','--config',str(ROOT/'configs'/(n+'.json')),
                '--device',a.device,'--size',str(a.size),'--batch-size',str(a.batch_size),'--report',str(report)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            failed.append({'id':n,'exit_code':result.returncode,'log':str(log)});print(json.dumps({'failed':n,'log':str(log)}),flush=True)
        else:passed.append(n);print(json.dumps({'passed':n,'count':len(passed),'total':len(ids)}),flush=True)
        write_json(a.output/'index.json',{'code_sha256':sha,'requested':ids,'passed':passed,'failures':failed,'cuda_validated':a.device=='cuda','training_results':False})
    if a.data:write_json(a.output/'dataset_manifest.json',manifest(a.data))
    return 1 if failed else 0

if __name__=='__main__':raise SystemExit(main())
