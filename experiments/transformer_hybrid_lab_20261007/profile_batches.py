#!/usr/bin/env python3
"""Profile batch4/8/16 in separate CUDA workers without starting experiment training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from komap_phase.config import ROOT
from komap_phase.runtime import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batches',type=int,nargs='+',default=[4,8,16],choices=(4,8,16))
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    reports=ROOT/'checks/batch_profiles';reports.mkdir(parents=True,exist_ok=True)
    rows=[]
    for batch in args.batches:
        path=reports/f'batch{batch}.json'
        log=reports/f'batch{batch}.log'
        with log.open('w') as stream:
            code=subprocess.call([sys.executable,'-u',str(ROOT/'preflight.py'),'--device','cuda',
                '--size','448','--batch-size',str(batch),'--threads',str(args.threads),'--report',str(path)],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        result=json.loads(path.read_text()) if path.exists() else {}
        passed=code==0 and result.get('passed',False)
        row=dict(batch=batch,passed=passed,report=str(path.relative_to(ROOT)),log=str(log.relative_to(ROOT)),
                 checks=result.get('checks',[]))
        rows.append(row)
        print(json.dumps(row),flush=True)
    successful=[r for r in rows if r['passed']]
    result=dict(profiles=rows,largest_passing_batch=max((r['batch'] for r in successful),default=None),
        note='Memory fit is not a model-quality result. Compare B and hybrids at the same chosen batch; retain separate output folders.')
    write_json(reports/'summary.json',result)
    print(json.dumps(result),flush=True)
    if not successful:
        raise SystemExit('No batch passed; inspect profile logs')


if __name__=='__main__':
    main()
