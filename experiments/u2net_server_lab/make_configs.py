"""Generate matching fresh-run configs; existing differing files are preserved."""
import argparse
import copy
import json
from pathlib import Path

from komap_phase.config import ROOT,load_config,validate
from komap_phase.runtime import write_json


def generate(output,seed,epochs,batch_size=4):
    if seed<0 or epochs<1 or batch_size not in (2,4): raise ValueError('Invalid seed/epoch/batch budget')
    output=Path(output)
    prepared=[]
    for path in sorted((ROOT/'configs').glob('*.json')):
        c=copy.deepcopy(load_config(path))
        c['train'].update(seed=seed,epochs=epochs,batch_size=batch_size)
        validate(c)
        target=output/path.name
        if target.exists() and json.loads(target.read_text())!=c:
            raise ValueError(f'Existing config differs: {target}')
        prepared.append((target,c))
    output.mkdir(parents=True,exist_ok=True)
    for target,c in prepared:
        if not target.exists(): write_json(target,c)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--epochs',type=int,default=150)
    p.add_argument('--batch-size',type=int,choices=(2,4),default=4)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(generate(a.output,a.seed,a.epochs,a.batch_size).resolve())
