"""Create a self-contained source+dataset ZIP and verify CRC/SHA256; never train."""
import argparse
import ast
import hashlib
import json
import zipfile
from pathlib import Path

from komap_phase.config import ROOT, load_config
from komap_phase.data import manifest

PACKAGE = 'KoMaP_Phase_Discrimination_Server_20261003'


def digest_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def build(data,output):
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    configs={name:load_config(ROOT/'configs'/f'{name}.json') for name in ('B','D01','D02','D03','D04')}
    if any((c['train']['epochs'],c['train']['seed'],c['train']['patch'],c['train']['target'],c['model']['backbone'])
           !=(150,42,448,224,'resnet101') for c in configs.values()):
        raise ValueError('Release configs must use150/seed42/448/224/ResNet101')
    sources=[p for p in sorted(ROOT.iterdir()) if p.is_file() and p.suffix in ('.py','.md','.txt')]
    sources.append(ROOT/'.gitignore')
    for folder in ('komap_phase','configs','tests','verification'):
        sources.extend(p for p in sorted((ROOT/folder).rglob('*')) if p.is_file()
                       and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.md'))
    for p in sources:
        if p.suffix=='.py':
            ast.parse(p.read_text(),filename=str(p))
    checked=json.loads((ROOT/'verification/synthetic_cpu.json').read_text())
    if {v['id'] for v in checked['checks'] if v['status']=='passed'}!=set(configs):
        raise ValueError('Missing successful synthetic checks')
    unit=json.loads((ROOT/'verification/unit_tests.json').read_text())
    if not unit['passed']:
        raise ValueError('Unit tests did not pass')
    dataset_manifest=manifest(data,patch=224)
    counts={}
    datasets=[]
    for split in ('train','valid','test'):
        images=sorted((data/split/'images').glob('*.png'))
        masks=sorted((data/split/'masks').glob('*.png')) if split!='test' else []
        counts[split]={'images':len(images),'masks':len(masks)}
        datasets.extend(images+masks)
    if counts!={'train':{'images':70,'masks':70},'valid':{'images':20,'masks':20},'test':{'images':10,'masks':0}}:
        raise ValueError(f'Unexpected dataset counts: {counts}')
    entries={p.relative_to(ROOT).as_posix():p for p in sources}
    entries.update({'Data/'+p.relative_to(data).as_posix():p for p in datasets})
    checksums={name:digest_file(path) for name,path in entries.items()}
    metadata={'package':PACKAGE,'configs':list(configs),'dataset_counts':counts,
              'training_started':False,'train_valid_identity':dataset_manifest,
              'source_checksums':{k:v for k,v in checksums.items() if not k.startswith('Data/')}}
    manifest_bytes=(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n').encode()
    checksums['RELEASE_MANIFEST.json']=hashlib.sha256(manifest_bytes).hexdigest()
    sum_bytes=''.join(f'{sha}  {name}\n' for name,sha in sorted(checksums.items())).encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,path in sorted(entries.items()):
            z.write(path,f'{PACKAGE}/{name}')
        z.writestr(f'{PACKAGE}/RELEASE_MANIFEST.json',manifest_bytes)
        z.writestr(f'{PACKAGE}/SHA256SUMS.txt',sum_bytes)
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP CRC check failed')
        for name,expected in checksums.items():
            with z.open(f'{PACKAGE}/{name}') as f:
                h=hashlib.sha256()
                for chunk in iter(lambda:f.read(1024*1024),b''):
                    h.update(chunk)
            if h.hexdigest()!=expected:
                raise ValueError(f'ZIP checksum mismatch: {name}')
    print(json.dumps({'zip':str(output.resolve()),'bytes':output.stat().st_size,'files':len(entries)+2,
        'sha256':digest_file(output),'counts':counts,'training_started':False},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    build(a.data.resolve(),a.output.resolve())
