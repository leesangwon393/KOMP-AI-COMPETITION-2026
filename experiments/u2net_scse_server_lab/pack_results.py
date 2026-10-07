"""Compact report export for review; no checkpoint/data copies or deletion."""
import argparse
import hashlib
import zipfile
from pathlib import Path

def pack(plan,output,masks=False):
    if output.exists():raise ValueError('Choose a new ZIP path')
    if not (plan/'plan.json').exists():raise ValueError('Missing plan.json')
    files=[]
    for path in sorted(plan.rglob('*')):
        if not path.is_file():continue
        if path.suffix in ('.json','.csv','.md','.log'):files.append(path)
        elif masks and path.suffix=='.png' and 'valid_masks' in path.parts:files.append(path)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        sums=[]
        for path in files:
            name=path.relative_to(plan).as_posix();content=path.read_bytes();z.writestr(plan.name+'/'+name,content)
            sums.append(hashlib.sha256(content).hexdigest()+'  '+name)
        z.writestr(plan.name+'/REPORT_SHA256SUMS.txt','\n'.join(sums)+'\n')
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC failed')
    print('Packed',len(files),'reports;',output,';',output.stat().st_size,'bytes')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--masks',action='store_true');a=p.parse_args();pack(a.plan.resolve(),a.output.resolve(),a.masks)
