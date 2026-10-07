"""Build source/data ZIPs only after recorded executable checks pass."""
import argparse
import ast
import hashlib
import json
import zipfile
from pathlib import Path
from komap_phase.config import ROOT,load_config
from komap_phase.data import manifest
from komap_phase.runtime import code_hash
from server_suite import PROFILES

PACKAGE='KoMaP_Paper_Architecture_Server_20261004'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def report(path):
    r=json.loads(path.read_text());sha=r.get('code_sha256',r.get('environment',{}).get('code_sha256'))
    if sha!=code_hash():raise ValueError('Verification code changed: '+str(path))
    return r

def write_zip(output,entries,extras=None):
    checks={name:digest(path) for name,path in entries.items()};payloads=extras or {}
    checks.update({name:hashlib.sha256(content).hexdigest() for name,content in payloads.items()})
    sums=''.join(f'{sha}  {name}\n' for name,sha in sorted(checks.items())).encode()
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,path in sorted(entries.items()):z.write(path,PACKAGE+'/'+name)
        for name,content in payloads.items():z.writestr(PACKAGE+'/'+name,content)
        test=bool(entries) and all(n.startswith('Data/test') for n in entries)
        z.writestr(PACKAGE+('/TEST10_SHA256SUMS.txt' if test else '/SHA256SUMS.txt'),sums)
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC failed')
        for name,sha in checks.items():
            if hashlib.sha256(z.read(PACKAGE+'/'+name)).hexdigest()!=sha:raise ValueError('ZIP SHA mismatch: '+name)
    return checks

def build(data,output,test_output):
    if output.exists() or test_output.exists():raise ValueError('Release files already exist; choose fresh filenames')
    configs={p.stem:load_config(p) for p in sorted((ROOT/'configs').glob('*.json'))}
    if len(configs)!=127:raise ValueError('Expected127 unique configurations')
    if any((c['train']['epochs'],c['train']['seed'],c['train']['patch'],c['train']['target'],c['train']['batch_size'])!=(150,42,448,224,4) for c in configs.values()):raise ValueError('Release budget differs')
    index=report(ROOT/'verification/synthetic_cpu/index.json')
    if set(index['passed'])!=set(configs) or index['failures']:raise ValueError('Missing successful synthetic checks')
    for n in configs:
        r=report(ROOT/'verification/synthetic_cpu'/(n+'.json'));entry=r['checks'][0]
        if entry['id']!=n or entry['status']!='passed' or len(entry['steps'])!=2:raise ValueError('Invalid check: '+n)
    full=report(ROOT/'verification/fullsize_cpu/index.json')
    if full['failures'] or len(full['passed'])!=6:raise ValueError('Missing six full-size checks')
    for n in full['passed']:
        e=report(ROOT/'verification/fullsize_cpu'/(n+'.json'))['checks'][0]
        if (e['synthetic_size'],e['synthetic_batch_size'])!=(448,4):raise ValueError('Not full448/batch4')
    unit=report(ROOT/'verification/unit_tests.json');smoke=report(ROOT/'verification/server_plan_smoke.json')
    if not unit['passed'] or unit['tests']<7 or not smoke['passed'] or not smoke['resume_completed_runs_skipped']:raise ValueError('Unit/supervisor checks failed')
    weights=report(ROOT/'verification/micronet_weight_load.json')
    if not weights['passed']:raise ValueError('MicroNet load failed')
    provenance=json.loads((ROOT/'docs/U2NET_SOURCE_PROVENANCE.json').read_text())
    for item in provenance['files'].values():
        if digest(ROOT/item['path'])!=item['sha256']:raise ValueError('Upstream snapshot/license changed')
    identity=manifest(data);counts={};entries={};tests={}
    for split in ('train','valid','test'):
        images=sorted((data/split/'images').glob('*.png'));masks=sorted((data/split/'masks').glob('*.png'))
        counts[split]={'images':len(images),'masks':len(masks)}
        for path in images+masks:
            name='Data/'+path.relative_to(data).as_posix();(tests if split=='test' else entries)[name]=path
    if counts!={'train':{'images':70,'masks':70},'valid':{'images':20,'masks':20},'test':{'images':10,'masks':0}}:raise ValueError('Dataset split/count mismatch')
    for p in sorted(ROOT.rglob('*')):
        relative=p.relative_to(ROOT)
        if not p.is_file() or any(n in relative.parts for n in ('runs','Data','__pycache__','.venv')):continue
        if p.suffix not in {'.py','.json','.csv','.md','.txt','.sh'} and p.name!='.gitignore':continue
        if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
        entries[relative.as_posix()]=p
    output.parent.mkdir(parents=True,exist_ok=True);test_output.parent.mkdir(parents=True,exist_ok=True);write_zip(test_output,tests)
    meta={'package':PACKAGE,'date':'2026-10-04','configs':list(configs),'configuration_count':127,'profiles':PROFILES,
          'full_budget':{'runs':381,'seeds':[42,43,44],'epochs_per_run':150,'total_epochs':57150},
          'dataset_counts':counts,'dataset_manifest':identity,'main_zip_splits':['train','valid'],
          'test_zip':{'name':test_output.name,'bytes':test_output.stat().st_size,'sha256':digest(test_output),'required_for_training':False},
          'trained_checkpoints_included':False,'new_real_data_training_started':False,'code_sha256':code_hash(),
          'verification':{'device':'cpu','cuda_validated':False,'models_two_optimizer_steps':127,'unit_tests':unit['tests'],
                          'fullsize_448_batch4_models':full['passed'],'supervisor_fresh_and_resume':True,'micronet_imagenet_v1_strict_weight_loading':weights},
          'scope':'Native/library multiclass architectures and labelled derived modules; see COVERAGE_KO.md for full-paper exclusions'}
    checks=write_zip(output,entries,{'RELEASE_MANIFEST.json':(json.dumps(meta,ensure_ascii=False,indent=2)+'\n').encode()})
    if max(output.stat().st_size,test_output.stat().st_size)>104857600:raise ValueError('Release exceeds100MiB')
    print(json.dumps({'main_zip':str(output.resolve()),'bytes':output.stat().st_size,'files':len(checks)+1,'sha256':digest(output),
                     'test_zip':str(test_output.resolve()),'test_bytes':test_output.stat().st_size,'configs':127,'cuda_validated':False},ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--test-output',type=Path,required=True);a=p.parse_args()
    build(a.data.resolve(),a.output.resolve(),a.test_output.resolve())
