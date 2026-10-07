"""Build a standalone attention ablation release only after executable checks."""
import argparse
import ast
import csv
import hashlib
import json
import zipfile
from pathlib import Path
from komap_phase.config import ROOT, load_config
from komap_phase.data import manifest
from komap_phase.runtime import code_hash
from make_configs import catalogue
from server_suite import PROFILES

PACKAGE='KoMaP_U2Net_scSE_Server_20261004'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def report(path):
    r=json.loads(path.read_text())
    if r.get('code_sha256',r.get('environment',{}).get('code_sha256'))!=code_hash():
        raise ValueError('Stale verification: '+str(path))
    return r


def build(data,output):
    if output.exists():raise ValueError('Choose a fresh ZIP filename')
    records=catalogue();configs={p.stem:load_config(p) for p in (ROOT/'configs').glob('*.json')}
    if len(configs)!=16 or set(configs)!={c['id'] for c in records}:raise ValueError('Expected 16 configurations')
    for c in records:
        if configs[c['id']]!=c:raise ValueError('Stored/catalogue config differs')
        if (c['train']['epochs'],c['train']['seed'],c['train']['patch'],c['train']['target'],c['train']['batch_size'])!=(150,42,448,224,4):
            raise ValueError('Release training budget differs')
    index=report(ROOT/'verification/synthetic_cpu/index.json')
    if set(index['passed'])!=set(configs) or index['failures']:raise ValueError('Missing synthetic checks')
    resources=[]
    for n in configs:
        check=report(ROOT/f'verification/synthetic_cpu/{n}.json')['checks'][0]
        if check['id']!=n or check['status']!='passed' or len(check['steps'])!=2:raise ValueError('Invalid check: '+n)
        parameters=check['parameters']
        resources.append({'id':n,'parameters':parameters,'parameter_M':parameters/1e6,
                          'fp32_parameter_adam_bytes_lower_bound':parameters*16,
                          'best_plus_last_checkpoint_bytes_approx':parameters*20,
                          'three_seed_checkpoint_GiB_approx':parameters*60/2**30,
                          'activation_memory_included':False})
    with (ROOT/'RESOURCE_ESTIMATES.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(resources[0]));writer.writeheader();writer.writerows(resources)
    full=report(ROOT/'verification/fullsize_cpu/index.json')
    if set(full['passed'])!={'U2_ECA_SCSE_DEEP','U2_ECA_CBAM_DEEP'} or full['failures']:raise ValueError('Missing full-size checks')
    for n in full['passed']:
        entry=report(ROOT/f'verification/fullsize_cpu/{n}.json')['checks'][0]
        if (entry['synthetic_size'],entry['synthetic_batch_size'])!=(448,4):raise ValueError('Expected 448/batch4')
    tests=report(ROOT/'verification/unit_tests.json');supervisor=report(ROOT/'verification/server_plan_smoke.json')
    if not tests['passed'] or tests['tests']<13:raise ValueError('Contract tests failed')
    if not supervisor['passed'] or not supervisor['resume_completed_runs_skipped']:raise ValueError('Supervisor checks failed')
    provenance=json.loads((ROOT/'docs/U2NET_SOURCE_PROVENANCE.json').read_text())
    for item in provenance['files'].values():
        if digest(ROOT/item['path'])!=item['sha256']:raise ValueError('Upstream/license changed')
    identity=manifest(data);counts={};entries={}
    for split,count in (('train',70),('valid',20)):
        images=sorted((data/split/'images').glob('*.png'));masks=sorted((data/split/'masks').glob('*.png'))
        counts[split]={'images':len(images),'masks':len(masks)}
        if len(images)!=count or len(masks)!=count:raise ValueError('Incorrect dataset split')
        for path in images+masks:entries['Data/'+path.relative_to(data).as_posix()]=path
    for path in sorted(ROOT.rglob('*')):
        relative=path.relative_to(ROOT)
        if not path.is_file() or any(p in relative.parts for p in ('Data','runs','checks','__pycache__','.venv')):continue
        if path.suffix not in ('.py','.json','.csv','.md','.txt','.sh','.log'):continue
        if path.suffix=='.py':ast.parse(path.read_text(),filename=str(path))
        entries[relative.as_posix()]=path
    meta={'package':PACKAGE,'date':'2026-10-04','configuration_count':16,'configs':list(configs),
          'profiles':PROFILES,'code_sha256':code_hash(),'dataset_counts':counts,'data_manifest':identity,
          'main_budget':{'seeds':[42,43,44],'runs':48,'epochs_per_run':150,'total_epochs':7200,'batch_size':4},
          'trained_weights_included':False,'real_komap_training_started':False,
          'verification':{'device':'cpu','cuda_validated':False,'two_optimizer_steps_models':16,
                          'fullsize_448_batch4_models':full['passed'],'unit_tests':tests['tests'],
                          'supervisor_fresh_and_completed_resume':True},
          'three_seed_checkpoint_GiB_approx':sum(r['three_seed_checkpoint_GiB_approx'] for r in resources),
          'scope':'Independent full U2NET/scSE attention ablations; exact alloy-paper reproduction is not claimed'}
    extras={'RELEASE_MANIFEST.json':(json.dumps(meta,ensure_ascii=False,indent=2)+'\n').encode()}
    sums={name:digest(path) for name,path in entries.items()}
    sums.update({name:hashlib.sha256(content).hexdigest() for name,content in extras.items()})
    payload=''.join(f'{sha}  {name}\n' for name,sha in sorted(sums.items())).encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,path in sorted(entries.items()):z.write(path,PACKAGE+'/'+name)
        for name,content in extras.items():z.writestr(PACKAGE+'/'+name,content)
        z.writestr(PACKAGE+'/SHA256SUMS.txt',payload)
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC failure')
        for name,sha in sums.items():
            if hashlib.sha256(z.read(PACKAGE+'/'+name)).hexdigest()!=sha:raise ValueError('ZIP SHA failure: '+name)
    print(json.dumps({'zip':str(output),'bytes':output.stat().st_size,'files':len(sums)+1,'sha256':digest(output),
                      'configs':16,'runs':48,'cuda_validated':False},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.data.resolve(),a.output.resolve())
