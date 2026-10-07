"""Summarize matched-seed differences, keeping seed and image uncertainty distinct."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from komap_phase.config import load_config,protocol
from komap_phase.runtime import write_json


def summarize(plan):
    plan=Path(plan)
    identity=json.loads((plan/'plan.json').read_text())
    ids,seeds=identity['ids'],identity['seeds']
    if not ids or not seeds: raise ValueError('No selected reference/seeds')
    reference_id=ids[0]
    by_seed={}
    reference_protocol=reference_manifest=reference_environment=None
    for seed in seeds:
        rows=json.loads((plan/'comparisons'/f'seed{seed}'/'comparison.json').read_text())
        runs={}
        for name in ids:
            root=plan/f'seed{seed}'/name
            c=load_config(root/'config.json')
            if c['train']['seed']!=seed or c['train']['epochs']!=identity['epochs'] or c['id']!=name:
                raise ValueError('Seed/epoch/ID mismatch')
            pr=protocol(c);pr['train'].pop('seed')
            manifest=json.loads((root/'data_manifest.json').read_text())
            env=json.loads((root/'environment.json').read_text())
            environment={k:env.get(k) for k in ('code_sha256','device','torch','torchvision','cuda_build','gpu','dependencies')}
            if reference_protocol is None:
                reference_protocol,reference_manifest,reference_environment=pr,manifest,environment
            if pr!=reference_protocol or manifest!=reference_manifest or environment!=reference_environment:
                raise ValueError('Protocol/data/environment changed across seeds')
            selected={view:[r for r in rows if r['id']==name and r['view']==view] for view in ('single','d4')}
            if any(len(v)!=1 for v in selected.values()): raise ValueError('Missing/duplicate summary rows')
            runs[name]={view:entries[0] for view,entries in selected.items()}
        by_seed[seed]=runs
    summaries=[]
    for name in ids:
        for view in ('single','d4'):
            values=np.array([by_seed[s][name][view]['miou'] for s in seeds])
            deltas=np.array([by_seed[s][name][view]['miou']-by_seed[s][reference_id][view]['miou'] for s in seeds])*100
            summaries.append({'id':name,'view':view,'seed_count':len(seeds),'mean_miou':float(values.mean()),
                'seed_std_miou':float(values.std(ddof=1)) if len(seeds)>1 else None,
                'mean_delta_vs_reference_pp':float(deltas.mean()),'delta_seed_std_pp':float(deltas.std(ddof=1)) if len(seeds)>1 else None,
                'wins_vs_reference_seeds':int((deltas>0).sum()),'delta_vs_reference_by_seed_pp':dict(zip(map(str,seeds),deltas.tolist()))})
    contrasts=[]
    for name in ids:
        parents={'U2':'U0','F2':'F1'}
        if name not in parents or 'B_D03' not in ids or 'B' not in ids: continue
        parent=parents[name]
        if parent not in ids: continue
        for view in ('single','d4'):
            for seed in seeds:
                score=lambda key:by_seed[seed][key][view]['miou']
                contrasts.append({'id':name,'view':view,'seed':seed,
                    'delta_vs_structure_only_pp':100*(score(name)-score(parent)),
                    'delta_vs_D03_only_pp':100*(score(name)-score('B_D03')),
                    'interaction_pp':100*(score(name)-score(parent)-score('B_D03')+score('B'))})
    comparisons=[('U0','U1','scSE in RSU backbone model'),('U2','U0','contact loss in RSU backbone model'),
                 ('B_D03','B','contact loss in ResNet101 baseline'),('F1','F0','scSE in full U²-Net'),
                 ('F0_D03','F0','contact loss in full U²-Net without scSE'),
                 ('F2','F1','contact loss in full U²-Net with scSE'),('F3','F1','deep supervision'),
                 *[(name,'U0',name+' vs RSU baseline') for name in ('U3_R01','U3_R02','U3_R03')]]
    ablations=[]
    for candidate,reference,label in comparisons:
        if candidate not in ids or reference not in ids: continue
        for view in ('single','d4'):
            differences={str(seed):100*(by_seed[seed][candidate][view]['miou']-by_seed[seed][reference][view]['miou']) for seed in seeds}
            ablations.append({'candidate':candidate,'reference':reference,'label':label,'view':view,
                              'mean_delta_pp':float(np.mean(list(differences.values()))),'delta_by_seed_pp':differences})
    report={'reference_id':reference_id,'seeds':seeds,'summary':summaries,'ablation_contrasts':ablations,'combination_contrasts':contrasts,
        'note':'Validation results; descriptive seed mean/std, no claim of statistical superiority. Image bootstrap is recorded separately per seed.'}
    write_json(plan/'seed_summary.json',report)
    with (plan/'seed_summary.csv').open('w',newline='') as f:
        fields=[k for k in summaries[0] if k!='delta_vs_reference_by_seed_pp']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(summaries)
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    summarize(p.parse_args().plan)
