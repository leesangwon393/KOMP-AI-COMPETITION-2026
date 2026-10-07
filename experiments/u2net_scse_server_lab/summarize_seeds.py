"""Partial results remain visible. Averages use only paired completed seeds."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from komap_phase.runtime import write_json

def summarize_u2_pairs(plan,identity):
    """Match each candidate to its same-seed/same-supervision U²-Net baseline."""
    rows=[];details=[];missing=[]
    def complete(path):
        return (path/'summary.json').exists() and (path/'status.json').exists() and json.loads((path/'status.json').read_text()).get('state')=='completed'
    for n in identity['ids']:
        if not n.startswith('U2_'):continue
        ref='U2_PLAIN_DEEP' if n.endswith('_DEEP') else 'U2_PLAIN'
        for view in ('single','d4'):
            matched=[]
            for seed in identity['seeds']:
                root=plan/('seed'+str(seed));candidate=root/n;baseline=root/ref
                if not complete(candidate) or not complete(baseline):
                    if view=='single':missing.append({'id':n,'reference_id':ref,'seed':seed,'reason':'Candidate or matched U2 baseline incomplete'})
                    continue
                c=json.loads((candidate/'config.json').read_text());b=json.loads((baseline/'config.json').read_text())
                for key in ('train','normalization','loss','evaluation','aux'):
                    if c[key]!=b[key]:raise ValueError(f'U2 pair protocol differs: {n}/{ref}/{seed}/{key}')
                for filename in ('data_manifest.json','environment.json'):
                    if json.loads((candidate/filename).read_text())!=json.loads((baseline/filename).read_text()):
                        raise ValueError(f'U2 pair identity differs: {n}/{ref}/{seed}/{filename}')
                result=json.loads((candidate/'summary.json').read_text());reference=json.loads((baseline/'summary.json').read_text())
                if view not in result or view not in reference:continue
                result=result[view];reference=reference[view]
                if result['filenames']!=reference['filenames']:raise ValueError('U2 paired image ordering differs')
                delta=np.asarray(result['per_image_miou'])-np.asarray(reference['per_image_miou'])
                boot=np.random.default_rng(20261004).choice(delta,size=(10000,len(delta)),replace=True).mean(axis=1)*100
                entry={'id':n,'reference_id':ref,'seed':seed,'view':view,
                       'miou_percent':float(result['miou']*100),'reference_miou_percent':float(reference['miou']*100),
                       'delta_miou_pp':float(delta.mean()*100),'image_bootstrap_delta_pp_95':np.quantile(boot,[.025,.975]).tolist(),
                       'delta_al3ni_iou_pp':float((result['class_iou'][1]-reference['class_iou'][1])*100),
                       'delta_eutectic_iou_pp':float((result['class_iou'][2]-reference['class_iou'][2])*100),
                       'delta_contact_error_pp':float((result['contact']['error_rate']-reference['contact']['error_rate'])*100),
                       'delta_thin_skeleton_recall_pp':float((result['eutectic']['thin_skeleton_recall']-reference['eutectic']['thin_skeleton_recall'])*100)}
                details.append(entry);matched.append(entry)
            if not matched:continue
            deltas=[r['delta_miou_pp'] for r in matched]
            rows.append({'id':n,'reference_id':ref,'view':view,'completed_seed_count':len(matched),
                         'planned_seed_count':len(identity['seeds']),'all_seeds_completed':len(matched)==len(identity['seeds']),
                         'seeds':','.join(str(r['seed']) for r in matched),
                         'mean_miou_percent':float(np.mean([r['miou_percent'] for r in matched])),
                         'mean_delta_vs_u2_pp':float(np.mean(deltas)),
                         'sd_paired_delta_pp':float(np.std(deltas,ddof=1)) if len(deltas)>1 else None,
                         'wins_vs_u2':sum(d>0 for d in deltas),'losses_vs_u2':sum(d<0 for d in deltas),
                         'mean_delta_al3ni_iou_pp':float(np.mean([r['delta_al3ni_iou_pp'] for r in matched])),
                         'mean_delta_eutectic_iou_pp':float(np.mean([r['delta_eutectic_iou_pp'] for r in matched])),
                         'mean_delta_contact_error_pp':float(np.mean([r['delta_contact_error_pp'] for r in matched])),
                         'mean_delta_thin_skeleton_recall_pp':float(np.mean([r['delta_thin_skeleton_recall_pp'] for r in matched]))})
    rows.sort(key=lambda r:(r['view'],not r['all_seeds_completed'],-r['mean_delta_vs_u2_pp']))
    if rows:
        with (plan/'u2_paired_leaderboard.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(plan/'u2_paired_leaderboard.json',rows)
    write_json(plan/'u2_paired_seed_details.json',details)
    write_json(plan/'u2_pair_missing.json',missing)
    text=['# U²-Net 기본형 대비 attention 효과','',
          '같은 seed, 같은 deep supervision 설정의 U2_PLAIN 또는 U2_PLAIN_DEEP와 비교한다.',
          'mIoU/상 IoU/얇은 구조 recall은 증가가 개선이고, 접촉부 오류는 감소가 개선이다.',
          '이미지 bootstrap 구간은 학습 seed 변동의 신뢰구간이 아니다. 누락 seed는 평균에 포함하지 않는다.',
          '', '|ID|view|기준|완료 seed|mIoU %|U2 대비 %p|승/패|접촉 오류 Δ%p|',
          '|---|---|---|---|---:|---:|---|---:|']
    for r in rows:
        text.append(f"|{r['id']}|{r['view']}|{r['reference_id']}|{r['completed_seed_count']}/{r['planned_seed_count']}|{r['mean_miou_percent']:.4f}|{r['mean_delta_vs_u2_pp']:+.4f}|{r['wins_vs_u2']}/{r['losses_vs_u2']}|{r['mean_delta_contact_error_pp']:+.4f}|")
    (plan/'U2_ATTENTION_RESULTS_KO.md').write_text('\n'.join(text)+'\n')
    return rows,details

def summarize(plan):
    identity=json.loads((plan/'plan.json').read_text());rows=[];failures=[]
    for n in identity['ids']:
        for view in ('single','d4'):
            scores=[];deltas=[];seeds=[];secondary=[]
            for seed in identity['seeds']:
                root=plan/('seed'+str(seed));path=root/n/'summary.json';base=root/'B/summary.json'
                status=root/n/'status.json'
                if not path.exists() or not status.exists() or json.loads(status.read_text()).get('state')!='completed':
                    if view=='single':failures.append({'id':n,'seed':seed,'status':json.loads(status.read_text()) if status.exists() else {'state':'not_started'},'log':str(root/'logs'/(n+'.log'))})
                    continue
                summary=json.loads(path.read_text());baseline=json.loads(base.read_text())
                if view not in summary or view not in baseline:continue
                c=json.loads((root/n/'config.json').read_text());score=summary[view];scores.append(100*score['miou'])
                deltas.append(100*(score['miou']-baseline[view]['miou']));seeds.append(seed)
                secondary.append(score)
            if not scores:continue
            rows.append({'id':n,'view':view,'completed_seed_count':len(scores),'planned_seed_count':len(identity['seeds']),
                'all_seeds_completed':len(scores)==len(identity['seeds']),'seeds':','.join(map(str,seeds)),
                'mean_miou_percent':float(np.mean(scores)),'sd_miou_pp':float(np.std(scores,ddof=1)) if len(scores)>1 else None,
                'mean_delta_vs_paired_B_pp':float(np.mean(deltas)),'wins_vs_B':sum(d>0 for d in deltas),'losses_vs_B':sum(d<0 for d in deltas),
                'eutectic_iou':float(np.mean([s['class_iou'][2] for s in secondary])),
                'contact_error_rate':float(np.mean([s['contact']['error_rate'] for s in secondary])),
                'group':c['paper']['group'],'implementation':c['paper']['implementation'],'source':c['paper']['source'],
                'architecture':c['model']['architecture'],'backbone':c['model']['backbone'],'initialization':c['model']['initialization']})
    rows.sort(key=lambda r:(r['view'],not r['all_seeds_completed'],-r['mean_miou_percent']))
    if rows:
        with (plan/'leaderboard.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(plan/'leaderboard.json',rows);write_json(plan/'failures.json',failures)
    text=['# 서버 실험 결과','',f"예정 {len(identity['ids'])}개 × {len(identity['seeds'])}seed. 누락/실패 {len(failures)}run.",
          '','로컬 Valid20 결과이며 공식 Test 점수가 아니다. single로 고른 동일 checkpoint를 D4로 평가한다.',
          'partial 평균은 완료된 paired seed에 한정한다. 이미지 bootstrap 구간은 학습 seed 변동 구간이 아니다.',
          '','|ID|view|완료 seed|mIoU %|B 대비 %p|구현 수준|','|---|---|---|---:|---:|---|']
    for r in rows:text.append(f"|{r['id']}|{r['view']}|{r['completed_seed_count']}/{r['planned_seed_count']}|{r['mean_miou_percent']:.4f}|{r['mean_delta_vs_paired_B_pp']:+.4f}|{r['implementation']}|")
    (plan/'RESULTS_KO.md').write_text('\n'.join(text)+'\n')
    summarize_u2_pairs(plan,identity)
    print(json.dumps({'rows':len(rows),'missing_or_failed_runs':len(failures),'table':str(plan/'leaderboard.csv')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);summarize(p.parse_args().plan)
