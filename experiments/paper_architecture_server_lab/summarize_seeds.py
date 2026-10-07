"""Partial results remain visible. Averages use only paired completed seeds."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from komap_phase.runtime import write_json

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
    print(json.dumps({'rows':len(rows),'missing_or_failed_runs':len(failures),'table':str(plan/'leaderboard.csv')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);summarize(p.parse_args().plan)
