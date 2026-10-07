#!/usr/bin/env python3
"""Queue fixtures validate run counts, batch propagation, completion skip and pairing."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import run_suite
from komap_phase.config import ROOT
from komap_phase.runtime import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--report',type=Path,default=ROOT/'checks/queue.json')
    args=p.parse_args()
    original_popen=run_suite.subprocess.Popen
    original_argv=sys.argv[:]
    workers=[]
    fixture_identity=run_suite.manifest(args.data)
    fixture_env=run_suite.environment(run_suite.device_for("cpu"))
    class Worker:
        def __init__(self, command, **_kwargs):
            self.pid=999999
            config=json.loads(Path(command[command.index('--config')+1]).read_text())
            output=Path(command[command.index('--output')+1]);output.mkdir(parents=True,exist_ok=True)
            identity=fixture_identity
            env=fixture_env
            workers.append((config['train']['seed'],config['id'],config['train']['batch_size']))
            score=.5 if config['id']=='B' else .51
            metrics=dict(miou=score,filenames=[r['image'] for r in identity['valid']],
                per_image_miou=[score]*20,class_iou=[score]*4,
                eutectic=dict(precision=.8,recall=.8,thin_skeleton_recall=.7),
                contact=dict(error_rate=.2),phase_confusion=dict(al3ni_to_eutectic_rate=.1,eutectic_to_al3ni_rate=.1),
                seconds_per_image=.1)
            for name,value in [('config.json',config),('data_manifest.json',identity),('environment.json',env),
                ('history.json',[dict(epoch=i) for i in range(1,151)]),
                ('summary.json',dict(id=config['id'],best_epoch=150,single=metrics,d4=metrics,parameters_total=1,peak_cuda_memory_bytes=None)),
                ('status.json',dict(state='completed'))]:
                write_json(output/name,value)
            for name in ('best.pt','last.pt'):
                (output/name).write_bytes(b'QUEUE_FIXTURE_NOT_MODEL_WEIGHTS')
        def wait(self):return 0
        def poll(self):return 0
    run_suite.subprocess.Popen=Worker
    try:
        with tempfile.TemporaryDirectory(prefix='komap_queue_') as tmp:
            root=Path(tmp)/'queue'
            def invoke(phase='primary',batch=8,skip=False):
                sys.argv=['run_suite.py','--data',str(args.data.resolve()),'--output',str(root),'--device','cpu',
                    '--phase',phase,'--batch-size',str(batch),'--resume']+(['--skip-baseline'] if skip else [])
                run_suite.main()
            invoke();assert len(workers)==9 and all(w[2]==8 for w in workers)
            invoke();assert len(workers)==9,'Completed runs reran'
            invoke(skip=True);assert len(workers)==9,'Skip-baseline reran hybrids'
            invoke('followup');assert len(workers)==12
            invoke('all');assert len(workers)==12
            aggregates=json.loads((root/'reports/seed_aggregates.json').read_text())
            assert len(aggregates)==4 and all(r['n']==3 for r in aggregates)
            assert all(r['promotion_gate_passed'] for r in aggregates if r['id']!='B')
            try:
                invoke(batch=16)
            except ValueError:
                pass
            else:
                raise AssertionError('Changed batch silently accepted in same output')
    finally:
        run_suite.subprocess.Popen=original_popen;sys.argv=original_argv
    result=dict(passed=True,fixture_unique_runs=12,primary_runs=9,followup_runs=3,
        batch8_propagated=True,completed_and_skip_baseline_resume=True,batch_change_rejected=True,
        paired_summary_and_promotion=True,note='Queue fixtures only; no actual model scores or training')
    args.report.parent.mkdir(parents=True,exist_ok=True);write_json(args.report,result)
    print(json.dumps(result))


if __name__=='__main__':
    main()
