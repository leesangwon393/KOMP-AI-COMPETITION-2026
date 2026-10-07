"""Package execution checks using synthetic data only; no KoMaP training."""
import argparse
import io
import subprocess
import sys
import unittest
from pathlib import Path

from komap_phase.config import ROOT
from komap_phase.runtime import code_hash,write_json


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'verification/local_checks')
    p.add_argument('--full-size',action='store_true')
    p.add_argument('--batch-size',type=int,choices=(2,4),default=4)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    log=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (a.output/'unit_tests.txt').write_text(log.getvalue())
    write_json(a.output/'unit_tests.json',{'passed':result.wasSuccessful(),'tests':result.testsRun,
               'failures':len(result.failures),'errors':len(result.errors),'code_sha256':code_hash()})
    print(log.getvalue(),flush=True)
    if not result.wasSuccessful(): raise SystemExit(1)
    if a.full_size:
        for name in ('U2','F3'):
            subprocess.run([sys.executable,str(ROOT/'run.py'),'check','--config',str(ROOT/f'configs/{name}.json'),
                '--device','cpu','--size','448','--batch-size',str(a.batch_size),
                '--report',str(a.output/f'fullsize_{name}.json')],cwd=ROOT,check=True)
