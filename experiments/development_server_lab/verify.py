"""Run meaningful package tests and record evidence; no real-data training."""
import argparse
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path

from komap_phase.config import ROOT
from komap_phase.runtime import code_hash,write_json


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full-size',action='store_true',help='Two CPU optimizer steps, context448/batch4, new encoder candidates')
    p.add_argument('--output',type=Path,default=ROOT/'verification'/'local_checks')
    a=p.parse_args()
    output=a.output;output.mkdir(parents=True,exist_ok=True)
    buffer=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result=unittest.TextTestRunner(stream=buffer,verbosity=2).run(suite)
    (output/'unit_tests.txt').write_text(buffer.getvalue())
    write_json(output/'unit_tests.json',{'passed':result.wasSuccessful(),'tests':result.testsRun,
               'failures':len(result.failures),'errors':len(result.errors),'code_sha256':code_hash()})
    print(buffer.getvalue())
    if not result.wasSuccessful(): raise SystemExit(1)
    if a.full_size:
        for name in ('A01_CNX_D03','A02_OS16_D03','A03_R50_D03'):
            subprocess.run([sys.executable,str(ROOT/'run.py'),'check','--config',str(ROOT/f'configs/{name}.json'),
                            '--device','cpu','--size','448','--batch-size','4',
                            '--report',str(output/f'fullsize_{name}.json')],cwd=ROOT,check=True)
