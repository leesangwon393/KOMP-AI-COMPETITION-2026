"""Record executable contract tests with current release code identity."""
import io
import sys
import unittest
from komap_phase.config import ROOT
from komap_phase.runtime import code_hash, write_json


if __name__=='__main__':
    sys.path.insert(0,str(ROOT/'tests'))
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (ROOT/'verification').mkdir(exist_ok=True)
    (ROOT/'verification/unit_tests.txt').write_text(stream.getvalue())
    write_json(ROOT/'verification/unit_tests.json',{'code_sha256':code_hash(),'passed':result.wasSuccessful(),
               'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
               'device':'cpu','real_komap_training':False})
    print(stream.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)
