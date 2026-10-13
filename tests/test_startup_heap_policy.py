"""Real cyclic-lifetime checks in owned children,not the suite's GC heap."""
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]


class StartupHeapTests(unittest.TestCase):
    def child(self,body):
        result=subprocess.run([sys.executable,'-c',
            'import gc,weakref\ngc.unfreeze();gc.collect();baseline=gc.get_freeze_count()\nfrom tools.startup_heap_policy import StartupHeapPolicy\n'+body],
            cwd=ROOT,capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_dynamic_cycles_collect_and_startup_cycles_release_on_close(self):
        self.child('''
class Node:pass
thresholds=gc.get_threshold();old=Node();old.link=old;old_ref=weakref.ref(old)
policy=StartupHeapPolicy('freeze',permanent_baseline=baseline)
try:
    policy.start();assert gc.get_freeze_count()>baseline
    del old;gc.collect();assert old_ref() is not None
    new=Node();new.link=new;new_ref=weakref.ref(new);del new;gc.collect()
    assert new_ref() is None and gc.isenabled() and gc.get_threshold()==thresholds
    try:StartupHeapPolicy('collect',permanent_baseline=baseline).start()
    except RuntimeError:pass
    else:raise AssertionError('Concurrent heap owner accepted')
finally:policy.close()
assert old_ref() is None and gc.get_freeze_count()==baseline and policy.report()['policy_restored']
policy.close()
''')

    def test_collect_only_and_foreign_freeze_are_not_overridden(self):
        self.child('''
policy=StartupHeapPolicy('collect',permanent_baseline=baseline)
try:policy.start();assert gc.get_freeze_count()==baseline
finally:policy.close()
assert policy.report()['policy_restored']
gc.collect();gc.freeze()
try:
    try:StartupHeapPolicy('freeze',permanent_baseline=baseline).start()
    except RuntimeError:pass
    else:raise AssertionError('Foreign heap accepted')
    assert gc.get_freeze_count()>baseline
finally:gc.unfreeze();gc.collect()
''')
