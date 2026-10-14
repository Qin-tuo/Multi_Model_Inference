"""Guard CUDA timeline accounting and real ROS-style descendant cleanup."""
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from tools.gpu_concurrency_probe import ROOT, analyze


class GpuConcurrencyProbeTests(unittest.TestCase):
    def test_clips_requests_unions_streams_and_maps_native_processes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'trace.sqlite'
            with sqlite3.connect(path) as connection:
                connection.executescript('''
                    CREATE TABLE PROCESSES(globalPid INTEGER,pid INTEGER);
                    INSERT INTO PROCESSES VALUES(100,1),(200,2),(300,3);
                    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,globalPid INTEGER);
                    INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
                        (0,9000000,0,100),(5000000,12000000,0,100),
                        (8000000,14000000,0,200),(20000000,22000000,0,300);
                    CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT);
                    INSERT INTO NVTX_EVENTS VALUES(6000000,16000000,'assistant:request:0:text');
                    CREATE TABLE DIAGNOSTIC_EVENT(severity INTEGER,text TEXT);
                ''')
            workers = {'processes': {name: {'members':[{'pid':pid}]}
                for name,pid in (('llama',1),('vision',2),('pose',3))}}
            measured = analyze(path, workers)
            phase = measured['phases'][0]
            self.assertEqual(measured['unmapped_kernel_count'],0)
            self.assertEqual(phase['components']['llama']['kernel_interval_union_ms'],6)
            self.assertEqual(phase['components']['llama']['kernel_count'],2)
            self.assertEqual(phase['components']['pose']['kernel_count'],0)
            self.assertEqual(phase['pairwise_kernel_interval_overlap_ms']['llama+vision'],4)
            self.assertEqual(phase['pairwise_kernel_interval_overlap_ms']['llama+pose'],0)
            with sqlite3.connect(path) as connection:
                connection.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES(3,'Graph correlation failed')")
            self.assertEqual(analyze(path,workers)['kernel_trace_errors'],{'Graph correlation failed':1})
            del workers['processes']['vision']
            self.assertEqual(analyze(path,workers)['unmapped_kernel_count'],1)

    def test_reaps_grandchild_when_launcher_exits_on_group_signal(self):
        # Isolate subreaper ownership from the test runner and other tests.
        code = '''
import json,subprocess,sys,time
from tools.gpu_concurrency_probe import adopt_children,terminate_group,group_members
adopt_children()
child_code = 'import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); print(p.pid,flush=True); time.sleep(60)'
p=subprocess.Popen([sys.executable,'-u','-c',child_code],stdout=subprocess.PIPE,text=True,start_new_session=True)
grandchild=int(p.stdout.readline())
assert grandchild in [r['pid'] for r in group_members(p.pid)]
try:
    terminate_group(p)
    assert not group_members(p.pid)
    print(json.dumps({'cleaned':True}))
finally:
    p.stdout.close()
'''
        result = subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=40)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout),{'cleaned':True})


if __name__ == '__main__':
    unittest.main()
