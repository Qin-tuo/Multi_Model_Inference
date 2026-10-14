"""Guard CUDA timeline accounting and real ROS-style descendant cleanup."""
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from tools.gpu_concurrency_probe import ROOT, analyze
from tools.gpu_trace_audit import launch_coverage


class GpuConcurrencyProbeTests(unittest.TestCase):
    def test_capture_calls_are_distinct_from_missing_device_events(self):
        with sqlite3.connect(':memory:') as connection:
            connection.row_factory=sqlite3.Row
            connection.executescript('''
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(globalPid INTEGER,correlationId INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE(globalPid INTEGER,correlationId INTEGER);
                INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(16777216,10);
                CREATE TABLE StringIds(id INTEGER,value TEXT);
                INSERT INTO StringIds VALUES(1,'cudaLaunchKernel_v7000'),
                    (2,'cudaStreamBeginCapture_v10000'),(3,'cudaStreamEndCapture_v10000');
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,
                    correlationId INTEGER,nameId INTEGER,returnValue INTEGER,callchainId INTEGER);
                INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
                    (3,4,16777219,10,1,0,NULL),
                    (10,11,16777219,100,2,0,NULL),
                    (12,13,16777219,20,1,0,NULL),
                    (20,21,16777219,101,3,0,NULL),
                    (30,31,16777219,30,1,0,NULL);
            ''')
            result=launch_coverage(connection,{1:'llama'})
            self.assertTrue(result['capture_ranges_balanced'])
            self.assertEqual(result['calls'][0]['matched_device_execution'],1)
            self.assertEqual(result['calls'][0]['captured_into_graph_without_immediate_execution'],1)
            self.assertEqual(result['calls'][0]['unexplained_missing_device_execution'],1)
            connection.execute('DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE nameId=3')
            self.assertFalse(launch_coverage(connection,{1:'llama'})['capture_ranges_balanced'])

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
            self.assertEqual(measured['unmapped_device_activity_count'],0)
            self.assertEqual(phase['components']['llama']['device_activity_interval_union_ms'],6)
            self.assertEqual(phase['components']['llama']['kernel_count'],2)
            self.assertEqual(phase['components']['pose']['kernel_count'],0)
            self.assertEqual(phase['pairwise_device_activity_interval_overlap_ms']['llama+vision'],4)
            self.assertEqual(phase['pairwise_device_activity_interval_overlap_ms']['llama+pose'],0)
            with sqlite3.connect(path) as connection:
                connection.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES(3,'Graph correlation failed')")
            self.assertEqual(analyze(path,workers)['kernel_trace_errors'],{'Graph correlation failed':1})
            del workers['processes']['vision']
            self.assertEqual(analyze(path,workers)['unmapped_device_activity_count'],1)
            with sqlite3.connect(path) as connection:
                connection.executescript('''
                    CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE(start INTEGER,end INTEGER,deviceId INTEGER,globalPid INTEGER);
                    INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES(11000000,18000000,0,100);
                ''')
            measured=analyze(path,workers)
            self.assertEqual(measured['kernel_count'],4)
            self.assertEqual(measured['graph_count'],1)
            # Graph lifetime extends the clipped union, without becoming a kernel.
            self.assertEqual(measured['phases'][0]['components']['llama']['device_activity_interval_union_ms'],10)
            self.assertEqual(measured['phases'][0]['components']['llama']['kernel_count'],2)

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
