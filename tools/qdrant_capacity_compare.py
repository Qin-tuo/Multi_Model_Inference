"""Run a native Qdrant capacity comparison without changing the application store."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from deploy.memory.server import QdrantProcess
from deploy.memory.prepare import VERSION,COMMIT,BINARY
from tools.project import stamp,write_json,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tier',type=int,choices=[1000,10000,50000],required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--baseline-run',type=Path)
    p.add_argument('--baseline-report',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=16333)
    args=p.parse_args()
    args.run_dir.mkdir(parents=True,exist_ok=True)
    report_path=args.output.with_name(args.output.stem+'-capacity.json')
    env=os.environ.copy();env.setdefault('OPENBLAS_NUM_THREADS','1');env.setdefault('OMP_NUM_THREADS','2')
    env['NO_PROXY']='127.0.0.1,::1,localhost'
    collection='capacity_'+str(args.tier)
    with QdrantProcess(args.run_dir/'server',port=args.port) as server:
        command=[sys.executable,str(ROOT/'tools/memory_capacity.py'),'--tier',str(args.tier),
            '--seed-method','sdk-batch','--run-dir',str(args.run_dir/'measurement'),'--output',str(report_path),
            '--qdrant-url',server.url,'--qdrant-collection',collection,'--qdrant-pid',str(server.process.pid)]
        if args.baseline_run:command+=['--import-local-fixture',str(args.baseline_run)]
        with (args.run_dir/'measurement.log').open('w') as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=5400)
        measured=json.loads(report_path.read_text()) if report_path.exists() else {}
        checks={'measurement_completed':measured.get('phase')=='complete','latency_checks':measured.get('latency_status')=='PASS',
            'server_alive':server.process.poll() is None,'exact_vector_search':measured.get('qdrant_collection_config',{}).get('config',{}).get('hnsw_config',{}).get('m')==0}
        baseline={}
        if args.baseline_report:
            baseline=json.loads(args.baseline_report.read_text())
            checks['same_corpus']=baseline.get('dataset_sha256')==measured.get('dataset_sha256')
            checks['same_hit_mask']=bool(measured.get('queries')) and [x.get('expected_key_in_top5') for x in baseline.get('queries',[])]==[x.get('expected_key_in_top5') for x in measured.get('queries',[])]
        report={'created_at':stamp(),'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
            'scope':'Isolated native Qdrant CPU service comparison; exact dense search and native payload indexes. Does not change production defaults or establish independent memory quality/restore acceptance.',
            'server':{'version':VERSION,'source_commit':COMMIT,'binary_sha256':sha256(BINARY),'loopback_url':server.url,'grpc_enabled':False,'telemetry_enabled':False},
            'capacity_report':str(report_path),'subprocess_returncode':result.returncode,
            'query_summary':measured.get('query_summary'),'write_summary':measured.get('write_summary'),
            'baseline_query_summary':baseline.get('query_summary'),'baseline_write_summary':baseline.get('write_summary'),
            'baseline_numbered_recall':baseline.get('exact_project_recall_at_5'),'numbered_recall':measured.get('exact_project_recall_at_5'),
            'resources':measured.get('resources'),'assistant_acceptance':'NOT_VERIFIED'}
    report['server_stopped_after_measurement']=True
    write_json(args.output,report);print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
