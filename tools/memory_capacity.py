"""Reproducible real BGE/Mem0/Qdrant capacity workload, isolated from user data."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['MEM0_TELEMETRY']='false'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
from app.memory import Mem0Adapter,build_mem0_config
from tools.project import CACHE, DATA, meminfo, stamp, write_json, sha256


class MemoryMonitor:
    def __init__(self,external_pid=None):
        self.stop=threading.Event(); self.minimum_available=None; self.peak_rss=0; self.samples=0
        self.external_pid=external_pid;self.external_peak_rss=0
        self.thread=threading.Thread(target=self.run,daemon=True)
    def run(self):
        while not self.stop.is_set():
            available=meminfo()['MemAvailable']
            self.minimum_available=available if self.minimum_available is None else min(self.minimum_available,available)
            for line in Path('/proc/self/status').read_text().splitlines():
                if line.startswith('VmRSS:'): self.peak_rss=max(self.peak_rss,int(line.split()[1])*1024)
            if self.external_pid:
                try:
                    for line in Path(f'/proc/{self.external_pid}/status').read_text().splitlines():
                        if line.startswith('VmRSS:'):self.external_peak_rss=max(self.external_peak_rss,int(line.split()[1])*1024)
                except FileNotFoundError:pass
            self.samples+=1
            self.stop.wait(.1)
    def __enter__(self): self.thread.start(); return self
    def __exit__(self,*_): self.stop.set(); self.thread.join(2)
    def report(self): return {'minimum_available_bytes':self.minimum_available,'peak_process_rss_bytes':self.peak_rss,
        'external_pid':self.external_pid,'peak_external_rss_bytes':self.external_peak_rss,'sample_count':self.samples,'period_seconds':.1}


def statistics(rows):
    ordered=sorted(row['elapsed_ms'] for row in rows)
    return {'count':len(rows),'errors':sum('error' in row for row in rows),
            'p50_ms':ordered[math.ceil(.5*len(rows))-1] if rows else None,
            'p95_ms':ordered[math.ceil(.95*len(rows))-1] if rows else None,
            'max_ms':max(ordered) if rows else None}


def timed(callback):
    start=time.perf_counter_ns()
    try: result=callback(); error=None
    except Exception as exc: result=None; error=type(exc).__name__+': '+str(exc)[:200]
    row={'elapsed_ms':(time.perf_counter_ns()-start)/1e6}
    if error: row['error']=error
    return row,result


def backend_state():
    result={}
    for name in ('llama','asr','speech','camera','vision','pose'):
        r=subprocess.run(['systemctl','--user','show',f'mmassistant-{name}.service','--property=ActiveState,MainPID,NRestarts'],capture_output=True,text=True,timeout=5)
        result[name]=dict(line.split('=',1) for line in r.stdout.splitlines() if '=' in line)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tier',type=int,choices=[1000,10000,50000],required=True)
    parser.add_argument('--run-dir',type=Path,required=True,help='Dedicated new or resumable benchmark directory, never application data')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed-method',choices=['adapter','sdk-batch'],default='adapter',
                        help='SDK batch is isolated fixture preparation only; measured writes always use the full adapter')
    parser.add_argument('--qdrant-url',help='Explicit isolated loopback server; application defaults stay unchanged')
    parser.add_argument('--qdrant-collection')
    parser.add_argument('--qdrant-pid',type=int,help='Owned evaluation server PID for resource sampling')
    parser.add_argument('--import-local-fixture',type=Path,help='Closed capacity run whose seed records/vectors should be preserved')
    args=parser.parse_args()
    corpus_path=ROOT/'benchmarks/corpora/memory-capacity-v1.json'
    corpus=json.loads(corpus_path.read_text())
    digest=sha256(corpus_path)
    model=CACHE/'models/bge-small-zh-v1.5-onnx'
    if args.qdrant_url or args.qdrant_collection:
        build_mem0_config(args.run_dir/'memory',model,qdrant_url=args.qdrant_url,qdrant_collection=args.qdrant_collection)
        if not args.qdrant_collection.startswith('capacity_'):raise ValueError('Evaluation collection must start with capacity_')
    if args.import_local_fixture and not args.qdrant_url:raise ValueError('Fixture import requires an isolated Qdrant server')
    folder=args.run_dir.resolve()
    # Only directories carrying this benchmark's marker may be resumed.
    marker=folder/'capacity-run.json'
    config={'dataset_sha256':digest,'tier':args.tier,'schema_version':1}
    if args.seed_method!='adapter': config['seed_method']=args.seed_method
    if args.qdrant_url:config.update(qdrant_url=args.qdrant_url,qdrant_collection=args.qdrant_collection)
    if folder.exists():
        if not marker.is_file() or json.loads(marker.read_text())!=config:
            raise ValueError('Refusing to use a directory not owned by this exact benchmark workload')
    else:
        folder.mkdir(parents=True,mode=0o700); write_json(marker,config)
    progress_path=folder/'population.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'owner':0,'guest':0}
    state_path=folder/'memory'
    report={'schema_version':1,'created_at':stamp(),'status':'RUNNING','tier':args.tier,'dataset_sha256':digest,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_state':'working tree; see checkpoint containing report','scope':corpus['purpose'],
        'source_files_sha256':{name:sha256(ROOT/name) for name in ('app/memory.py','app/adapters/embedding.py',
            'tools/memory_capacity.py','tools/memory_capacity_seed.py','tools/qdrant_capacity.py')},
        'run_directory':str(folder),'population_method':corpus['population_method'],'workload':corpus['backend_workload'],
        'thread_environment':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},
        'embedding':{'model':'bge-small-zh-v1.5','dimensions':512,'backend':'CPUExecutionProvider','intra_threads':2,'inter_threads':1,
            'onnx_sha256':sha256(model/'onnx/model.onnx')},
        'versions':{p:importlib.metadata.version(p) for p in ('mem0ai','qdrant-client','SQLAlchemy','onnxruntime')},
        'backends_before':backend_state(),'phase':'population','population':progress}
    report['vector_store']={'kind':'qdrant_server' if args.qdrant_url else 'qdrant_local',
                           'url':args.qdrant_url,'collection':args.qdrant_collection or 'assistant_memories'}
    if args.seed_method=='sdk-batch':
        report['population_method']='Isolated fixture import using actual BGE batch embeddings, native Mem0 Qdrant insert/history and adapter operation journal; prepopulation skips per-record conflict scans. Timed writes use unmodified Mem0Adapter.remember including embedding and all commits.'
    write_json(args.output,report)
    def fact(index):
        k=index%len(corpus['topics'])
        return corpus['record_template'].format(index=index,topic=corpus['topics'][k],value=corpus['values'][k])
    def create_adapter():
        if args.qdrant_url:
            from tools.qdrant_capacity import adapter
            return adapter(state_path,model,args.qdrant_url,args.qdrant_collection)
        return Mem0Adapter(state_path,bge_path=model,embedding_backend='onnx')
    with MemoryMonitor(args.qdrant_pid) as monitor:
        try:
            started=time.monotonic(); last_log=started
            if args.import_local_fixture:
                import_report=folder/'import.json'
                if not import_report.exists():
                    source_marker=json.loads((args.import_local_fixture/'capacity-run.json').read_text())
                    if source_marker['dataset_sha256']!=digest:raise ValueError('Import corpus differs from the frozen workload')
                    from tools.qdrant_capacity import import_closed_fixture
                    report['phase']='import';write_json(args.output,report)
                    imported=import_closed_fixture(args.import_local_fixture,state_path,args.qdrant_url,args.qdrant_collection,corpus,args.tier)
                    write_json(import_report,imported)
                    progress.update(imported['counts']);write_json(progress_path,progress)
                report['import']=json.loads(import_report.read_text())
                report['population_method']='Frozen synthetic seed IDs/vectors/payloads and SDK history/journal imported from the declared closed local baseline; its50 measured-write additions and receipts excluded. Timed operations use the full production adapter.'
            with create_adapter() as memory:
                from threadpoolctl import threadpool_info
                report['native_threadpools']=threadpool_info()
                if args.qdrant_url:
                    report['qdrant_collection_config']=memory.sdk.vector_store.client.get_collection(args.qdrant_collection).model_dump(mode='json')
                # Replay at most the last checkpoint batch after interruption;
                # adapter request IDs reconcile SDK/history/journal partial writes.
                for profile,total in [('owner',args.tier),('guest',corpus['guest_records'])]:
                    if args.seed_method=='sdk-batch':
                        from tools.memory_capacity_seed import populate
                        def checkpoint(count):
                            progress[profile]=count
                            write_json(progress_path,progress)
                            report['population']=dict(progress);report['resources']=monitor.report();write_json(args.output,report)
                            print(json.dumps({'phase':'population','profile':profile,'completed':count,'target':total},ensure_ascii=False),flush=True)
                        populate(memory,corpus,profile,total,args.tier,progress[profile],fact,checkpoint)
                        continue
                    for index in range(progress[profile],total):
                        if meminfo()['MemAvailable']<700*1024*1024: raise RuntimeError('CAPACITY_BOUNDARY: less than700MiB available')
                        content=fact(index if profile=='owner' else int((index+.5)*args.tier/corpus['queries']))
                        result=memory.remember(corpus[profile],'fact',f'project-{index:05d}',content,
                            source_id=f'synthetic-{profile}-{index}',request_id=f'seed-{profile}-{index}',namespace=corpus['namespace'])
                        if not result.get('durable'): raise RuntimeError('Seed write not durable')
                        progress[profile]=index+1
                        if index%50==49 or index+1==total:
                            write_json(progress_path,progress)
                            report['population']=dict(progress); report['resources']=monitor.report(); write_json(args.output,report)
                        if time.monotonic()-last_log>=5:
                            print(json.dumps({'phase':'population','profile':profile,'completed':index+1,'target':total,'elapsed_s':round(time.monotonic()-started,1)},ensure_ascii=False),flush=True)
                            last_log=time.monotonic()
            report['population_elapsed_seconds_this_process']=time.monotonic()-started
            reopen=time.perf_counter_ns()
            with create_adapter() as memory:
                report['reopen_ms']=(time.perf_counter_ns()-reopen)/1e6
                from qdrant_client.models import FieldCondition,Filter,MatchValue
                def count(user):
                    filt=Filter(must=[FieldCondition(key='user_id',match=MatchValue(value=user)),
                        FieldCondition(key='namespace',match=MatchValue(value=corpus['namespace']))])
                    return memory.sdk.vector_store.client.count(memory.sdk.collection_name,count_filter=filt,exact=True).count
                report['counts_before']={k:count(corpus[k]) for k in ('owner','guest')}
                if report['counts_before']!={'owner':args.tier,'guest':corpus['guest_records']}:
                    raise RuntimeError('Frozen population size mismatch; use a new run directory after measurement writes')
                report['phase']='queries';write_json(args.output,report)
                queries=[]
                indices=[int((j+.5)*args.tier/corpus['queries']) for j in range(corpus['queries'])]
                for index in indices[:corpus['warmups']]:
                    memory.search(corpus['owner'],corpus['query_template'].format(index=index),namespace=corpus['namespace'])
                for index in indices:
                    query=corpus['query_template'].format(index=index)
                    row,result=timed(lambda:memory.search(corpus['owner'],query,namespace=corpus['namespace'],top_k=corpus['top_k']))
                    row.update(index=index,query_characters=len(query),query_utf8_bytes=len(query.encode()))
                    if result is not None:
                        items=result['items']
                        row.update(returned=len(items),expected_key_in_top5=any(i['key']==f'project-{index:05d}' for i in items),
                            owner_leaks=sum(i['user_id']!=corpus['owner'] for i in items),retrieval_method=result['retrieval_method'])
                    queries.append(row)
                report['queries']=queries; report['query_summary']=statistics(queries)
                report['phase']='transactions';write_json(args.output,report)
                writes=[]
                for j in range(corpus['write_transactions']):
                    key=f'transaction-{j:03d}';text=f'临时测量记录T{j:03d}要求在离开实验室前关闭台灯。'
                    row,result=timed(lambda:memory.remember(corpus['owner'],'fact',key,text,
                        source_id=f'write-probe-{j}',request_id=f'write-probe-{j}',namespace=corpus['namespace']))
                    row['index']=j
                    if result is not None:
                        stored=memory.search(corpus['owner'],key=key,namespace=corpus['namespace'])['items']
                        row['verified_after_timer']=bool(result.get('durable') and len(stored)==1 and stored[0]['content']==text
                            and memory.sdk.history(result['memory_id']) and memory.operation_status(corpus['owner'],f'write-probe-{j}')['status']=='COMPLETED')
                    writes.append(row)
                report['writes']=writes;report['write_summary']=statistics(writes)
                report['counts_after']={k:count(corpus[k]) for k in ('owner','guest')}
                report['checks']={
                    'query_p95_200ms':report['query_summary']['p95_ms']<=corpus['query_p95_target_ms'],
                    'write_p95_500ms':report['write_summary']['p95_ms']<=corpus['write_p95_target_ms'],
                    'no_query_errors':not report['query_summary']['errors'],
                    'no_write_errors':not report['write_summary']['errors'],
                    'all_writes_confirmed':all(x.get('verified_after_timer',False) for x in writes),
                    'no_owner_leaks':all(x.get('owner_leaks')==0 for x in queries),
                    'no_empty_queries':all(x.get('returned',0)>0 for x in queries)}
                report['exact_project_recall_at_5']=sum(x.get('expected_key_in_top5',False) for x in queries)/len(queries)
                report['recall_scope']='Synthetic numbered near-duplicate stress observations; not independent JT-10 language quality'
                report['latency_status']='PASS' if all(report['checks'].values()) else 'FAIL'
                report['status']='MEASURED' if report['latency_status']=='PASS' else 'FAIL'
                report['qualification']='Capacity/latency observations only; report numbered-project recall separately. Independent memory quality and final JT acceptance remain unverified.'
                report['phase']='complete'
        except BaseException as error:
            report['status']='FAIL';report['error']={'type':type(error).__name__,'message':str(error)[:1000]}
            raise
        finally:
            report['resources']=monitor.report(); report['backends_after']=backend_state();report['updated_at']=stamp()
            report['assistant_acceptance']='NOT_VERIFIED'
            write_json(args.output,report)
    print(json.dumps({k:report[k] for k in ('status','tier','query_summary','write_summary','exact_project_recall_at_5','checks','resources')},ensure_ascii=False),flush=True)
    return 0 if report.get('latency_status')=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
