"""Inclusive span diagnostics on a closed, copied capacity fixture; never user data."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['MEM0_TELEMETRY']='false';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from app.memory import Mem0Adapter
from tools.project import CACHE,stamp,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    marker=json.loads((args.run_dir/'capacity-run.json').read_text())
    if marker.get('tier') not in (1000,10000,50000): raise ValueError('Not a capacity fixture')
    spans=defaultdict(list)
    with tempfile.TemporaryDirectory(prefix='capacity-span-') as tmp:
        # The caller must finish the benchmark first; confirm its one-writer
        # lease before copying the SDK-owned files, without opening the model.
        import fcntl
        with (args.run_dir/'memory/adapter.writer.lock').open('a+') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            shutil.copytree(args.run_dir/'memory',Path(tmp)/'memory')
        with Mem0Adapter(Path(tmp)/'memory',bge_path=CACHE/'models/bge-small-zh-v1.5-onnx',embedding_backend='onnx') as m:
            phase='warmup'
            def watch(obj,name,label):
                original=getattr(obj,name)
                def wrapper(*a,**kw):
                    start=time.perf_counter_ns()
                    try:return original(*a,**kw)
                    finally:spans[phase+':'+label].append((time.perf_counter_ns()-start)/1e6)
                setattr(obj,name,wrapper)
            for obj,name,label in [(m.sdk.embedding_model,'embed','embedding'),(m.sdk.vector_store,'search','qdrant_search'),
                    (m.sdk.vector_store,'list','qdrant_filtered_list'),(m.sdk.vector_store,'insert','qdrant_insert'),
                    (m.sdk.db,'add_history','history_insert'),(m,'_start','journal_start'),(m,'_finish','journal_finish')]:
                watch(obj,name,label)
            m.search('capacity-owner','请检索合成项目P00050的登记要求，保持原始来源。',namespace='capacity-v1')
            for i in range(10):
                phase='query'; start=time.perf_counter_ns()
                m.search('capacity-owner',f'请检索合成项目P{i*100+50:05d}的登记要求，保持原始来源。',namespace='capacity-v1')
                spans['query:total'].append((time.perf_counter_ns()-start)/1e6)
                phase='write';start=time.perf_counter_ns()
                m.remember('capacity-owner','fact',f'span-{i}',f'诊断编号D{i:03d}的记录应保存在本机。',
                    source_id=f'span-{i}',request_id=f'span-{i}',namespace='capacity-v1')
                spans['write:total'].append((time.perf_counter_ns()-start)/1e6)
    summary={k:{'calls':len(v),'total_ms':sum(v),'mean_per_call_ms':sum(v)/len(v)} for k,v in spans.items() if not k.startswith('warmup:')}
    report={'created_at':stamp(),'status':'MEASURED','source_tier':marker['tier'],
        'scope':'Ten queries and writes on an offline copy of the completed capacity fixture, including its50 measurement records; inclusive spans may overlap, not a new acceptance run',
        'thread_environment':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},
        'spans':summary,'samples_ms':{k:v for k,v in spans.items() if not k.startswith('warmup:')}}
    write_json(args.output,report);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
