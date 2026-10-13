"""Actual SIGKILL at native reindex boundaries, then resume the same operation."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.memory_runtime import open_memory
from app.memory_profiles import LEGACY,WINDOWS,model_path,entity_name
from app.memory_store import check_transition
from deploy.memory.migrate import production_profile
from deploy.memory.reindex import reindex
from deploy.memory.runtime import load
from tools.long_memory_probe import documents
from tools.project import DATA,stamp,write_json


def worker(args):
    def crash(phase):
        if phase==args.phase:os.kill(os.getpid(),signal.SIGKILL)
    reindex(args.worker,production_profile(model_path(WINDOWS),WINDOWS),checkpoint=crash)


def main(args):
    service=load(DATA/'qdrant');cases=[]
    with tempfile.TemporaryDirectory(prefix='window-reindex-recovery-') as temp:
        root=Path(temp)
        for phase in ('batch','verified','published'):
            for iteration in range(args.cycles):
                app=root/f'{phase}-{iteration}';bindings=[];checks={}
                try:
                    settings={'memory':{'vector_store':'qdrant_server','embedding_profile':LEGACY}}
                    with open_memory(app/'memory',settings=settings) as memory:
                        original=documents()[0]['content']
                        saved=memory.remember('owner','document','project',original,source_id='synthetic-source',request_id='save')
                    bindings.append(json.loads((app/'memory/store.json').read_text()))
                    killed=subprocess.run([sys.executable,__file__,'--worker',str(app),'--phase',phase],cwd=ROOT,capture_output=True,text=True,timeout=120)
                    checks['terminated_by_sigkill']=killed.returncode==-signal.SIGKILL
                    try:check_transition(app/'memory');checks['partial_store_blocked']=False
                    except RuntimeError:checks['partial_store_blocked']=True
                    resumed=subprocess.run([sys.executable,str(ROOT/'tools/project.py'),'reindex-memory','--data-dir',str(app)],cwd=ROOT,capture_output=True,text=True,timeout=120)
                    checks['public_resume_completed']=resumed.returncode==0
                    if resumed.returncode:raise RuntimeError(resumed.stderr[-2000:])
                    with open_memory(app/'memory') as memory:
                        found=memory.search('owner','海岚项目的最终交付箱编号是什么？')['items'][0]
                        checks['canonical_fact_and_source_preserved']=found['memory_id']==saved['memory_id'] and found['content']==original and found['sources'][0]['source_id']=='synthetic-source'
                        checks['tail_retrievable']= '海岚七号' in found['evidence_window']['text']
                        memory.forget('owner',saved['memory_id'],request_id='forget')
                    replay=subprocess.run([sys.executable,str(ROOT/'tools/project.py'),'reindex-memory','--data-dir',str(app)],cwd=ROOT,capture_output=True,text=True,timeout=120)
                    checks['completed_rerun_is_idempotent']=replay.returncode==0 and json.loads(replay.stdout).get('already_completed') is True
                    with open_memory(app/'memory') as memory:checks['forgotten_fact_not_replayed']=memory.list_memories('owner')==[]
                    row={'phase':phase,'iteration':iteration+1,'checks':checks,'status':'PASS' if all(checks.values()) else 'FAIL'}
                except Exception as error:
                    row={'phase':phase,'iteration':iteration+1,'checks':checks,'status':'FAIL','error':{'type':type(error).__name__,'message':str(error)}}
                finally:
                    path=app/'memory/store.json'
                    if path.exists():bindings.append(json.loads(path.read_text()))
                    marker=app/'memory/reindex-transition.json'
                    if marker.exists():
                        state=json.loads(marker.read_text());bindings.extend([state['original_binding'],state['binding']])
                    with closing(service.client()) as client:
                        for name in {name for binding in bindings for name in [binding['collection'],entity_name(binding['collection'])]}:
                            if client.collection_exists(name):client.delete_collection(name)
                cases.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
                write_json(args.output.with_suffix('.running.json'),{'status':'RUNNING','pid':os.getpid(),'cases':cases,'expected':3*args.cycles})
    report={'created_at':stamp(),'status':'PASS' if len(cases)==3*args.cycles and all(r['status']=='PASS' for r in cases) else 'FAIL',
        'cases':cases,'scope':'Real native Qdrant,BGE int8 and production adapter/CLI;3 reindex stages with actual SIGKILL,strict access fence,same-ID/source recovery and no replay after forget. Synthetic data; not power-loss/disk-loss or full JT acceptance.'}
    write_json(args.output,report);args.output.with_suffix('.running.json').unlink(missing_ok=True)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',type=Path);parser.add_argument('--phase')
    parser.add_argument('--cycles',type=int,default=10);parser.add_argument('--output',type=Path,default=ROOT/'benchmarks/results/window-reindex-recovery.json')
    args=parser.parse_args()
    if args.worker:worker(args)
    elif not 1<=args.cycles<=100:parser.error('cycles must be1..100')
    else:raise SystemExit(main(args))
