"""Four real SIGKILL transition points, actual BGE and managed native Qdrant."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.memory import Mem0Adapter
from deploy.memory.migrate import migrate,production_profile
from deploy.memory.runtime import load
from tools.project import CACHE,DATA,stamp,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child',choices=['marked','batch','verified','ready'])
    parser.add_argument('--directory',type=Path)
    args=parser.parse_args()
    service=load(DATA/'qdrant');model=CACHE/'models/bge-small-zh-v1.5-onnx'
    if args.child:
        def kill(phase):
            if phase==args.child:os.kill(os.getpid(),signal.SIGKILL)
        migrate(args.directory,service,production_profile(model),checkpoint=kill)
        raise RuntimeError('Expected kill point was not reached')
    results=[]
    with tempfile.TemporaryDirectory(prefix='migration-kill-') as tmp:
        root=Path(tmp);template=root/'template'
        with Mem0Adapter(template,bge_path=model,embedding_backend='onnx') as memory:
            kept=memory.remember('synthetic-owner','preference','drink','我喜欢桂花茶。',source_id='seed')['memory_id']
            old=memory.remember('synthetic-owner','preference','old','应该遗忘的测试内容。',source_id='old')['memory_id']
            stale=memory.sdk.vector_store.client.retrieve('assistant_memories',[old],with_vectors=True)[0]
            memory.forget('synthetic-owner',old)
            from qdrant_client.models import PointStruct
            memory.sdk.vector_store.client.upsert('assistant_memories',[PointStruct(id=stale.id,vector=stale.vector,payload=stale.payload)])
        del memory
        for phase in ('marked','batch','verified','ready'):
            app=root/phase;directory=app/'memory'
            # Closed, isolated synthetic fixture copy only; not a backup feature.
            shutil.copytree(template,directory)
            checks={};case={'phase':phase,'checks':checks}
            try:
                killed=subprocess.run([sys.executable,__file__,'--child',phase,'--directory',str(directory)],
                    cwd=ROOT,capture_output=True,text=True,timeout=120)
                checks['actual_sigkill']=killed.returncode==-signal.SIGKILL
                if not checks['actual_sigkill']:raise RuntimeError(killed.stderr[-1200:])
                try:Mem0Adapter(directory)
                except RuntimeError as error:checks['partial_open_rejected']='migration is incomplete' in str(error)
                else:checks['partial_open_rejected']=False
                resumed=subprocess.run([sys.executable,str(ROOT/'tools/project.py'),'migrate-memory','--data-dir',str(app)],
                    cwd=ROOT,capture_output=True,text=True,timeout=120)
                checks['public_cli_resumed']=resumed.returncode==0
                if resumed.returncode:raise RuntimeError(resumed.stderr[-1200:])
                binding=json.loads((directory/'store.json').read_text())
                checks['committed']=json.loads((directory/'store-transition.json').read_text())['status']=='COMPLETED'
                with closing(service.client()) as client:
                    points=client.scroll(binding['collection'])[0]
                    checks['acknowledged_fact_kept_once']=len(points)==1 and str(points[0].id)==kept and points[0].payload['data']=='我喜欢桂花茶。'
                    checks['forgotten_fact_not_restored']=client.retrieve(binding['collection'],[old])==[]
            except Exception as error:case['error']=str(error);checks['exception']=False
            finally:
                path=directory/'store.json'
                if path.exists():
                    binding=json.loads(path.read_text())
                    with closing(service.client()) as client:
                        name=binding['collection']
                        if client.collection_exists(name):
                            if client.get_collection(name).config.metadata.get('assistant_store_id')!=binding['store_id']:
                                raise RuntimeError('Unrelated collection cannot be cleaned')
                            client.delete_collection(name)
            case['status']='PASS' if all(checks.values()) else 'FAIL';results.append(case)
            print(json.dumps(case,ensure_ascii=False),flush=True)
    report={'created_at':stamp(),'status':'PASS' if all(x['status']=='PASS' for x in results) else 'FAIL',
        'actual_sigkills':sum(x['checks'].get('actual_sigkill',False) for x in results),'cases':results,
        'scope':'One real SIGKILL at each of four migration publication phases. Real BGE/Mem0/Qdrant/SQLite and public resume command, synthetic data; not ten repetitions per phase, power loss, complete backup restore, or full JT acceptance.'}
    write_json(ROOT/'benchmarks/results/memory-migration-recovery-smoke.json',report)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
