"""Real BGE, public migration CLI and public application adapter; synthetic data."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.memory import Mem0Adapter
from app.memory_runtime import open_memory
from app.records import Records
from deploy.memory.runtime import load
from tools.project import CACHE,DATA,stamp,write_json


def main():
    service=load(DATA/'qdrant');checks={};migration=None
    with tempfile.TemporaryDirectory(prefix='real-memory-migration-') as tmp:
        app=Path(tmp);directory=app/'memory'
        try:
            with Mem0Adapter(directory,bge_path=CACHE/'models/bge-small-zh-v1.5-onnx',embedding_backend='onnx') as memory:
                first=memory.remember('synthetic-owner','preference','drink','我喜欢喝绿茶。',source_id='seed',request_id='save')
                memory.correct('synthetic-owner',first['memory_id'],'我喜欢喝桂花茶。',expected_version=1,source_id='correction',request_id='correct')
                forgotten=memory.remember('synthetic-owner','preference','obsolete','我喜欢旧款水杯。',source_id='obsolete')['memory_id']
                guest=memory.remember('synthetic-guest','preference','drink','我喜欢喝咖啡。',source_id='guest')['memory_id']
                entity=str(uuid.uuid4())
                memory.sdk.entity_store.insert(vectors=[memory.sdk.embedding_model.embed('饮品','add')],ids=[entity],payloads=[
                    {'data':'饮品','user_id':'synthetic-owner','linked_memory_ids':[first['memory_id'],forgotten]}])
            # Labelled intent persisted before a hypothetical interrupted forget.
            with Records(app/'records.sqlite') as records:records.purge_memory('synthetic-owner',forgotten)
            result=subprocess.run([sys.executable,str(ROOT/'tools/project.py'),'migrate-memory','--data-dir',str(app)],
                cwd=ROOT,capture_output=True,text=True,timeout=180)
            checks['public_cli_completed']=result.returncode==0
            if result.returncode:raise RuntimeError(result.stderr[-1800:])
            migration=json.loads(result.stdout)
            checks['real_model_validated']=migration['embedding_scope']=='PRODUCTION'
            checks['only_live_facts_copied']=migration['counts']=={'memories':2,'entities':1}
            with open_memory(directory) as memory:
                owner=memory.search('synthetic-owner',key='drink')['items']
                checks['same_id_version_content_sources']=(len(owner)==1 and owner[0]['memory_id']==first['memory_id']
                    and owner[0]['version']==2 and owner[0]['content']=='我喜欢喝桂花茶。'
                    and owner[0]['sources'][0]['source_id']=='correction')
                checks['semantic_recall_after_process_switch']=any(x['memory_id']==first['memory_id'] for x in memory.search('synthetic-owner',query='平时爱喝什么饮品？')['items'])
                checks['guest_isolated']=memory.search('synthetic-guest',key='drink')['items'][0]['memory_id']==guest
                checks['deleted_absent']=memory.search('synthetic-owner',key='obsolete')['items']==[]
                replay=memory.correct('synthetic-owner',first['memory_id'],'我喜欢喝桂花茶。',expected_version=1,source_id='correction',request_id='correct')
                checks['old_receipt_replays']=replay['replayed'] and replay['version']==2
                memory.forget('synthetic-owner',first['memory_id'],request_id='post-migration-forget')
                checks['native_entity_links_erased']=memory.sdk.vector_store.client.retrieve(migration['collection']+'_entities',[entity])==[]
            repeat=subprocess.run([sys.executable,str(ROOT/'tools/project.py'),'migrate-memory','--data-dir',str(app)],
                cwd=ROOT,capture_output=True,text=True,timeout=180)
            checks['repeat_does_not_reimport']=repeat.returncode==0 and json.loads(repeat.stdout)['already_completed']
            with open_memory(directory) as memory:
                checks['post_migration_forget_survives_reopen']=memory.list_memories('synthetic-owner')==[]
                checks['guest_survives']=len(memory.list_memories('synthetic-guest'))==1
            with closing(sqlite3.connect(directory/'history.sqlite')) as db:
                checks['forgotten_history_empty']=db.execute('SELECT count(*) FROM history WHERE memory_id IN (?,?)',(first['memory_id'],forgotten)).fetchone()[0]==0
        except Exception as error:
            checks['exception']=False;failure=str(error)
        finally:
            path=directory/'store.json'
            if path.exists():
                binding=json.loads(path.read_text())
                with closing(service.client()) as client:
                    for name in (binding['collection']+'_entities',binding['collection']):
                        if client.collection_exists(name):
                            if client.get_collection(name).config.metadata.get('assistant_store_id')!=binding['store_id']:
                                raise RuntimeError('Refusing to clean an unrelated smoke collection')
                            client.delete_collection(name)
    report={'created_at':stamp(),'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'migration':migration,
        'scope':'Public migration command with actual BGE ONNX, Mem0 and authenticated managed Qdrant; synthetic temporary profiles, real reopen/correct/forget. Not independent recall quality, full backup/restore, or physical storage erasure.'}
    if 'failure' in locals():report['error']=failure
    write_json(ROOT/'benchmarks/results/memory-migration-smoke.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
