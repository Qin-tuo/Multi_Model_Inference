"""Qdrant server setup/import for capacity comparisons using actual Mem0 SDK paths."""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from qdrant_client import QdrantClient,models
from app.memory import Mem0Adapter,build_mem0_config
from app.adapters.embedding import BGEQueryAdapter


def adapter(path,model,url,collection):
    import onnxruntime as ort
    from mem0 import Memory
    Path(path).mkdir(parents=True,exist_ok=True,mode=0o700)
    config=build_mem0_config(path,model,embedding_backend='onnx',qdrant_url=url,qdrant_collection=collection)
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    config['embedder']['config']['model_kwargs']['model_kwargs']['session_options']=options
    sdk=Memory.from_config(config)
    sdk.embedding_model=BGEQueryAdapter(sdk.embedding_model)
    memory=Mem0Adapter(path,sdk=sdk,embedding_scope='PRODUCTION')
    try:
        client=sdk.vector_store.client
        info=client.get_collection(collection)
        if info.config.hnsw_config.m!=0:raise RuntimeError('Capacity comparison requires exact scanning; HNSW must be disabled')
        for field in ('namespace','key','memory_type','operation_key','derived_from'):
            client.create_payload_index(collection,field,models.PayloadSchemaType.KEYWORD,wait=True)
        return memory
    except BaseException:
        memory.close();raise


def copy_database(source,dest):
    with sqlite3.connect(source) as src,sqlite3.connect(dest) as dst:src.backup(dst)


def import_closed_fixture(source,dest,url,collection,corpus,tier):
    """Keep source IDs/vectors/payloads, remove only its50 measurement additions.

    This imports this tool's synthetic fixture only. It is not an application
    backup/restore API and never merges arbitrary user deletion journals.
    """
    source=Path(source).resolve();dest=Path(dest).resolve()
    config=json.loads((source/'capacity-run.json').read_text())
    if config['tier']!=tier:raise ValueError('Source fixture tier differs')
    if dest.exists():raise ValueError('Import destination must be new')
    dest.mkdir(parents=True,mode=0o700)
    remote=QdrantClient(url=url,timeout=30)
    old=None
    try:
        if remote.collection_exists(collection):raise ValueError('Refusing to overwrite an existing server collection')
        remote.create_collection(collection,vectors_config=models.VectorParams(size=512,distance=models.Distance.COSINE,on_disk=True),
            sparse_vectors_config={'bm25':models.SparseVectorParams(modifier=models.Modifier.IDF)},
            hnsw_config=models.HnswConfigDiff(m=0))
        for field in ('user_id','namespace','key','memory_type','operation_key','derived_from'):
            remote.create_payload_index(collection,field,models.PayloadSchemaType.KEYWORD,wait=True)
        with (source/'memory/adapter.writer.lock').open('a+b') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            old=QdrantClient(path=str(source/'memory/qdrant'))
            offset=None;counts={'owner':0,'guest':0};digest=hashlib.sha256();sample=[]
            while True:
                rows,offset=old.scroll('assistant_memories',limit=128,offset=offset,with_vectors=True,with_payload=True)
                points=[]
                for row in rows:
                    p=row.payload;match=re.fullmatch(r'seed-(owner|guest)-(\d+)',p.get('operation_key',''))
                    if not match:continue
                    profile,index=match[1],int(match[2]);total=tier if profile=='owner' else corpus['guest_records']
                    if p.get('namespace')!=corpus['namespace'] or p.get('user_id')!=corpus[profile] or not 0<=index<total:
                        raise ValueError('Source fixture contains unexpected seed metadata')
                    points.append(models.PointStruct(id=row.id,vector=row.vector,payload=p));counts[profile]+=1
                    digest.update(json.dumps({'id':str(row.id),'payload':p,'vector':row.vector},sort_keys=True,separators=(',',':')).encode())
                    if len(sample)<10:sample.append(row)
                if points:remote.upsert(collection,points=points,wait=True)
                if offset is None:break
            if counts!={'owner':tier,'guest':corpus['guest_records']}:raise RuntimeError('Source seed count mismatch')
            for name in ('operations.sqlite','history.sqlite'):copy_database(source/'memory'/name,dest/name)
            # SDK source contains benchmark transaction writes, which are not
            # part of the frozen population. Their receipts must not turn the
            # next measured writes into idempotent-replay timings.
            with sqlite3.connect(dest/'operations.sqlite') as db:
                if db.execute("SELECT count(*) FROM operations WHERE status!='COMPLETED'").fetchone()[0]:raise RuntimeError('Source contains pending operations')
                discarded=db.execute("SELECT object_id FROM operations WHERE request_id LIKE 'write-probe-%'").fetchall()
                db.execute("DELETE FROM operations WHERE request_id LIKE 'write-probe-%'")
            with sqlite3.connect(dest/'history.sqlite') as db:
                db.executemany('DELETE FROM history WHERE memory_id=?',discarded)
            verified=[]
            for row in sample:
                received=remote.retrieve(collection,[row.id],with_vectors=True)[0]
                error=max(abs(a-b) for a,b in zip(row.vector,received.vector))
                verified.append({'id':str(row.id),'payload_equal':row.payload==received.payload,'max_vector_abs_error':error})
            if not all(r['payload_equal'] and r['max_vector_abs_error']<1e-6 for r in verified):raise RuntimeError('Imported data verification failed')
        return {'source_directory':str(source),'source_dataset_sha256':config['dataset_sha256'],'counts':counts,
            'source_points_sha256':digest.hexdigest(),'discarded_measurement_rows':len(discarded),'sample_verification':verified,
            'scope':'Native public Qdrant point import; SQLite backup for SDK history/journal; only synthetic seed records preserved'}
    finally:
        if old:old.close()
        remote.close()
