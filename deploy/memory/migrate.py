"""Resume an in-place Mem0 local -> managed Qdrant transition using public APIs.

The operation journal and SDK history stay in their original directory. A
durable transition blocks application access until all target rows are checked.
The old Qdrant directory is retained, never automatically replayed after commit.
"""
from contextlib import ExitStack, closing
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from app.memory import build_mem0_config
from app.memory_store import StoreBinding, check_transition, collection_metadata, embedding_fingerprint, new_binding
from deploy.memory.runtime import atomic_json

SOURCE='assistant_memories'
TABLES={'operations.sqlite':('owner_generations','operations','deletion_barriers','cancelled_sources'),
        'history.sqlite':('history','messages')}


@dataclass(frozen=True)
class EmbeddingProfile:
    embedder:object
    fingerprint:str
    dimensions:int
    scope:str


def production_profile(model):
    from app.adapters.embedding import BGEQueryAdapter
    from mem0.utils.factory import EmbedderFactory
    import onnxruntime as ort
    config=build_mem0_config('.',model,embedding_backend='onnx')
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    config['embedder']['config']['model_kwargs']['model_kwargs']['session_options']=options
    embedder=EmbedderFactory.create('huggingface',config['embedder']['config'],config['vector_store']['config'])
    return EmbeddingProfile(BGEQueryAdapter(embedder),embedding_fingerprint(model),512,'PRODUCTION')


def lock(stack,path):
    handle=stack.enter_context(Path(path).open('a+b'))
    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Memory or conversation records are in use; close the application first') from None


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def pages(client,collection):
    cursor=None
    while True:
        rows,cursor=client.scroll(collection,limit=32,offset=cursor,with_vectors=True,with_payload=True)
        if rows:yield rows
        if cursor is None:break


def dense(vector,dimensions):
    if isinstance(vector,dict):
        # Mem0's unnamed dense vector can coexist with its optional BM25 slot.
        vector=vector.get('')
    if (not isinstance(vector,list) or len(vector)!=dimensions
            or any(not isinstance(x,(int,float)) or not math.isfinite(x) for x in vector)
            or sum(x*x for x in vector)<=0):
        raise RuntimeError('Unsupported or invalid source embedding')
    return vector


def matching(left,right,dimensions):
    a,b=dense(left,dimensions),dense(right,dimensions)
    cosine=sum(x*y for x,y in zip(a,b))/math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
    return cosine>=.99999


def sqlite_digest(path,tables):
    digest=hashlib.sha256()
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db:
        if db.execute('PRAGMA quick_check').fetchall()!=[('ok',)]:raise RuntimeError('Source SQLite integrity check failed')
        names={x[0] for x in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not set(tables)<=names:raise RuntimeError('Unsupported source SQLite schema')
        for table in tables:
            schema=db.execute('SELECT sql FROM sqlite_master WHERE type=? AND name=?',('table',table)).fetchone()[0]
            digest.update(canonical([table,schema]))
            columns=[x[1] for x in db.execute(f'PRAGMA table_info("{table}")')]
            order=','.join('"'+x.replace('"','""')+'"' for x in columns)
            for row in db.execute(f'SELECT * FROM "{table}" ORDER BY {order}'):digest.update(canonical(row))
    return digest.hexdigest()


def source_digest(client,collections,directory,records):
    digest=hashlib.sha256()
    # Local cosine vectors can be renormalized on open. Check their semantic
    # identity against the pinned model separately, rather than hashing floats.
    for name in collections:
        digest.update(canonical(name))
        for batch in pages(client,name):
            for row in batch:digest.update(canonical([str(row.id),row.payload]))
    for filename,tables in TABLES.items():digest.update(canonical([filename,sqlite_digest(directory/filename,tables)]))
    if records: digest.update(canonical(sqlite_digest(records,('deleted_memory_refs',))))
    return digest.hexdigest()


def deletion_policy(source,directory,records,*,collection=SOURCE):
    """Merge existing authorized deletion intents and their derived memories."""
    owners={};children={}
    for batch in pages(source,collection):
        for row in batch:
            p=row.payload or {};owner=p.get('user_id');parents=p.get('derived_from',[])
            if not isinstance(owner,str) or not owner or not isinstance(p.get('data'),str):
                raise RuntimeError('Unsupported memory payload or owner')
            if not isinstance(parents,list) or not all(isinstance(x,str) for x in parents):
                raise RuntimeError('Unsupported derived memory references')
            owners[str(row.id)]=owner
            for parent in parents:children.setdefault(parent,[]).append(str(row.id))
    with closing(sqlite3.connect(directory/'operations.sqlite')) as db:
        db.execute('PRAGMA synchronous=FULL')
        barriers=dict(db.execute('SELECT memory_id,user_id FROM deletion_barriers'))
        if records:
            with closing(sqlite3.connect(records.as_uri()+'?mode=ro',uri=True)) as record_db:
                for owner,mid in record_db.execute('SELECT user_id,memory_id FROM deleted_memory_refs'):
                    if mid in barriers and barriers[mid]!=owner:raise RuntimeError('Conflicting deletion owner')
                    barriers[mid]=owner
        pending=list(barriers)
        while pending:
            mid=pending.pop();owner=barriers[mid]
            if mid in owners and owners[mid]!=owner:raise RuntimeError('Deletion barrier owner differs from source fact')
            for child in children.get(mid,[]):
                if owners[child]!=owner:raise RuntimeError('Derived memory crosses owners')
                if child not in barriers:barriers[child]=owner;pending.append(child)
        generations={}
        with db:
            for mid,owner in barriers.items():
                if db.execute('SELECT 1 FROM deletion_barriers WHERE memory_id=?',(mid,)).fetchone():continue
                if owner not in generations:
                    db.execute('INSERT OR IGNORE INTO owner_generations VALUES (?,0)',(owner,))
                    db.execute('UPDATE owner_generations SET generation=generation+1 WHERE user_id=?',(owner,))
                    generations[owner]=db.execute('SELECT generation FROM owner_generations WHERE user_id=?',(owner,)).fetchone()[0]
                db.execute('INSERT INTO deletion_barriers VALUES (?,?,?,?,?)',
                    (mid,owner,generations[owner],'PENDING',datetime.now(timezone.utc).isoformat()))
    return owners,barriers


def migrate(directory,service,profile,*,records=None,checkpoint=lambda phase:None):
    """Exclusive, restartable import. The checkpoint hook is for fault tests."""
    from qdrant_client import QdrantClient,models
    from mem0.memory.storage import SQLiteManager
    from patches.mem0.purge_history import purge_history
    directory=Path(directory).resolve()
    records=Path(records).resolve() if records else None
    if not directory.is_dir():raise RuntimeError('Existing memory directory is required')
    if profile.scope not in {'PRODUCTION','FIXTURE'}:raise RuntimeError('Embedding scope must be explicit')
    marker=directory/'store-transition.json';binding_path=directory/'store.json'
    with ExitStack() as stack:
        lock(stack,directory/'adapter.writer.lock')
        from app.data_lifecycle import check_available
        check_available(directory.parent)
        if records:
            if not records.is_file():raise RuntimeError('Recorded conversation database is missing')
            lock(stack,str(records)+'.owner.lock')
        if marker.exists():
            state=json.loads(marker.read_text())
            expected=new_binding(directory,service,profile.fingerprint,profile.dimensions)
            expected.update(store_id=state['binding']['store_id'],collection=state['binding']['collection'])
            if (state.get('schema_version')!=1 or state.get('binding')!=expected
                    or state.get('embedding_scope')!=profile.scope
                    or state.get('records')!=(str(records) if records else None)):
                raise RuntimeError('Migration source, service or embedding profile differs')
            if state['status']=='COMPLETED':
                check_transition(directory)
                binding=StoreBinding(directory,service,embedding_sha256=profile.fingerprint,dimensions=profile.dimensions)
                stack.callback(binding.close)
                return dict(state['result'],already_completed=True)
            if state['status'] not in {'MIGRATING','VERIFIED'}:raise RuntimeError('Unsupported migration state')
        else:
            if binding_path.exists():raise RuntimeError('Already bound store requires a different restore operation')
            if not (directory/'qdrant/meta.json').is_file():raise RuntimeError('Legacy Qdrant source is missing')
            for name,tables in TABLES.items():sqlite_digest(directory/name,tables)
            state={'schema_version':1,'status':'MIGRATING','binding':new_binding(directory,service,profile.fingerprint,profile.dimensions),
                'records':str(records) if records else None,'embedding_scope':profile.scope,'source_sha256':None}
        if not (directory/'qdrant/meta.json').is_file():raise RuntimeError('Legacy Qdrant source is missing')
        source=stack.enter_context(closing(QdrantClient(path=str(directory/'qdrant'))))
        collections=sorted(x.name for x in source.get_collections().collections)
        if SOURCE not in collections or set(collections)-{SOURCE,SOURCE+'_entities'}:
            raise RuntimeError('Unsupported source collections; refusing a partial migration')
        for name in collections:
            vectors=source.get_collection(name).config.params.vectors
            if not isinstance(vectors,models.VectorParams) or vectors.size!=profile.dimensions or vectors.distance!=models.Distance.COSINE:
                raise RuntimeError('Source collection vector configuration differs')
        atomic_json(marker,state);checkpoint('marked')
        if binding_path.exists():
            existing=json.loads(binding_path.read_text())
            if existing not in [state['binding'],dict(state['binding'],status='READY')]:raise RuntimeError('Target binding changed during migration')
        else:atomic_json(binding_path,state['binding'])
        binding=StoreBinding(directory,service,embedding_sha256=profile.fingerprint,dimensions=profile.dimensions)
        stack.callback(binding.close);target=binding.client
        # Only erase source history for an already-persisted authorized intent.
        owners,barriers=deletion_policy(source,directory,records)
        manager=SQLiteManager(str(directory/'history.sqlite'));stack.callback(manager.close)
        for owner in sorted(set(barriers.values())):
            purge_history(SimpleNamespace(db=manager),[mid for mid,value in barriers.items() if value==owner],owner)
        fingerprint=source_digest(source,collections,directory,records)
        if state['source_sha256'] and state['source_sha256']!=fingerprint:raise RuntimeError('Source changed after migration began')
        state['source_sha256']=fingerprint;atomic_json(marker,state)
        counts={};excluded=0;pruned=0
        for name in collections:
            entity=name.endswith('_entities');dest=binding.collection+('_entities' if entity else '')
            metadata=dict(binding.expected,role='entities') if entity else binding.expected
            if entity and not target.collection_exists(dest):
                target.create_collection(dest,vectors_config=models.VectorParams(size=profile.dimensions,distance=models.Distance.COSINE,on_disk=True),
                    hnsw_config=models.HnswConfigDiff(m=0),metadata=metadata)
                for field in ('user_id','linked_memory_ids'):target.create_payload_index(dest,field,models.PayloadSchemaType.KEYWORD,wait=True)
            info=target.get_collection(dest)
            if (info.config.metadata!=metadata or info.config.params.vectors.size!=profile.dimensions
                    or info.config.params.vectors.distance!=models.Distance.COSINE or info.config.hnsw_config.m!=0):
                raise RuntimeError('Target collection ownership or vector configuration differs')
            count=0
            for batch in pages(source,name):
                points=[]
                for row in batch:
                    payload=dict(row.payload or {});owner=payload.get('user_id')
                    if not entity and str(row.id) in barriers:excluded+=1;continue
                    if entity:
                        links=payload.get('linked_memory_ids')
                        if not isinstance(owner,str) or not owner or not isinstance(links,list) or not all(isinstance(x,str) for x in links):
                            raise RuntimeError('Unsupported entity schema')
                        if any(mid in owners and owners[mid]!=owner for mid in links):raise RuntimeError('Entity reference crosses owners')
                        kept=[mid for mid in links if mid in owners and mid not in barriers]
                        pruned+=len(links)-len(kept)
                        if not kept:excluded+=1;continue
                        payload['linked_memory_ids']=kept
                    if not isinstance(payload.get('data'),str):raise RuntimeError('Missing embedding source text')
                    points.append(models.PointStruct(id=row.id,vector=row.vector,payload=payload))
                if not points:continue
                vectors=profile.embedder.embed_batch([p.payload['data'] for p in points],'add')
                if len(vectors)!=len(points) or any(not matching(p.vector,v,profile.dimensions) for p,v in zip(points,vectors)):
                    raise RuntimeError('Source embeddings differ from the qualified model; re-embedding requires a separate migration')
                target.upsert(dest,points,wait=True)
                returned={str(p.id):p for p in target.retrieve(dest,[p.id for p in points],with_payload=True,with_vectors=True)}
                for point in points:
                    actual=returned.get(str(point.id))
                    if actual is None or actual.payload!=point.payload or not matching(actual.vector,point.vector,profile.dimensions):
                        raise RuntimeError('Target point verification failed')
                count+=len(points);checkpoint('batch')
            if target.count(dest,exact=True).count!=count:raise RuntimeError('Target count differs; refusing to publish unknown points')
            counts['entities' if entity else 'memories']=count
        if source_digest(source,collections,directory,records)!=fingerprint:raise RuntimeError('Source changed during migration')
        state['result']={'status':'PASS','counts':counts,'excluded_rows':excluded,'entity_links_pruned':pruned,
            'deletion_barriers':len(barriers),'embedding_scope':profile.scope,'source_sha256':fingerprint,
            'collection':binding.collection,'already_completed':False,
            'scope':'Legacy Mem0 local to authenticated native Qdrant, original SQLite journal/history retained; not full backup/restore or physical erasure'}
        state['status']='VERIFIED';atomic_json(marker,state);checkpoint('verified')
        binding.ready();checkpoint('ready')
        state['status']='COMPLETED';atomic_json(marker,state)
        return state['result']


def abort(directory,service,*,records=None,checkpoint=lambda phase:None):
    """Remove only owned unpublished copies; a completed transition cannot roll back."""
    from qdrant_client import QdrantClient
    directory=Path(directory).resolve();records=Path(records).resolve() if records else None
    if not directory.is_dir():raise RuntimeError('Existing memory directory is required')
    marker=directory/'store-transition.json';path=directory/'store.json'
    with ExitStack() as stack:
        lock(stack,directory/'adapter.writer.lock')
        from app.data_lifecycle import check_available
        check_available(directory.parent)
        if records:
            if not records.is_file():raise RuntimeError('Recorded conversation database is missing')
            lock(stack,str(records)+'.owner.lock')
        if not marker.is_file():raise RuntimeError('No incomplete migration to abort')
        state=json.loads(marker.read_text());binding=state['binding']
        if state.get('status')=='COMPLETED':raise RuntimeError('Completed migration cannot roll back to retired memory')
        expected=new_binding(directory,service,binding['embedding_sha256'],binding['dimensions'])
        expected.update(store_id=binding['store_id'],collection=binding['collection'])
        if (state.get('schema_version')!=1 or state.get('status') not in {'MIGRATING','VERIFIED','ABORTING'}
                or binding!=expected or state.get('records')!=(str(records) if records else None)
                or len(binding['store_id'])!=32 or any(x not in '0123456789abcdef' for x in binding['store_id'])
                or binding['collection']!='assistant_'+binding['store_id']):
            raise RuntimeError('Migration source or service binding differs')
        if path.exists() and json.loads(path.read_text()) not in [binding,dict(binding,status='READY')]:
            raise RuntimeError('Target binding changed during migration')
        if not (directory/'qdrant/meta.json').is_file():raise RuntimeError('Legacy source is missing; abort is unsafe')
        source=stack.enter_context(closing(QdrantClient(path=str(directory/'qdrant'))))
        collections=sorted(x.name for x in source.get_collections().collections)
        if state['source_sha256'] and source_digest(source,collections,directory,records)!=state['source_sha256']:
            raise RuntimeError('Source changed after migration began; abort requires explicit recovery')
        lease=service.directory/'leases'/(binding['collection']+'.lock');lock(stack,lease)
        client=stack.enter_context(closing(service.client()));metadata=collection_metadata(binding)
        names=[(binding['collection']+'_entities',dict(metadata,role='entities')),(binding['collection'],metadata)]
        # Validate all ownership before deleting either collection.
        for name,expected_metadata in names:
            if client.collection_exists(name) and client.get_collection(name).config.metadata!=expected_metadata:
                raise RuntimeError('Target collection metadata differs; refusing to remove unrelated data')
        state['status']='ABORTING';atomic_json(marker,state);checkpoint('aborting')
        for name,_ in names:
            if client.collection_exists(name):client.delete_collection(name)
            checkpoint('collection_removed')
        if path.exists():path.unlink()
        descriptor=os.open(directory,os.O_DIRECTORY)
        try:
            os.fsync(descriptor);checkpoint('binding_removed')
            marker.unlink();os.fsync(descriptor)
        finally:os.close(descriptor)
    return {'status':'ABORTED','scope':'Owned unpublished target removed; original local store and current deletion barriers retained'}
