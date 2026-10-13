"""Restore native snapshots with current control databases and a fenced cutover.

Current SQLite controls are mandatory: a historical backup cannot know later
erasures or deliveries. If a later acknowledged fact is missing from both the
snapshot and the current vector store, publication fails instead of rolling back.
"""
from contextlib import ExitStack,closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from types import SimpleNamespace
import uuid

from app.data_lifecycle import MARKER,check_available,read_state
from app.memory_store import StoreBinding,check_transition,collection_metadata,new_binding
from deploy.data.backup import DATABASES,REQUIRED,collection_state,file_info,now,sqlite_copy,sqlite_state,sync_directory,upload_snapshot_to_new_collection,validate
from deploy.memory.migrate import canonical,deletion_policy,lock,matching,pages
from deploy.memory.runtime import VERSION,atomic_json,load


def data_locks(stack,directory):
    lock(stack,directory/'memory/adapter.writer.lock')
    lock(stack,directory/'records.sqlite.owner.lock')
    lock(stack,directory/'reminders.sqlite.owner.lock')


def rows(path,table):
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db:
        db.row_factory=sqlite3.Row
        return [dict(row) for row in db.execute('SELECT * FROM "'+table+'"')]


def controls(source,backup,manifest):
    """Reject an older or unrelated authority before modifying either directory."""
    names=[name for name in DATABASES if (source/name).is_file()]
    if not set(manifest['databases'])<=set(names):raise RuntimeError('Current control databases are missing; old backup alone cannot authorize restoration')
    state={name:sqlite_state(source/name,REQUIRED[name]) for name in names}
    journal=source/'memory/operations.sqlite'
    current={(x['user_id'],x['request_id']):x for x in rows(journal,'operations')}
    for prior in rows(backup/'memory/operations.sqlite','operations'):
        latest=current.get((prior['user_id'],prior['request_id']))
        if (latest is None or latest['fingerprint']!=prior['fingerprint'] or latest['operation']!=prior['operation']
                or prior['status']=='COMPLETED' and latest['status']!='COMPLETED'):
            raise RuntimeError('Current operation journal is older than or conflicts with the backup')
    for table,keys in [('deletion_barriers',('memory_id','user_id')),('cancelled_sources',('user_id','source_id'))]:
        latest={tuple(x[k] for k in keys) for x in rows(journal,table)}
        if not {tuple(x[k] for k in keys) for x in rows(backup/'memory/operations.sqlite',table)}<=latest:
            raise RuntimeError('Current deletion/cancellation authority is older than the backup')
    old_generations={x['user_id']:x['generation'] for x in rows(backup/'memory/operations.sqlite','owner_generations')}
    latest_generations={x['user_id']:x['generation'] for x in rows(journal,'owner_generations')}
    if any(latest_generations.get(owner,-1)<value for owner,value in old_generations.items()):
        raise RuntimeError('Current owner generation regressed')
    if any(x['status']!='COMPLETED' and x['operation'] in {'remember','correct'} for x in current.values()):
        raise RuntimeError('Reconcile pending memory writes before restoring; their arguments are not recoverable from the journal alone')
    if 'records.sqlite' in manifest['databases']:
        old={(x['user_id'],x['memory_id']) for x in rows(backup/'records.sqlite','deleted_memory_refs')}
        if not old<={(x['user_id'],x['memory_id']) for x in rows(source/'records.sqlite','deleted_memory_refs')}:
            raise RuntimeError('Current transcript deletion records regressed')
    if 'reminders.sqlite' in manifest['databases']:
        latest={x['id']:x for x in rows(source/'reminders.sqlite','assistant_reminders')}
        for old in rows(backup/'reminders.sqlite','assistant_reminders'):
            item=latest.get(old['id'])
            if (item is None or any(item[k]!=old[k] for k in ('user_id','text','due_utc','created_utc'))
                    or old['status'] not in {'active','fired'} and item['status'] in {'active','fired'}):
                raise RuntimeError('Current reminder state is missing or regressed')
        latest_ops={x['id']:x for x in rows(source/'reminders.sqlite','assistant_reminder_operations')}
        if any(latest_ops.get(old['id'])!=old for old in rows(backup/'reminders.sqlite','assistant_reminder_operations')):
            raise RuntimeError('Current reminder operation receipts regressed')
        latest_notices={x['id']:x for x in rows(source/'reminders.sqlite','assistant_notifications')}
        for old in rows(backup/'reminders.sqlite','assistant_notifications'):
            item=latest_notices.get(old['id'])
            if (item is None or any(item[key]!=old[key] for key in ('reminder_id','user_id','text','created_utc'))
                    or item['delivery_attempts']<old['delivery_attempts']
                    or old['status'] not in {'pending','claimed'} and item['status'] in {'pending','claimed'}
                    or old['status'] not in {'pending','claimed','playing'} and item['status']=='playing'):
                raise RuntimeError('Current notification delivery state regressed')
    return state,current


def expected_memories(operations):
    result={}
    for operation in operations.values():
        if operation['operation'] not in {'remember','correct'} or operation['status']!='COMPLETED':continue
        receipt=json.loads(operation['result_json']);mid=receipt.get('memory_id');version=receipt.get('version')
        if not mid or type(version)!=int:raise RuntimeError('Unsupported confirmed memory receipt')
        old=result.get(mid)
        if old and old['owner']!=operation['user_id']:raise RuntimeError('Confirmed memory ID crosses owners')
        if old is None or version>old['version']:
            result[mid]={'owner':operation['user_id'],'version':version,'request_id':operation['request_id']}
    return result


def erase_restored(client,collection,directory,records,changed):
    from mem0.memory.storage import SQLiteManager
    from patches.mem0.purge_history import purge_history
    from patches.mem0.purge_entities import purge_entities
    from qdrant_client import models
    owners,barriers=deletion_policy(client,directory,records,collection=collection)
    manager=SQLiteManager(str(directory/'history.sqlite'))
    try:
        sdk=SimpleNamespace(db=manager,collection_name=collection,vector_store=SimpleNamespace(client=client))
        for owner in sorted(set(barriers.values())):
            ids=[mid for mid,value in barriers.items() if value==owner]
            # Redact retained transcripts both by references and source turns.
            source_ids=[]
            for begin in range(0,len(ids),256):
                found=client.retrieve(collection,ids[begin:begin+256],with_payload=True)
                source_ids.extend(p.payload['source_id'] for p in found if p.payload and p.payload.get('source_id'))
                client.delete(collection,models.PointIdsList(points=ids[begin:begin+256]),wait=True)
            purge_history(sdk,ids,owner);purge_entities(sdk,ids,owner)
            if records:
                with sqlite3.connect(records) as db:
                    db.execute('PRAGMA secure_delete=ON')
                    db.executemany('INSERT OR IGNORE INTO deleted_memory_refs VALUES (?,?)',[(owner,mid) for mid in ids])
                    for mid in ids:
                        db.execute("UPDATE turns SET status='redacted',user_text='',assistant_text='' WHERE user_id=? AND id IN (SELECT turn_id FROM turn_memory_refs WHERE memory_id=?)",(owner,mid))
                    for source_id in source_ids:
                        db.execute("UPDATE turns SET status='redacted',user_text='',assistant_text='' WHERE user_id=? AND id=?",(owner,source_id))
        # Old entity associations to a corrected fact are derived from its old
        # body. Invalidate those associations, preserving other valid links.
        for owner in sorted(set(changed.values())):purge_entities(sdk,[mid for mid,value in changed.items() if value==owner],owner)
        with sqlite3.connect(directory/'operations.sqlite') as db:db.execute("UPDATE deletion_barriers SET status='COMPLETED'")
    finally:manager.close()
    return barriers


def reembed_snapshots(client,backup,manifest,state,directory,records,operations,profile,publish,checkpoint):
    """Import an older layout privately, then rebuild only currently live facts."""
    from qdrant_client import models
    from app.memory_profiles import profile_of,layout_matches,lexical_layout_matches,vector_params
    from app.memory_lexical import upsert_canonical
    base=state['binding']['collection'];staged={}
    for role,evidence in manifest['collections'].items():
        name=base+('_entities' if role=='entities' else '')+'_source'
        source_meta=collection_metadata(manifest['source_binding'])
        if role=='entities':source_meta=dict(source_meta,role='entities')
        owned=dict(collection_metadata(state['binding']),role=role,reindex_source=True)
        state.setdefault('staged_snapshots',{})[role]=name;publish()
        if not client.collection_exists(name):
            upload_snapshot_to_new_collection(client,name,backup/evidence['file'],manifest['files'][evidence['file']]['sha256'])
        info=client.get_collection(name)
        allowed=(source_meta,owned) if role=='memory' else (source_meta,owned,None,{})
        if (info.config.metadata not in allowed or not layout_matches(info.config.params.vectors,manifest['source_binding']['dimensions'],profile_of(manifest['source_binding']))
                or (role=='memory' and not lexical_layout_matches(info,profile_of(manifest['source_binding'])))):
            raise RuntimeError('Cross-profile snapshot staging differs from its recorded source')
        if collection_state(client,name)!={key:evidence[key] for key in ('sha256','points')}:
            raise RuntimeError('Cross-profile snapshot upload is unverified')
        client.update_collection(name,metadata=owned);staged[role]=name
        publish();checkpoint('snapshot_staged')
    _,barriers=deletion_policy(client,directory,records,collection=staged['memory'])
    expected=expected_memories(operations);active={mid:v for mid,v in expected.items() if mid not in barriers}
    for role in ('memory','entities'):
        target=base+('_entities' if role=='entities' else '');metadata=collection_metadata(state['binding'])
        if role=='entities':metadata=dict(metadata,role='entities')
        if not client.collection_exists(target):
            client.create_collection(target,vectors_config=vector_params(profile.dimensions,profile.profile_id),
                sparse_vectors_config={'bm25':models.SparseVectorParams(modifier=models.Modifier.IDF)},hnsw_config=models.HnswConfigDiff(m=0),metadata=metadata)
        info=client.get_collection(target)
        if (info.config.metadata!=metadata or not layout_matches(info.config.params.vectors,profile.dimensions,profile.profile_id)
                or (role=='memory' and not lexical_layout_matches(info,profile.profile_id))):
            raise RuntimeError('Cross-profile restore target is unrelated')
        if role not in staged:continue
        for batch in pages(client,staged[role]):
            chosen=[]
            for point in batch:
                mid=str(point.id);p=dict(point.payload or {});owner=p.get('user_id')
                if role=='memory':
                    if mid in barriers:continue
                    want=active.get(mid)
                    if want is None:raise RuntimeError('Snapshot contains an unconfirmed fact')
                    if owner!=want['owner']:raise RuntimeError('Snapshot fact crosses owners')
                    # Newer versions must come from current confirmed data.
                    if p.get('version')!=want['version'] or p.get('operation_key')!=want['request_id']:continue
                else:
                    links=p.get('linked_memory_ids')
                    if not isinstance(links,list) or not isinstance(owner,str):raise RuntimeError('Unsupported snapshot entity')
                    if any(mid in expected and expected[mid]['owner']!=owner for mid in links):raise RuntimeError('Snapshot entity crosses owners')
                    p['linked_memory_ids']=[mid for mid in links if mid in active and active[mid]['version']==1]
                    if not p['linked_memory_ids']:continue
                if not isinstance(p.get('data'),str):raise RuntimeError('Snapshot embedding source is missing')
                chosen.append((point.id,p))
            vectors=profile.embedder.embed_batch([p['data'] for _,p in chosen],'add')
            if len(vectors)!=len(chosen):raise RuntimeError('Snapshot re-embedding length differs')
            if chosen:upsert_canonical(client,target,[models.PointStruct(id=mid,vector={'':vector},payload=p) for (mid,p),vector in zip(chosen,vectors)],profile.profile_id,entities=role=='entities')
        state['snapshots'][role]='REEMBEDDED';publish();checkpoint('snapshot')


def cleanup_staged(client,state):
    for role,name in state.get('staged_snapshots',{}).items():
        expected_name=state['binding']['collection']+('_entities' if role=='entities' else '')+'_source'
        expected=dict(collection_metadata(state['binding']),role=role,reindex_source=True)
        if role not in {'memory','entities'} or name!=expected_name:raise RuntimeError('Invalid staged snapshot ownership')
        if client.collection_exists(name) and client.get_collection(name).config.metadata!=expected:raise RuntimeError('Staged snapshot metadata changed')
    for name in state.get('staged_snapshots',{}).values():
        if client.collection_exists(name):client.delete_collection(name)


def merge_vectors(client,collection,current_client,current_collection,directory,records,operations,profile):
    from qdrant_client import models
    from app.memory_profiles import vector_params
    from app.memory_lexical import upsert_canonical
    expected=expected_memories(operations)
    deleted={x['memory_id']:x['user_id'] for x in rows(directory/'operations.sqlite','deletion_barriers')}
    if records:
        for item in rows(records,'deleted_memory_refs'):
            if item['memory_id'] in deleted and deleted[item['memory_id']]!=item['user_id']:raise RuntimeError('Deletion owner conflict')
            deleted[item['memory_id']]=item['user_id']
    for mid,owner in deleted.items():
        if mid in expected and expected[mid]['owner']!=owner:raise RuntimeError('Deletion owner differs from confirmed memory')
    active={mid:value for mid,value in expected.items() if mid not in deleted};changed={};overlaid=0
    for batch in pages(client,collection):
        for point in batch:
            mid=str(point.id)
            if mid not in expected and mid not in deleted:raise RuntimeError('Snapshot contains unconfirmed memory; refusing to publish it')
            if mid in expected and (point.payload or {}).get('user_id')!=expected[mid]['owner']:
                raise RuntimeError('Snapshot memory owner differs from the journal')
    mids=list(active)
    for begin in range(0,len(mids),32):
        ids=mids[begin:begin+32]
        old={str(p.id):p for p in client.retrieve(collection,ids,with_payload=True,with_vectors=True)}
        current={str(p.id):p for p in current_client.retrieve(current_collection,ids,with_payload=True,with_vectors=True)} if current_client else {}
        points=[]
        for mid in ids:
            want=active[mid]
            def valid(point):
                return (point is not None and (point.payload or {}).get('user_id')==want['owner']
                    and point.payload.get('version')==want['version'] and point.payload.get('operation_key')==want['request_id']
                    and isinstance(point.payload.get('data'),str) and bool(point.payload['data']) and bool(point.vector))
            candidate=current.get(mid) if valid(current.get(mid)) else old.get(mid)
            if not valid(candidate):
                raise RuntimeError('Acknowledged memory version is unavailable; a newer verified snapshot or repaired current store is required')
            if mid not in old or old[mid].payload!=candidate.payload:
                changed[mid]=want['owner'];overlaid+=1
            points.append(models.PointStruct(id=candidate.id,vector=candidate.vector,payload=candidate.payload))
        vectors=profile.embedder.embed_batch([p.payload['data'] for p in points],'add')
        if len(vectors)!=len(points):raise RuntimeError('Restored embedding batch length differs')
        for index,(point,vector) in enumerate(zip(points,vectors)):
            try:valid_vector=matching(point.vector,vector,profile.dimensions)
            except RuntimeError:valid_vector=False
            if valid_vector:continue
            saved=old.get(str(point.id));want=active[str(point.id)]
            if (saved is None or (saved.payload or {}).get('user_id')!=want['owner'] or saved.payload.get('version')!=want['version']
                    or saved.payload.get('operation_key')!=want['request_id']
                    or not matching(saved.vector,profile.embedder.embed_batch([saved.payload['data']],'add')[0],profile.dimensions)):
                raise RuntimeError('Restored embeddings differ from the qualified model')
            points[index]=models.PointStruct(id=saved.id,vector=saved.vector,payload=saved.payload)
        upsert_canonical(client,collection,points,profile.profile_id)
    # Entity payloads have no fact-version provenance. Invalidate associations
    # to corrected facts consistently on retries, rather than trusting old tags.
    invalidated={mid:value['owner'] for mid,value in expected.items() if value['version']>1}
    entities=collection+'_entities'
    if current_client and current_client.collection_exists(current_collection+'_entities'):
        if not client.collection_exists(entities):
            client.create_collection(entities,vectors_config=vector_params(profile.dimensions,profile.profile_id),
                sparse_vectors_config={'bm25':models.SparseVectorParams(modifier=models.Modifier.IDF)},hnsw_config=models.HnswConfigDiff(m=0),
                metadata=dict(client.get_collection(collection).config.metadata,role='entities'))
        for batch in pages(current_client,current_collection+'_entities'):
            points=[]
            for point in batch:
                p=dict(point.payload or {});links=p.get('linked_memory_ids');owner=p.get('user_id')
                if not isinstance(links,list) or not all(isinstance(mid,str) for mid in links) or not isinstance(owner,str):
                    raise RuntimeError('Unsupported current entity references')
                if any(mid in expected and expected[mid]['owner']!=owner for mid in links):raise RuntimeError('Current entity reference crosses owners')
                p['linked_memory_ids']=[mid for mid in links if mid in active and mid not in invalidated]
                if p['linked_memory_ids']:points.append(models.PointStruct(id=point.id,vector=point.vector,payload=p))
            if points:
                vectors=profile.embedder.embed_batch([point.payload['data'] for point in points],'add')
                if len(vectors)!=len(points) or any(not matching(point.vector,vector,profile.dimensions) for point,vector in zip(points,vectors)):
                    raise RuntimeError('Current entity embeddings differ from the qualified model')
                client.upsert(entities,points,wait=True)
    barriers=erase_restored(client,collection,directory,records,invalidated)
    actual={}
    for batch in pages(client,collection):
        for point in batch:
            mid=str(point.id);p=point.payload or {};want=active.get(mid)
            if want is None or mid in barriers or p.get('user_id')!=want['owner'] or p.get('version')!=want['version']:
                raise RuntimeError('Merged memory verification failed')
            actual[mid]=p
    if set(actual)!=set(active)-set(barriers):raise RuntimeError('Confirmed memory count differs after restoration')
    if client.collection_exists(entities):
        for batch in pages(client,entities):
            for row in batch:
                p=row.payload or {};links=p.get('linked_memory_ids');owner=p.get('user_id')
                if not isinstance(links,list) or not all(isinstance(mid,str) for mid in links) or not isinstance(owner,str):
                    raise RuntimeError('Unsupported restored entity references')
                if any(mid in expected and expected[mid]['owner']!=owner for mid in links):
                    raise RuntimeError('Restored entity reference crosses owners')
                kept=[mid for mid in links if mid in actual]
                if kept:client.set_payload(entities,{'linked_memory_ids':kept},points=[row.id],wait=True)
                else:client.delete(entities,models.PointIdsList(points=[row.id]),wait=True)
    return {'memories':len(actual),'deleted_ids':len(barriers),'latest_facts_overlaid':overlaid}


def restore(backup,source,destination,service,profile,*,checkpoint=lambda phase:None):
    backup=Path(backup).resolve();source=Path(source).resolve();destination=Path(destination).resolve()
    for left,right in ((source,destination),(backup,destination),(source,backup)):
        if left==right or left.is_relative_to(right) or right.is_relative_to(left):raise RuntimeError('Backup, current data and restore destination must be separate')
    manifest=validate(backup);backup_hash=file_info(backup/'manifest.json')['sha256']
    if manifest['qdrant_version']!=VERSION or manifest['versions']['mem0ai']!='2.2.1':raise RuntimeError('Backup framework versions are incompatible')
    original=json.loads((source/'memory/store.json').read_text())
    from app.memory_profiles import profile_of,layout_matches
    if (profile_of(original)!=profile.profile_id or original.get('backend')!='qdrant_server' or original.get('status')!='READY'
            or original.get('directory')!=str(source/'memory') or not re.fullmatch('[0-9a-f]{32}',original.get('store_id',''))
            or original.get('collection')!='assistant_'+original['store_id']):
        raise RuntimeError('Current memory binding is invalid')
    if original['embedding_sha256']!=profile.fingerprint or original['dimensions']!=profile.dimensions:
        raise RuntimeError('Restore embedding profile differs')
    reembedding=(profile_of(manifest['source_binding'])!=profile.profile_id or manifest['embedding_sha256']!=profile.fingerprint)
    with ExitStack() as stack:
        data_locks(stack,source);source_state=read_state(source)
        state=read_state(destination) if destination.exists() else None
        lineage=(source_state or {}).get('lineage_id',original['store_id'])
        if manifest.get('lineage_id',manifest['source_binding']['store_id'])!=lineage:raise RuntimeError('Backup and current data belong to different lineages')
        if state:
            if (state.get('source')!=str(source) or state.get('backup_sha256')!=backup_hash
                    or state.get('original_binding')!=original
                    or state.get('embedding_scope')!=profile.scope or state['binding']['instance_id']!=service.instance_id
                    or state['binding']['service_directory']!=str(service.directory)):
                raise RuntimeError('Restore request differs from its existing transition')
            if state['status']=='ACTIVE':
                check_available(destination)
                if not source_state or source_state.get('status')!='RETIRED' or source_state.get('restore_id')!=state['restore_id']:
                    raise RuntimeError('Restored source fence differs')
                return dict(state['result'],already_completed=True)
            if state['status'] not in {'RESTORING','VERIFIED'}:raise RuntimeError('Unsupported restore transition')
            if source_state!=state.get('source_previous_state') and source_state and (source_state.get('restore_id')!=state['restore_id'] or source_state.get('status') not in {'RESTORE_SOURCE','RETIRED'}):
                raise RuntimeError('Source is owned by another restore')
        else:
            check_available(source);check_transition(source/'memory')
            if destination.exists() and any(destination.iterdir()):raise RuntimeError('Restore destination must be empty')
        current_state,operations=controls(source,backup,manifest)
        control_hash=hashlib.sha256(canonical(current_state)).hexdigest()
        if state and state['control_sha256']!=control_hash:raise RuntimeError('Current controls changed after restoration began')
        if not state:
            destination.mkdir(parents=True,exist_ok=True,mode=0o700);(destination/'memory').mkdir(mode=0o700)
            state={'schema_version':1,'directory':str(destination),'status':'RESTORING','restore_id':uuid.uuid4().hex,
                'source':str(source),'backup_sha256':backup_hash,'lineage_id':lineage,'control_sha256':control_hash,
                'binding':new_binding(destination/'memory',service,profile.fingerprint,profile.dimensions,profile.profile_id),
                'original_binding':original,'backup_binding':manifest['source_binding'],'source_previous_state':source_state,
                'embedding_scope':profile.scope,'snapshots':{},'created_at':now()}
        data_locks(stack,destination)
        def publish():atomic_json(destination/MARKER,state)
        publish();checkpoint('target_marked')
        source_fence={'schema_version':1,'directory':str(source),'status':'RESTORE_SOURCE','restore_id':state['restore_id'],
            'target':str(destination),'lineage_id':lineage}
        if not source_state or source_state.get('status')!='RETIRED':atomic_json(source/MARKER,source_fence)
        checkpoint('source_fenced')
        current_service=load(original['service_directory'])
        lock(stack,current_service.directory/'leases'/(original['collection']+'.lock'))
        lock(stack,service.directory/'leases'/(state['binding']['collection']+'.lock'))
        client=stack.enter_context(closing(service.client(timeout=60)))
        current_client=stack.enter_context(closing(current_service.client(timeout=60)))
        if not current_client.collection_exists(original['collection']):current_client=None
        elif current_client.get_collection(original['collection']).config.metadata!=collection_metadata(original):
            raise RuntimeError('Current native store metadata differs')
        for name in current_state:sqlite_copy(source/name,destination/name)
        records=destination/'records.sqlite' if 'records.sqlite' in current_state else None
        if reembedding:
            reembed_snapshots(client,backup,manifest,state,destination/'memory',records,operations,profile,publish,checkpoint)
        for role,evidence in ([] if reembedding else manifest['collections'].items()):
            name=state['binding']['collection']+('_entities' if role=='entities' else '')
            expected=collection_metadata(state['binding'])
            if role=='entities':expected=dict(expected,role='entities')
            source_metadata=collection_metadata(manifest['source_binding'])
            if role=='entities':source_metadata=dict(source_metadata,role='entities')
            if not client.collection_exists(name):
                if role in state['snapshots']:raise RuntimeError('Recorded restored collection is missing')
                upload_snapshot_to_new_collection(client,name,backup/evidence['file'],manifest['files'][evidence['file']]['sha256'])
            info=client.get_collection(name)
            allowed=(source_metadata,expected) if role=='memory' else (source_metadata,expected,None,{})
            if info.config.metadata not in allowed:raise RuntimeError('Restore target collection metadata is unrelated')
            from qdrant_client import models
            from app.memory_profiles import lexical_layout_matches
            if (not layout_matches(info.config.params.vectors,profile.dimensions,profile.profile_id) or info.config.hnsw_config.m!=0
                    or (role=='memory' and not lexical_layout_matches(info,profile.profile_id))):
                raise RuntimeError('Restored vector configuration differs from the qualified profile')
            if role not in state['snapshots']:
                if collection_state(client,name)!={key:evidence[key] for key in ('sha256','points')}:
                    raise RuntimeError('Snapshot upload completion is unverified; inspect the existing operation before retrying')
                state['snapshots'][role]='UPLOADED';publish()
            client.update_collection(name,metadata=expected)
            checkpoint('snapshot')
        result=merge_vectors(client,state['binding']['collection'],current_client,original['collection'],destination/'memory',records,operations,profile)
        checkpoint('merged')
        if 'reminders.sqlite' in current_state:
            # Rebuild only this application's jobs through APScheduler recovery;
            # never load stale absolute paths or deserialize copied job pickles.
            with sqlite3.connect(destination/'reminders.sqlite') as db:
                db.execute('PRAGMA secure_delete=ON');db.execute('DELETE FROM apscheduler_jobs')
        if {name:sqlite_state(source/name,REQUIRED[name]) for name in current_state}!=current_state:
            raise RuntimeError('Current controls changed during restoration')
        atomic_json(destination/'memory/store.json',dict(state['binding'],status='READY'))
        for name in current_state:sqlite_state(destination/name,REQUIRED[name])
        result.update(status='PASS',already_completed=False,embedding_scope=profile.scope,
            scope='Native snapshot plus current SQLite authority and exclusive directory cutover; newer missing facts cause refusal, no disk-level erasure')
        state.update(status='VERIFIED',result=result);publish();checkpoint('verified')
        atomic_json(source/MARKER,dict(source_fence,status='RETIRED'));checkpoint('source_retired')
        cleanup_staged(client,state)
        state.update(status='ACTIVE',completed_at=now());state.pop('source_previous_state',None);publish()
        return result


def abort(destination,*,checkpoint=lambda phase:None):
    """Discard an unpublished target, keeping current source data untouched."""
    destination=Path(destination).resolve();state=read_state(destination)
    if not state or state.get('status') not in {'RESTORING','VERIFIED','ABORTING','ABORTED'}:
        raise RuntimeError('Only an unpublished restoration can be aborted')
    source=Path(state['source']);service=load(state['binding']['service_directory'])
    with ExitStack() as stack:
        data_locks(stack,source);data_locks(stack,destination)
        state=read_state(destination)
        if state['status']=='ACTIVE':raise RuntimeError('Active restoration cannot be aborted')
        if json.loads((source/'memory/store.json').read_text())!=state['original_binding']:
            raise RuntimeError('Original memory binding changed; abort requires explicit recovery')
        current=read_state(source)
        if current!=state['source_previous_state'] and (not current or current.get('restore_id')!=state['restore_id']):
            raise RuntimeError('Source is owned by another restoration')
        lock(stack,service.directory/'leases'/(state['binding']['collection']+'.lock'))
        client=stack.enter_context(closing(service.client(timeout=60)))
        base=state['binding']['collection'];expected=collection_metadata(state['binding'])
        for role in ('memory','entities'):
            name=base+('_entities' if role=='entities' else '')
            if not client.collection_exists(name):continue
            metadata=client.get_collection(name).config.metadata
            original=collection_metadata(state.get('backup_binding',state['original_binding']))
            allowed=(expected,original) if role=='memory' else (dict(expected,role='entities'),dict(original,role='entities'),None,{})
            if metadata not in allowed:raise RuntimeError('Unrelated restore collection cannot be removed')
            owned_metadata=expected if role=='memory' else dict(expected,role='entities')
            if role not in state['snapshots'] and metadata!=owned_metadata:
                raise RuntimeError('Snapshot operation outcome is not recorded; resume verification before aborting')
        # Validate staged copies before changing either directory's state.
        for role,name in state.get('staged_snapshots',{}).items():
            expected_name=base+('_entities' if role=='entities' else '')+'_source'
            metadata=dict(expected,role=role,reindex_source=True)
            if name!=expected_name or (client.collection_exists(name) and client.get_collection(name).config.metadata!=metadata):raise RuntimeError('Unrelated staged restore collection')
        state['status']='ABORTING';atomic_json(destination/MARKER,state)
        for name in (base+'_entities',base):
            if client.collection_exists(name):client.delete_collection(name)
            checkpoint('collection_removed')
        cleanup_staged(client,state)
        state['status']='ABORTED';atomic_json(destination/MARKER,state);checkpoint('target_aborted')
        if state['source_previous_state'] is None:
            marker=source/MARKER
            if marker.exists():marker.unlink();sync_directory(source)
        else:atomic_json(source/MARKER,state['source_previous_state'])
        return {'status':'ABORTED','source_directory':str(source),'scope':'Unpublished target collections removed, current source databases and backups retained'}
