"""Fenced native re-embedding without copying or changing authoritative facts."""
from contextlib import ExitStack,closing
import json
from pathlib import Path
import re
from types import SimpleNamespace

from app.data_lifecycle import MARKER,check_available,read_state
from app.memory_profiles import LEGACY,WINDOWS,HYBRID,is_window_profile,lexical_layout_matches,profile_of,layout_matches,entity_name,entity_metadata
from app.memory_lexical import upsert_canonical
from app.memory_store import StoreBinding,check_transition,collection_metadata,new_binding
from deploy.data.backup import sync_directory
from deploy.data.restore import data_locks,expected_memories,rows
from deploy.memory.migrate import deletion_policy,lock,matching,pages,source_digest
from deploy.memory.runtime import atomic_json,load

MARK='reindex-transition.json'


def target_data_state(state):
    previous=state['previous_data_state']
    if previous:return dict(previous,binding=dict(state['binding'],status='READY'))
    return {'schema_version':1,'directory':state['directory'],'status':'ACTIVE','origin':'reindex',
        'lineage_id':state['original_binding']['store_id'],'binding':dict(state['binding'],status='READY')}


def validate_binding(binding,memory,service):
    if (not re.fullmatch('[0-9a-f]{32}',binding.get('store_id',''))
            or not re.fullmatch('[0-9a-f]{64}',binding.get('embedding_sha256',''))
            or type(binding.get('dimensions')) is not int or binding['dimensions']<=0):raise RuntimeError('Invalid reindex binding identity')
    expected=new_binding(memory,service,binding['embedding_sha256'],binding['dimensions'],profile_of(binding))
    expected.update(store_id=binding['store_id'],collection='assistant_'+binding['store_id'],status=binding.get('status'))
    if binding!=expected or binding['status'] not in {'PREPARING','READY'}:raise RuntimeError('Reindex binding ownership differs')


def validate_source(client,binding):
    if binding.get('status')!='READY':raise RuntimeError('Reindex source is not ready')
    info=client.get_collection(binding['collection'])
    if (info.config.metadata!=collection_metadata(binding)
            or info.config.hnsw_config.m!=0 or not layout_matches(info.config.params.vectors,binding['dimensions'],profile_of(binding))
            or not lexical_layout_matches(info,profile_of(binding))):
        raise RuntimeError('Reindex source collection does not match its binding')


def cleanup_collections(client,binding):
    names=[(entity_name(binding['collection']),entity_metadata(binding)),(binding['collection'],collection_metadata(binding))]
    for name,expected in names:
        if client.collection_exists(name):
            actual=client.get_collection(name).config.metadata
            allowed=(expected,None,{}) if name.endswith('_entities') and not is_window_profile(profile_of(binding)) else (expected,)
            if actual not in allowed:raise RuntimeError('Refusing to remove an unrelated reindex collection')
    for name,_ in names:
        if client.collection_exists(name):client.delete_collection(name)


def reindex(directory,profile,*,checkpoint=lambda phase:None):
    from qdrant_client import models
    from mem0.memory.storage import SQLiteManager
    from patches.mem0.purge_history import purge_history
    directory=Path(directory).resolve();memory=directory/'memory';marker=memory/MARK;binding_path=memory/'store.json'
    if not is_window_profile(profile.profile_id):raise RuntimeError('Reindex target must be a supported token-window profile')
    with ExitStack() as stack:
        data_locks(stack,directory)
        state=json.loads(marker.read_text()) if marker.exists() else None
        previous_reindex=None
        if state and state.get('status')=='COMPLETED' and profile_of(state['binding'])!=profile.profile_id:
            check_available(directory);check_transition(memory)
            previous_reindex=state;state=None
        if state:
            target=state['binding'];original=state['original_binding'];service=load(original['service_directory'])
            validate_binding(original,memory,service);validate_binding(target,memory,service)
            expected=new_binding(memory,service,profile.fingerprint,profile.dimensions,profile.profile_id)
            expected.update(store_id=target['store_id'],collection=target['collection'])
            if (state.get('schema_version')!=1 or state.get('directory')!=str(directory) or target!=expected or state.get('scope')!=profile.scope):
                raise RuntimeError('Reindex request differs from the recorded transition')
            if state['status']=='COMPLETED':
                check_available(directory);check_transition(memory)
                owned=StoreBinding(memory,service,embedding_sha256=profile.fingerprint,dimensions=profile.dimensions,profile=profile.profile_id)
                owned.close();return dict(state['result'],already_completed=True)
            if state['status'] not in {'WRITING','VERIFIED','PUBLISHED'}:raise RuntimeError('Unsupported reindex state')
        else:
            check_available(directory);check_transition(memory)
            original=json.loads(binding_path.read_text());service=load(original['service_directory'])
            validate_binding(original,memory,service)
            if {LEGACY:0,WINDOWS:1,HYBRID:2}[profile_of(original)]>{LEGACY:0,WINDOWS:1,HYBRID:2}[profile.profile_id]:
                raise RuntimeError('Reindex cannot downgrade the bound memory profile')
            if original.get('status')!='READY' or original.get('directory')!=str(memory):raise RuntimeError('Reindex requires an existing ready native store')
            if profile_of(original)==profile.profile_id and original['embedding_sha256']==profile.fingerprint:
                owned=StoreBinding(memory,service,embedding_sha256=profile.fingerprint,dimensions=profile.dimensions,profile=profile.profile_id)
                owned.close();return {'status':'PASS','already_completed':True,'already_current_profile':True}
            ops=rows(memory/'operations.sqlite','operations')
            if any(r['status']!='COMPLETED' and r['operation'] in {'remember','correct'} for r in ops):raise RuntimeError('Reconcile pending writes before reindexing')
            state={'schema_version':1,'directory':str(directory),'status':'WRITING','original_binding':original,
                'binding':new_binding(memory,service,profile.fingerprint,profile.dimensions,profile.profile_id),
                'previous_data_state':read_state(directory),'previous_reindex_state':previous_reindex,'source_sha256':None,'scope':profile.scope}
        lock(stack,service.directory/'leases'/(original['collection']+'.lock'))
        client=stack.enter_context(closing(service.client(timeout=60)))
        def publish():atomic_json(marker,state)
        if state['status']=='PUBLISHED':
            owned=StoreBinding(memory,service,embedding_sha256=profile.fingerprint,dimensions=profile.dimensions,profile=profile.profile_id)
            stack.callback(owned.close)
            cleanup_collections(client,original)
            state['status']='COMPLETED';publish();return state['result']
        validate_source(client,original)
        current_binding=json.loads(binding_path.read_text())
        if current_binding not in [original,state['binding'],dict(state['binding'],status='READY')]:raise RuntimeError('Reindex binding changed outside its transition')
        publish();checkpoint('marked')
        atomic_json(binding_path,state['binding'])
        target=StoreBinding(memory,service,embedding_sha256=profile.fingerprint,dimensions=profile.dimensions,profile=profile.profile_id)
        stack.callback(target.close)
        records=directory/'records.sqlite';records=records if records.exists() else None
        owners,barriers=deletion_policy(client,memory,records,collection=original['collection'])
        manager=SQLiteManager(str(memory/'history.sqlite'));stack.callback(manager.close)
        for owner in set(barriers.values()):purge_history(SimpleNamespace(db=manager),[mid for mid,value in barriers.items() if value==owner],owner)
        source_names=[original['collection']]
        if client.collection_exists(entity_name(original['collection'])):source_names.append(entity_name(original['collection']))
        signature=source_digest(client,source_names,memory,records)
        if state['source_sha256'] not in (None,signature):raise RuntimeError('Reindex source changed after work began')
        state['source_sha256']=signature;publish()
        operations={(r['user_id'],r['request_id']):r for r in rows(memory/'operations.sqlite','operations')}
        expected=expected_memories(operations);active={mid:v for mid,v in expected.items() if mid not in barriers}
        counts={};seen=set()
        for src in source_names:
            is_entity=src.endswith('_entities');dst=entity_name(target.collection) if is_entity else target.collection;count=0
            for batch in pages(client,src):
                payloads=[];ids=[]
                for point in batch:
                    mid=str(point.id);payload=dict(point.payload or {});owner=payload.get('user_id')
                    if is_entity:
                        links=payload.get('linked_memory_ids')
                        if not isinstance(links,list) or not isinstance(owner,str):raise RuntimeError('Unsupported entity ownership')
                        if any(link in owners and owners[link]!=owner for link in links):raise RuntimeError('Entity link crosses owners')
                        payload['linked_memory_ids']=[link for link in links if link in active]
                        if not payload['linked_memory_ids']:continue
                    else:
                        if mid in barriers:continue
                        want=active.get(mid)
                        if not want or owner!=want['owner'] or payload.get('version')!=want['version'] or payload.get('operation_key')!=want['request_id']:
                            raise RuntimeError('Source fact differs from its confirmed operation')
                        seen.add(mid)
                    if not isinstance(payload.get('data'),str) or not payload['data']:raise RuntimeError('Source text is missing')
                    ids.append(point.id);payloads.append(payload)
                if not ids:continue
                vectors=profile.embedder.embed_batch([p['data'] for p in payloads],'add')
                if len(vectors)!=len(ids):raise RuntimeError('Reindex embedding batch differs')
                upsert_canonical(client,dst,[models.PointStruct(id=mid,vector={'':vector},payload=payload) for mid,vector,payload in zip(ids,vectors,payloads)],profile.profile_id,entities=is_entity)
                actual={str(p.id):p for p in client.retrieve(dst,ids,with_payload=True,with_vectors=True)}
                for mid,vector,payload in zip(ids,vectors,payloads):
                    returned=actual.get(str(mid))
                    if returned is None or returned.payload!=payload or not matching(returned.vector,vector,profile.dimensions):raise RuntimeError('Reindex target verification failed')
                count+=len(ids);checkpoint('batch')
            if client.count(dst,exact=True).count!=count:raise RuntimeError('Reindex target contains unexpected points')
            counts['entities' if is_entity else 'memories']=count
        if seen!=set(active):raise RuntimeError('A confirmed fact is unavailable in the reindex source')
        if source_digest(client,source_names,memory,records)!=signature:raise RuntimeError('Reindex source changed during copying')
        state['result']={'status':'PASS','already_completed':False,'counts':counts,'profile':profile.profile_id,'embedding_scope':profile.scope,
            'scope':'Same canonical facts/versions/sources and current controls; native vectors rebuilt under an exclusive fence, retired index removed.'}
        state['status']='VERIFIED';publish();checkpoint('verified')
        target.ready()
        previous=state['previous_data_state'];updated=target_data_state(state)
        if read_state(directory) not in (previous,updated):raise RuntimeError('Application lifecycle state changed during reindex')
        atomic_json(directory/MARKER,updated)
        state['status']='PUBLISHED';publish();checkpoint('published')
        cleanup_collections(client,original);checkpoint('retired')
        state['status']='COMPLETED';publish();return state['result']


def abort(directory):
    directory=Path(directory).resolve();memory=directory/'memory';marker=memory/MARK
    with ExitStack() as stack:
        data_locks(stack,directory);state=json.loads(marker.read_text())
        if state.get('schema_version')!=1 or state.get('directory')!=str(directory) or state.get('status') not in {'WRITING','VERIFIED','ABORTING'}:raise RuntimeError('Published reindex cannot be aborted')
        original=state['original_binding'];target=state['binding'];service=load(original['service_directory'])
        validate_binding(original,memory,service);validate_binding(target,memory,service)
        lock(stack,service.directory/'leases'/(original['collection']+'.lock'));lock(stack,service.directory/'leases'/(target['collection']+'.lock'))
        client=stack.enter_context(closing(service.client(timeout=60)));validate_source(client,original)
        if json.loads((memory/'store.json').read_text()) not in [original,target,dict(target,status='READY')]:raise RuntimeError('Reindex binding changed before abort')
        records=directory/'records.sqlite';records=records if records.exists() else None
        names=[original['collection']]
        if client.collection_exists(entity_name(original['collection'])):names.append(entity_name(original['collection']))
        if state['source_sha256'] and source_digest(client,names,memory,records)!=state['source_sha256']:raise RuntimeError('Source changed; abort requires explicit recovery')
        previous=state['previous_data_state'];current=read_state(directory)
        if current not in (previous,target_data_state(state)):raise RuntimeError('Application state changed before abort')
        state['status']='ABORTING';atomic_json(marker,state);cleanup_collections(client,target)
        atomic_json(memory/'store.json',original)
        if previous:atomic_json(directory/MARKER,previous)
        elif (directory/MARKER).exists():(directory/MARKER).unlink();sync_directory(directory)
        if state.get('previous_reindex_state'):atomic_json(marker,state['previous_reindex_state'])
        else:marker.unlink();sync_directory(memory)
        return {'status':'ABORTED','source_preserved':True}
