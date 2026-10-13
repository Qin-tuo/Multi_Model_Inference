"""Isolated capacity-fixture import; never a production memory write path.

Uses the pinned Mem0 Qdrant/history providers and real BGE batch inference.
Only fixture preparation skips repeated conflict scans of a growing collection.
Timed transactions in memory_capacity.py always call the full production adapter.
"""
from datetime import datetime, timezone
import hashlib
import json
import uuid

from mem0.utils.lemmatization import lemmatize_for_bm25
from tools.project import meminfo


def populate(memory,corpus,profile,total,tier,start,fact,checkpoint):
    marker=memory.data_dir.parent/'capacity-run.json'
    if not marker.is_file() or json.loads(marker.read_text()).get('seed_method')!='sdk-batch':
        raise ValueError('Batch population requires a dedicated SDK capacity-fixture directory')
    if corpus['namespace']!='capacity-v1' or corpus[profile] not in {'capacity-owner','capacity-guest'}:
        raise ValueError('Batch import is restricted to synthetic capacity profiles')
    user=corpus[profile]
    generation=memory.generation(user)
    if generation!=0: raise ValueError('Fixture population cannot overwrite maintenance history')
    with memory._gate:
        for begin in range(start,total,32):
            if meminfo()['MemAvailable']<700*1024*1024: raise RuntimeError('CAPACITY_BOUNDARY: less than700MiB available')
            batch=[]
            for index in range(begin,min(begin+32,total)):
                content=fact(index if profile=='owner' else int((index+.5)*tier/corpus['queries']))
                key=f'project-{index:05d}';source=f'synthetic-{profile}-{index}';request=f'seed-{profile}-{index}'
                arguments={'type':'fact','key':key,'content':content,'source_id':source,'source_excerpt':content,
                    'namespace':corpus['namespace'],'expected_generation':None,'derived_from':[]}
                _,prior=memory._start(user,request,'remember',arguments)
                if prior: continue
                memory_id=str(uuid.uuid5(uuid.NAMESPACE_URL,f'assistant-capacity-v1/{tier}/{profile}/{index}'))
                created=datetime.now(timezone.utc).isoformat()
                # Payload layout matches pinned Memory._create_memory; the
                # setup-equivalence probe compares it with ordinary SDK writes.
                payload={'namespace':corpus['namespace'],'memory_type':'fact','key':key,'version':1,
                    'generation':0,'source_id':source,'source_excerpt':content,'operation_key':request,
                    'derived_from':[],'user_id':user,'role':'user','data':content,
                    'hash':hashlib.md5(content.encode()).hexdigest(),'created_at':created,'updated_at':created,
                    'text_lemmatized':lemmatize_for_bm25(content)}
                batch.append((memory_id,payload,request))
            vectors=memory.sdk.embedding_model.embed_batch([p['data'] for _,p,_ in batch],'add')
            if batch:
                memory.sdk.vector_store.insert(vectors=vectors,ids=[i for i,_,_ in batch],payloads=[p for _,p,_ in batch])
                for memory_id,payload,request in batch:
                    if not memory.sdk.history(memory_id):
                        memory.sdk.db.add_history(memory_id,None,payload['data'],'ADD',
                            created_at=payload['created_at'],updated_at=payload['updated_at'],role='user')
                    memory._finish(user,request,{'memory_id':memory_id,'version':1,'generation':0,'durable':True,
                        'replayed':False,'atomicity':'reconciled_non_atomic'})
            checkpoint(min(begin+32,total))
