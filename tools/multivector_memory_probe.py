"""Isolated Mem0/Qdrant multivector feasibility with the existing real BGE model.

This is not the production adapter or a migration. One point retains the full
record; Qdrant MaxSim searches token-window embeddings under the owner filter.
"""
from contextlib import closing
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.memory import build_mem0_config
from app.adapters.embedding import BGEQueryAdapter
from app.context_memory import memory_message
from deploy.memory.runtime import load
from tools.long_memory_probe import documents
from tools.project import CACHE,DATA,stamp,write_json


class WindowEmbedding:
    def __init__(self,upstream):
        self.base=BGEQueryAdapter(upstream);self.model=self.base.model;self.config=upstream.config

    def spans(self,text):
        offsets=self.model.tokenizer(text,add_special_tokens=False,truncation=False,return_offsets_mapping=True)['offset_mapping']
        if len(offsets)<=self.model.max_seq_length-2:return [(0,len(text))]
        spans=[]
        for offset in range(0,len(offsets),224):
            end=min(offset+256,len(offsets))
            spans.append((0 if offset==0 else offsets[offset][0],len(text) if end==len(offsets) else offsets[end][0]))
            if end==len(offsets):break
        return spans

    def embed(self,text,memory_action=None):
        if memory_action=='search':return [self.base.embed(text,'search')]
        return self.model.encode([text[start:end] for start,end in self.spans(text)],
            convert_to_numpy=True,normalize_embeddings=True).tolist()

    def embed_batch(self,texts,memory_action='add'):
        return [self.embed(text,memory_action) for text in texts]


def main(args):
    import numpy as np
    import onnxruntime as ort
    from mem0 import Memory
    from mem0.memory.main import _entity_collection_name
    from qdrant_client import models
    owner='synthetic-multivector';name='probe_multivectors_'+uuid.uuid4().hex;service=load(DATA/'qdrant')
    inputs=documents();rows=[];sdk=None
    entity_name=_entity_collection_name('qdrant',name)
    with tempfile.TemporaryDirectory(prefix='multivector-memory-') as temporary,closing(service.client(timeout=60)) as client:
        path=Path(temporary)
        client.create_collection(name,vectors_config=models.VectorParams(size=512,distance=models.Distance.COSINE,on_disk=True,
            multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM)),
            hnsw_config=models.HnswConfigDiff(m=0),metadata={'probe':'long-document-windows'})
        try:
            # Mem0 shares its embedder with its auxiliary entity index. Both
            # stores must accept the same matrix shape; an unconfigured entity
            # store would otherwise be created with a single-vector schema.
            client.create_collection(entity_name,vectors_config=models.VectorParams(size=512,distance=models.Distance.COSINE,on_disk=True,
                multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM)),
                hnsw_config=models.HnswConfigDiff(m=0),metadata={'probe':'long-document-entities'})
            config=build_mem0_config(path,args.embedding_path,embedding_backend='onnx',qdrant_url=service.url,qdrant_collection=name)
            config['vector_store']['config']['client']=client
            options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
            config['embedder']['config']['model_kwargs']['model_kwargs']['session_options']=options
            sdk=Memory.from_config(config);encoder=WindowEmbedding(sdk.embedding_model);sdk.embedding_model=encoder
            for index,document in enumerate(inputs):
                text=document['content'];started=time.monotonic()
                result=sdk.add([{'role':'user','content':text}],user_id=owner,infer=False,metadata={
                    'namespace':'personal','source_id':f'synthetic-{index}','source_excerpt':'请记住，'+text,'version':1})
                mid=result['results'][0]['id'];stored=sdk.get(mid)
                rows.append({'project':document['project'],'memory_id':mid,'content_sha256':hashlib.sha256(text.encode()).hexdigest(),
                    'full_text_preserved':stored['memory']==text,'windows':len(encoder.spans(text)),'write_seconds':time.monotonic()-started})
            # A different owner deliberately has the same highly relevant text.
            sdk.add([{'role':'user','content':inputs[0]['content']}],user_id='synthetic-guest',infer=False,metadata={'namespace':'personal'})
            for row,document in zip(rows,inputs):
                question=f'{row["project"]}项目的最终交付箱编号是什么？';started=time.monotonic()
                found=sdk.search(question,filters={'user_id':owner,'namespace':'personal'},top_k=5,rerank=False)['results']
                row['search_seconds']=time.monotonic()-started
                row['retrieved_count']=len(found)
                row['top1_correct']=bool(found) and found[0]['id']==row['memory_id']
                row['top5_correct']=any(item['id']==row['memory_id'] for item in found)
                row['owner_isolation']=all(sdk.get(item['id'])['user_id']==owner for item in found)
                point=client.retrieve(name,ids=[row['memory_id']],with_vectors=True)[0]
                vectors=point.vector[''] if isinstance(point.vector,dict) else point.vector
                scores=np.asarray(vectors)@np.asarray(encoder.embed(question,'search')[0]);best=int(scores.argmax())
                start,end=encoder.spans(document['content'])[best];excerpt=document['content'][start:end]
                item={'memory_id':row['memory_id'],'version':1,'content':excerpt,'sources':[{
                    'source_id':sdk.get(row['memory_id'])['metadata']['source_id'],'evidence':excerpt,
                    'span':{'field':'memory','start':start,'end':end},'speaker':'user'}]}
                _,packed=memory_message([item],'http://127.0.0.1:8088/v1',compact_references=True)
                row.update(selected_span=[start,end],span_is_exact_original_substring=True,
                    answer_in_selected_span=row['project']+'七号' in excerpt,
                    evidence_tokens=packed['memory_tokens'],selected_span_fits=packed['injected_count']==1)
            target=rows[0]['memory_id'];revised=inputs[0]['content'].replace('海岚七号','海岚九号')
            prior=client.retrieve(name,ids=[target],with_vectors=True)[0]
            old_vectors=prior.vector[''] if isinstance(prior.vector,dict) else prior.vector
            sdk.update(target,revised,metadata={'version':2,'source_id':'synthetic-correction','source_excerpt':'更正为：'+revised})
            updated=sdk.get(target);after=client.retrieve(name,ids=[target],with_vectors=True)[0]
            updated_vectors=after.vector[''] if isinstance(after.vector,dict) else after.vector
            checks={'one_point_per_record':client.count(name,exact=True).count==9,
                'all_original_texts_preserved':all(r['full_text_preserved'] for r in rows),
                'all_targets_top1':all(r['top1_correct'] for r in rows),
                'all_targets_top5':all(r['top5_correct'] for r in rows),
                'top5_bound_applied':all(r['retrieved_count']<=5 for r in rows),
                'owner_filter_before_search':all(r['owner_isolation'] for r in rows),
                'matching_verbatim_spans_fit512':all(r['answer_in_selected_span'] and r['selected_span_fits'] for r in rows),
                'update_keeps_identity_and_full_content':updated['id']==target and updated['memory']==revised and updated['metadata']['version']==2,
                'update_replaces_source':updated['metadata']['source_id']=='synthetic-correction' and updated['metadata']['source_excerpt']=='更正为：'+revised,
                'update_replaces_matrix':np.allclose(updated_vectors,encoder.embed(revised,'update'),atol=1e-5) and not np.allclose(updated_vectors,old_vectors,atol=1e-5)}
            # Explicitly exercise the SDK entity path without claiming an NLP
            # entity extractor was enabled for these synthetic documents.
            sdk._upsert_entity('海岚','PROJECT',target,{'user_id':owner})
            entities=sdk.entity_store.search('海岚',encoder.embed('海岚','search'),top_k=1,filters={'user_id':owner})
            checks['entity_index_accepts_matrix']=len(entities)==1 and target in entities[0].payload.get('linked_memory_ids',[])
            checks['entity_owner_filter']=not sdk.entity_store.search('海岚',encoder.embed('海岚','search'),top_k=1,filters={'user_id':'synthetic-guest'})
            sdk.delete(target);checks['delete_removes_entire_point']=not client.retrieve(name,ids=[target],with_vectors=True) and client.count(name,exact=True).count==8
            checks['loaded_entity_link_erased']=client.count(entity_name,exact=True).count==0
        finally:
            if sdk is not None:
                sdk.db.close()
                llm_client=getattr(sdk.llm,'client',None)
                if llm_client is not None:llm_client.close()
            for collection in client.get_collections().collections:
                if collection.name==name or collection.name.startswith(name+'_'):client.delete_collection(collection.name)
    report={'created_at':stamp(),'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'cases':rows,
        'embedding_path':str(args.embedding_path.resolve()),'embedding_sha256':hashlib.sha256((args.embedding_path/'onnx/model.onnx').read_bytes()).hexdigest(),
        'schema':'One original Mem0 memory per Qdrant point;BGE token windows stored as native512-dimensional MaxSim multivectors. Query is one BGE query vector.',
        'scope':'Isolated feasibility experiment through actual Mem0 SDK and authenticated Qdrant. Synthetic long documents and real BGE/native tokenizer; no new reranker. Single-point CRUD,span packing and explicitly populated loaded entity index. Production binding,journal,history erasure,restart recovery,backup/migration,latency and full language quality are NOT qualified.'}
    write_json(args.output,report)
    print(json.dumps(report,ensure_ascii=False,indent=2));return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embedding-path',type=Path,default=CACHE/'models/bge-small-zh-v1.5-onnx')
    parser.add_argument('--output',type=Path,default=ROOT/'benchmarks/results/multivector-memory-feasibility.json')
    raise SystemExit(main(parser.parse_args()))
