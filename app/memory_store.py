"""Bind a Mem0 journal directory to native Qdrant collection metadata.

Only ownership/configuration lives here. Facts, vectors and indexes remain in
Mem0/Qdrant; the same adapter journal continues to own deletion barriers.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid

from deploy.memory.runtime import atomic_json
from app.memory_profiles import LEGACY,WINDOWS,HYBRID,is_window_profile,retrieval_fingerprint,lexical_layout_matches,profile_of,vector_params,layout_matches,entity_name,entity_metadata


def new_binding(directory, service, embedding_sha256, dimensions=512,profile=LEGACY):
    if profile not in {LEGACY,WINDOWS,HYBRID}:raise ValueError('Unsupported memory embedding profile')
    store_id=uuid.uuid4().hex
    return {'schema_version':1 if profile==LEGACY else (3 if profile==HYBRID else 2),
        **({'embedding_profile':profile} if profile!=LEGACY else {}),
        **({'retrieval_sha256':retrieval_fingerprint()} if profile==HYBRID else {}),'backend':'qdrant_server','store_id':store_id,
        'collection':'assistant_'+store_id,'directory':str(Path(directory).resolve()),'instance_id':service.instance_id,
        'service_directory':str(service.directory),'embedding_sha256':embedding_sha256,'dimensions':dimensions,'status':'PREPARING'}


def check_transition(directory):
    """A crashed migration must never reopen the retired local store."""
    directory=Path(directory)
    reindex=directory/'reindex-transition.json'
    if reindex.exists():
        state=json.loads(reindex.read_text());path=directory/'store.json'
        if (state.get('status')!='COMPLETED' or not path.exists()
                or json.loads(path.read_text())!=dict(state.get('binding',{}),status='READY')):
            raise RuntimeError('Memory reindex is incomplete; resume reindex-memory before opening this store')
        return
    marker=directory/'store-transition.json'
    if not marker.exists():return
    transition=json.loads(marker.read_text())
    path=directory/'store.json'
    if transition.get('status')!='COMPLETED' or not path.is_file():
        raise RuntimeError('Memory migration is incomplete; resume migrate-memory before opening this store')
    binding=json.loads(path.read_text())
    expected=dict(transition['binding'],status='READY')
    if binding!=expected:
        raise RuntimeError('Memory migration binding differs; explicit recovery is required')


def collection_metadata(binding):
    profile=profile_of(binding)
    return {'assistant_store_id':binding['store_id'],'instance_id':binding['instance_id'],
        'directory_sha256':hashlib.sha256(binding['directory'].encode()).hexdigest(),
        'embedding_sha256':binding['embedding_sha256'],'dimensions':binding['dimensions'],
        **({'embedding_profile':profile} if profile!=LEGACY else {}),
        **({'retrieval_sha256':binding['retrieval_sha256']} if profile==HYBRID else {})}


class StoreBinding:
    def __init__(self,directory,service,*,embedding_sha256,dimensions=512,profile=None):
        from qdrant_client import models
        self.directory=Path(directory).resolve();self.service=service
        self.client=self.lease=None
        self.path=self.directory/'store.json'
        try:
            if self.path.exists():
                self.binding=json.loads(self.path.read_text())
                recorded=profile_of(self.binding)
                if ((profile is not None and recorded!=profile) or self.binding.get('backend')!='qdrant_server'
                        or self.binding.get('directory')!=str(self.directory)
                        or self.binding.get('instance_id')!=service.instance_id
                        or self.binding.get('service_directory')!=str(service.directory)
                        or self.binding.get('embedding_sha256')!=embedding_sha256
                        or self.binding.get('dimensions')!=dimensions
                        or self.binding.get('status') not in {'PREPARING','READY'}):
                    raise RuntimeError('Memory store binding differs; explicit migration/restore is required')
            else:
                if (self.directory/'qdrant').exists():raise RuntimeError('Existing embedded memory requires explicit migration')
                journal=self.directory/'operations.sqlite'
                if journal.exists():
                    with sqlite3.connect(journal) as db:
                        if db.execute('SELECT count(*) FROM operations').fetchone()[0]:raise RuntimeError('Existing memory journal requires explicit migration')
                self.binding=new_binding(self.directory,service,embedding_sha256,dimensions,profile or LEGACY)
                atomic_json(self.path,self.binding)
            self.profile=profile_of(self.binding)
            store_id=self.binding['store_id']
            if len(store_id)!=32 or any(x not in '0123456789abcdef' for x in store_id) or self.binding['collection']!='assistant_'+store_id:
                raise RuntimeError('Invalid collection binding')
            self.collection=self.binding['collection']
            self.expected=collection_metadata(self.binding)
            self.lease=(service.directory/'leases'/(self.collection+'.lock')).open('a+b')
            fcntl.flock(self.lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.client=service.client()
            exists=self.client.collection_exists(self.collection)
            if not exists:
                if self.binding['status']=='READY':raise RuntimeError('Bound memory collection is missing; refusing to create an empty replacement')
                self.client.create_collection(self.collection,
                    vectors_config=vector_params(dimensions,self.profile),
                    sparse_vectors_config={'bm25':models.SparseVectorParams(modifier=models.Modifier.IDF)},
                    hnsw_config=models.HnswConfigDiff(m=0),metadata=self.expected)
            if is_window_profile(self.profile):
                entity=entity_name(self.collection)
                if not self.client.collection_exists(entity):
                    if self.binding['status']=='READY':raise RuntimeError('Bound entity collection is missing; explicit recovery is required')
                    self.client.create_collection(entity,vectors_config=vector_params(dimensions,self.profile),
                        sparse_vectors_config={'bm25':models.SparseVectorParams(modifier=models.Modifier.IDF)},
                        hnsw_config=models.HnswConfigDiff(m=0),metadata=entity_metadata(self.binding))
            self.validate()
            for field in ('user_id','namespace','key','memory_type','operation_key','derived_from','source_date','source_month_day','source_speaker'):
                self.client.create_payload_index(self.collection,field,models.PayloadSchemaType.KEYWORD,wait=True)
        except Exception as error:
            self.close()
            if isinstance(error,(RuntimeError,BlockingIOError)):raise
            raise RuntimeError('Native memory service request failed; check the managed backend') from error
        except BaseException:
            self.close();raise

    def validate(self):
        info=self.client.get_collection(self.collection)
        if (info.config.metadata!=self.expected or info.config.hnsw_config.m!=0
                or not layout_matches(info.config.params.vectors,self.binding['dimensions'],self.profile)
                or not lexical_layout_matches(info,self.profile)):
            raise RuntimeError('Native collection metadata/configuration does not match this journal')
        if is_window_profile(self.profile):
            entity=self.client.get_collection(entity_name(self.collection))
            if (entity.config.metadata!=entity_metadata(self.binding) or entity.config.hnsw_config.m!=0
                    or not layout_matches(entity.config.params.vectors,self.binding['dimensions'],self.profile)):
                raise RuntimeError('Native entity collection configuration differs from the memory profile')
        return info

    def ready(self):
        self.validate()
        if self.binding['status']!='READY':
            self.binding['status']='READY';atomic_json(self.path,self.binding)

    def close(self):
        if self.client:self.client.close();self.client=None
        if self.lease:self.lease.close();self.lease=None


def embedding_fingerprint(path):
    """Hash model and tokenizer/pooling inputs, independently of directory name."""
    root=Path(path)
    if not (root/'onnx/model.onnx').is_file() or not (root/'tokenizer.json').is_file():
        raise RuntimeError('Native memory embedding artifacts are incomplete')
    names=['onnx/model.onnx','config.json','modules.json','1_Pooling/config.json',
           'tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt',
           'sentence_bert_config.json','config_sentence_transformers.json']
    digest=hashlib.sha256()
    for name in names:
        p=root/name
        if not p.exists():continue
        digest.update(name.encode())
        with p.open('rb') as stream:
            while block:=stream.read(1024*1024):digest.update(block)
    return digest.hexdigest()
