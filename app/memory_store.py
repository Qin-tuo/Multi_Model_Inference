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


class StoreBinding:
    def __init__(self,directory,service,*,embedding_sha256,dimensions=512):
        from qdrant_client import models
        self.directory=Path(directory).resolve();self.service=service
        self.client=self.lease=None
        self.path=self.directory/'store.json'
        try:
            if self.path.exists():
                self.binding=json.loads(self.path.read_text())
                if (self.binding.get('schema_version')!=1 or self.binding.get('backend')!='qdrant_server'
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
                store_id=uuid.uuid4().hex
                self.binding={'schema_version':1,'backend':'qdrant_server','store_id':store_id,
                    'collection':'assistant_'+store_id,'directory':str(self.directory),'instance_id':service.instance_id,
                    'service_directory':str(service.directory),'embedding_sha256':embedding_sha256,'dimensions':dimensions,'status':'PREPARING'}
                atomic_json(self.path,self.binding)
            store_id=self.binding['store_id']
            if len(store_id)!=32 or any(x not in '0123456789abcdef' for x in store_id) or self.binding['collection']!='assistant_'+store_id:
                raise RuntimeError('Invalid collection binding')
            self.collection=self.binding['collection']
            self.expected={'assistant_store_id':store_id,'instance_id':service.instance_id,
                'directory_sha256':hashlib.sha256(str(self.directory).encode()).hexdigest(),
                'embedding_sha256':embedding_sha256,'dimensions':dimensions}
            self.lease=(service.directory/'leases'/(self.collection+'.lock')).open('a+b')
            fcntl.flock(self.lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.client=service.client()
            exists=self.client.collection_exists(self.collection)
            if not exists:
                if self.binding['status']=='READY':raise RuntimeError('Bound memory collection is missing; refusing to create an empty replacement')
                self.client.create_collection(self.collection,
                    vectors_config=models.VectorParams(size=dimensions,distance=models.Distance.COSINE,on_disk=True),
                    sparse_vectors_config={'bm25':models.SparseVectorParams(modifier=models.Modifier.IDF)},
                    hnsw_config=models.HnswConfigDiff(m=0),metadata=self.expected)
            self.validate()
            for field in ('user_id','namespace','key','memory_type','operation_key','derived_from'):
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
                or info.config.params.vectors.size!=self.binding['dimensions']):
            raise RuntimeError('Native collection metadata/configuration does not match this journal')
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
