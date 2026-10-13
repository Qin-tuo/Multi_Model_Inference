"""BGE token windows for Qdrant's native single-point MaxSim representation."""
import hashlib

from app.adapters.embedding import BGEQueryAdapter
from app.memory_profiles import WINDOW_TOKENS,WINDOW_OVERLAP


class BGEWindowAdapter(BGEQueryAdapter):
    def __init__(self,upstream):
        super().__init__(upstream)
        if not getattr(self.model.tokenizer,'is_fast',False):raise ValueError('Exact window offsets require the pinned fast tokenizer')
        if self.model.max_seq_length<WINDOW_TOKENS+2:raise ValueError('Embedding model cannot encode configured token windows')
        self._query=None

    def spans(self,text):
        offsets=self.model.tokenizer(text,add_special_tokens=False,truncation=False,return_offsets_mapping=True,verbose=False)['offset_mapping']
        if len(offsets)<=WINDOW_TOKENS:return [(0,len(text))]
        spans=[]
        for offset in range(0,len(offsets),WINDOW_TOKENS-WINDOW_OVERLAP):
            end=min(offset+WINDOW_TOKENS,len(offsets))
            start=0 if offset==0 else offsets[offset][0]
            stop=len(text) if end==len(offsets) else offsets[end][0]
            if not 0<=start<stop<=len(text):raise ValueError('Invalid tokenizer source offsets')
            spans.append((start,stop))
            if end==len(offsets):break
        return spans

    def query_vector(self,text):
        if self._query is None or self._query[0]!=text:
            vector=self.model.encode([self.query_instruction+text],convert_to_numpy=True,normalize_embeddings=True)[0].tolist()
            self._query=(text,vector)
        return self._query[1]

    def embed(self,text,memory_action=None):
        return self.embed_batch([text],memory_action)[0]

    def embed_batch(self,texts,memory_action='add'):
        if memory_action=='search':return [[self.query_vector(text)] for text in texts]
        if not texts:return []
        # Dynamic int8 activation ranges depend on the batch. Keep a record's
        # windows together and independent of other records, so add/update,
        # reindex and restore all reproduce the same vector representation.
        result=[]
        for text in texts:
            pieces=[text[start:end] for start,end in self.spans(text)]
            result.append(self.model.encode(pieces,batch_size=16,convert_to_numpy=True,normalize_embeddings=True).tolist())
        return result

    def best_window(self,text,vectors,query):
        import numpy as np
        spans=self.spans(text)
        values=np.asarray(vectors,dtype=np.float32);needle=np.asarray(self.query_vector(query),dtype=np.float32)
        if values.shape!=(len(spans),len(needle)) or not np.isfinite(values).all():raise ValueError('Stored window vectors do not match source text')
        start,end=spans[int((values@needle).argmax())]
        return {'start':start,'end':end,'unit':'unicode_codepoints','text':text[start:end],
            'content_sha256':hashlib.sha256(text.encode()).hexdigest()}
