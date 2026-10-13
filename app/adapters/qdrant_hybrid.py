"""Mem0 Qdrant provider with native local BM25/DBSF candidate retrieval.

Mem0 still creates and owns IDs, payloads, history and entity relationships.
The provider overrides indexing/search only, using Qdrant's existing APIs.
"""
from mem0.vector_stores.qdrant import Qdrant
from qdrant_client import models

from app.memory_lexical import document,upsert_canonical
from app.memory_profiles import HYBRID,retrieval_spec


class NativeHybridQdrant(Qdrant):
    native_hybrid=True

    def __init__(self,upstream):
        # Only this memory instance changes provider. No global factory patch or
        # additional client; the existing bound client retains auth and leases.
        self.__dict__.update(upstream.__dict__)

    def insert(self,vectors,payloads=None,ids=None):
        if payloads is None or ids is None or not len(vectors)==len(payloads)==len(ids):
            raise ValueError('Hybrid writes require complete canonical IDs, bodies and vectors')
        points=[models.PointStruct(id=mid,vector={'':vector},payload=payload) for mid,vector,payload in zip(ids,vectors,payloads)]
        upsert_canonical(self.client,self.collection_name,points,HYBRID)

    def update(self,vector_id,vector=None,payload=None):
        if vector is not None and payload is not None:
            self.insert([vector],[payload],[vector_id])
            return
        # Metadata-only changes must not silently invalidate lexical evidence.
        if payload is not None and any(key in payload for key in ('data','text_lemmatized')):
            raise ValueError('Text changes require a complete dense/lexical update')
        if vector is not None:raise ValueError('Vector changes require a complete canonical payload')
        return super().update(vector_id,vector=vector,payload=payload)

    def search(self,query,vectors,top_k=5,filters=None):
        if not filters or not isinstance(filters.get('user_id'),str) or not isinstance(filters.get('namespace'),str):
            raise ValueError('Hybrid retrieval requires explicit owner and namespace')
        condition=self._create_filter(filters);spec=retrieval_spec()
        request=models.QueryRequest(prefetch=[models.Prefetch(query=vectors,filter=condition,limit=spec['prefetch']),
            models.Prefetch(query=document(query),using=spec['slot'],filter=condition,limit=spec['prefetch'])],
            query=models.FusionQuery(fusion=models.Fusion.DBSF),filter=condition,limit=top_k,with_payload=True)
        result=self.client.http.search_api.query_points(collection_name=self.collection_name,query_request=request).result
        if result is None:raise RuntimeError('Native hybrid query did not return a result')
        # DBSF scores can exceed1 when both branches agree. Mem0 subsequently
        # clamps scores to1, which would erase ordering among these candidates.
        # One shared divisor preserves native order and stays within its scale.
        scale=max(1.0,max((point.score for point in result.points),default=0))
        for point in result.points:point.score/=scale
        return result.points

    def search_batch(self,queries,vectors_list,top_k=1,filters=None):
        if len(queries)!=len(vectors_list):raise ValueError('Hybrid query/vector batch differs')
        return [self.search(query,vector,top_k=top_k,filters=filters) for query,vector in zip(queries,vectors_list)]

    def keyword_search(self,query,top_k=5,filters=None):
        # Keywords already participated in native candidate fusion; Mem0 must
        # not run optional FastEmbed and add the same signal a second time.
        return None
