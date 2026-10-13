"""Strict derived-entity erasure for pinned Mem02.2.1/Qdrant.

Upstream cleanup skips an uninitialized entity store, truncates at10000 rows,
and suppresses errors. Use the existing Qdrant client's public cursor/payload
APIs, including after restart; do not create an alternate entity store.
"""
from qdrant_client.models import FieldCondition,Filter,MatchAny,MatchValue,PointIdsList


def purge_entities(sdk,memory_ids,user_id):
    client=sdk.vector_store.client
    collection=sdk.collection_name+'_entities'
    result={'entity_rows_removed':0,'entity_links_removed':0}
    if not client.collection_exists(collection):return result
    for begin in range(0,len(memory_ids),256):
        removed=set(memory_ids[begin:begin+256])
        condition=Filter(must=[FieldCondition(key='user_id',match=MatchValue(value=user_id)),
            FieldCondition(key='linked_memory_ids',match=MatchAny(any=list(removed)))])
        offset=None
        while True:
            rows,offset=client.scroll(collection,scroll_filter=condition,limit=128,offset=offset,with_vectors=False,with_payload=True)
            for row in rows:
                payload=row.payload or {}
                if payload.get('user_id')!=user_id:raise RuntimeError('Entity owner differs from deletion barrier')
                links=payload.get('linked_memory_ids')
                if not isinstance(links,list) or not all(isinstance(x,str) for x in links):
                    raise RuntimeError('Unsupported entity reference schema; erasure not confirmed')
                kept=[value for value in links if value not in removed]
                result['entity_links_removed']+=len(links)-len(kept)
                if kept:client.set_payload(collection,{'linked_memory_ids':kept},points=[row.id],wait=True)
                else:
                    client.delete(collection,points_selector=PointIdsList(points=[row.id]),wait=True)
                    result['entity_rows_removed']+=1
            if offset is None:break
    return result
