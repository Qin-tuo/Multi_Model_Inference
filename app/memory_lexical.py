"""Canonical lexical indexing executed by the pinned local Qdrant server."""
import math

from app.memory_profiles import HYBRID,retrieval_spec


def document(text):
    from qdrant_client import models
    if not isinstance(text,str) or not text:raise ValueError('Canonical memory text is required for lexical indexing')
    spec=retrieval_spec()
    return models.Document(text=text,model=spec['model'],options=spec['options'])


def valid_sparse(vector):
    from qdrant_client import models
    value=vector.get('bm25') if isinstance(vector,dict) else None
    if not isinstance(value,models.SparseVector):return False
    return (len(value.indices)==len(value.values) and len(set(value.indices))==len(value.indices)
        and all(type(x) is int and 0<=x<=2**32-1 for x in value.indices)
        and all(math.isfinite(x) and x>=0 for x in value.values))


def upsert_canonical(client,collection,points,profile,*,entities=False):
    """Recreate lexical vectors from authoritative text, including restoration.

    Snapshot sparse vectors are never copied into an active hybrid fact index.
    The SDK's public REST API sends Document directly to the local server and
    avoids client-side FastEmbed discovery. Existing authenticated client reused.
    """
    from qdrant_client import models
    if not points:return
    if profile!=HYBRID or entities:
        client.upsert(collection,points,wait=True)
        return
    from deploy.memory.migrate import matching
    encoded=[]
    for point in points:
        dense=point.vector.get('') if isinstance(point.vector,dict) else point.vector
        encoded.append(models.PointStruct(id=point.id,payload=point.payload,
            vector={'':dense,'bm25':document((point.payload or {}).get('data'))}))
    result=client.http.points_api.upsert_points(collection_name=collection,wait=True,
        point_insert_operations=models.PointsList(points=encoded)).result
    if result is None or result.status!=models.UpdateStatus.COMPLETED:raise RuntimeError('Native lexical write has not completed')
    returned={str(p.id):p for p in client.retrieve(collection,[p.id for p in points],with_payload=True,with_vectors=True)}
    for point in points:
        actual=returned.get(str(point.id))
        dense=point.vector.get('') if isinstance(point.vector,dict) else point.vector
        dimensions=len(dense[0]) if dense and isinstance(dense[0],list) else len(dense)
        if (actual is None or actual.payload!=point.payload or not valid_sparse(actual.vector)
                or not matching(actual.vector,point.vector,dimensions)):
            raise RuntimeError('Canonical dense/lexical write verification failed')
