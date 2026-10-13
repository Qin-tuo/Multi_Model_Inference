"""Pinned representations for the same BGE model and native Qdrant store."""
import json
LEGACY='bge-fp32-single-v1'
WINDOWS='bge-int8-windows-v1'
WINDOW_TOKENS=256
WINDOW_OVERLAP=32


def profile_of(binding):
    if not isinstance(binding,dict) or type(binding.get('schema_version')) is not int:raise RuntimeError('Invalid memory binding schema')
    if binding.get('schema_version')==1:
        if 'embedding_profile' in binding:raise RuntimeError('Legacy binding cannot override its embedding profile')
        return LEGACY
    if binding.get('schema_version')==2 and binding.get('embedding_profile')==WINDOWS:return WINDOWS
    raise RuntimeError('Unknown memory binding/profile; explicit migration is required')


def model_path(profile):
    from tools.project import CACHE,ROOT
    if profile==LEGACY:return CACHE/'models/bge-small-zh-v1.5-onnx'
    if profile==WINDOWS:
        from app.memory_store import embedding_fingerprint
        declared=json.loads((ROOT/'models/manifest.json').read_text())['memory_profiles'][profile]
        if (declared['window_tokens']!=WINDOW_TOKENS or declared['window_overlap']!=WINDOW_OVERLAP
                or declared['model_directory']!='bge-small-zh-v1.5-onnx-arm64-int8'):
            raise RuntimeError('Window profile differs from its deployment lock')
        path=CACHE/'models'/declared['model_directory']
        if embedding_fingerprint(path)!=declared['embedding_sha256']:raise RuntimeError('Window embedding artifacts differ from the deployment lock')
        return path
    raise ValueError('Unsupported memory embedding profile')


def vector_params(dimensions,profile):
    from qdrant_client import models
    if profile not in {LEGACY,WINDOWS}:raise ValueError('Unsupported memory embedding profile')
    extra={'multivector_config':models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM)} if profile==WINDOWS else {}
    return models.VectorParams(size=dimensions,distance=models.Distance.COSINE,on_disk=True,**extra)


def layout_matches(params,dimensions,profile):
    from qdrant_client import models
    if not isinstance(params,models.VectorParams) or params.size!=dimensions or params.distance!=models.Distance.COSINE:return False
    multi=params.multivector_config
    return multi is None if profile==LEGACY else profile==WINDOWS and multi is not None and multi.comparator==models.MultiVectorComparator.MAX_SIM


def entity_name(collection):
    # Pinned Mem0 2.2.1 Qdrant naming; no SDK import before telemetry setup.
    return collection+'_entities'


def entity_metadata(binding):
    from app.memory_store import collection_metadata
    return dict(collection_metadata(binding),role='entities')
