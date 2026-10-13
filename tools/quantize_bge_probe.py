"""Export a separate ARM64 int8 BGE candidate using Sentence Transformers."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.project import CACHE,stamp,write_json


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source=(CACHE/'models/bge-small-zh-v1.5-onnx').resolve()
    target=(CACHE/'models/bge-small-zh-v1.5-onnx-arm64-int8').resolve()
    marker=target/'probe-export.json';source_sha=sha(source/'onnx/model.onnx')
    if marker.exists():
        prior=json.loads(marker.read_text())
        if prior['source_sha256']!=source_sha or sha(target/'onnx/model.onnx')!=prior['output_sha256']:
            raise RuntimeError('Existing candidate differs from recorded export')
        print(json.dumps({'status':'EXISTS','path':str(target)}));return 0
    if target.exists():raise RuntimeError('Candidate directory exists without completed provenance; use explicit recovery')
    target.mkdir(parents=True,mode=0o700)
    import onnxruntime as ort
    from sentence_transformers import SentenceTransformer,export_dynamic_quantized_onnx_model
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    model=SentenceTransformer(str(source),backend='onnx',device='cpu',local_files_only=True,
        model_kwargs={'provider':'CPUExecutionProvider','file_name':'onnx/model.onnx','session_options':options})
    started=time.monotonic()
    export_dynamic_quantized_onnx_model(model,'arm64',str(target),push_to_hub=False,file_suffix='qint8_arm64')
    for path in source.iterdir():
        if path.name=='onnx':continue
        if path.is_dir():shutil.copytree(path,target/path.name)
        elif path.is_file():shutil.copy2(path,target/path.name)
    output=target/'onnx/model_qint8_arm64.onnx'
    if not output.is_file():raise RuntimeError('Upstream exporter did not produce the expected model')
    (target/'onnx/model.onnx').symlink_to(output.name)
    if sha(source/'onnx/model.onnx')!=source_sha:raise RuntimeError('Source model changed during export')
    report={'created_at':stamp(),'status':'EXPORTED_NOT_DEPLOYED','source':str(source),'destination':str(target),
        'source_sha256':source_sha,'output_sha256':sha(output),'source_bytes':(source/'onnx/model.onnx').stat().st_size,
        'output_bytes':output.stat().st_size,'export_seconds':time.monotonic()-started,'quantization_config':'arm64 dynamic int8',
        'packages':{p:importlib.metadata.version(p) for p in ['sentence-transformers','onnxruntime','optimum-onnx','onnx']},
        'scope':'Separate local candidate through the maintained Sentence Transformers exporter. No production model/config/store changed; no quality or speed acceptance yet.'}
    write_json(marker,report);write_json(ROOT/'benchmarks/results/bge-int8-export.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
