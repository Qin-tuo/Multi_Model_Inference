"""Register every selected model artifact/engine without loading duplicate models.

Hashes prove artifact identity and configuration binding, not inference quality.
Live native properties, process mappings and previous execution evidence are
reported separately. Inactive engine variants are explicitly identified.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.memory_profiles import model_path
from app.memory_store import embedding_fingerprint
from deploy import services
from tools.gpu_concurrency_probe import inventory
from tools.project import CACHE, DATA, request_json, sha256, stamp, write_json


def artifact(path, *, expected=None, role=None):
    path=Path(path)
    digest=sha256(path)
    if expected is not None and digest != expected:
        raise RuntimeError('Artifact differs from declared hash: '+str(path))
    return {'path':str(path.resolve()),'configured_path':str(path),'sha256':digest,
        'bytes':path.stat().st_size,'role':role,
        'verification':'matches prior lock' if expected is not None else 'observed registration hash'}


def evidence(name):
    path=ROOT/'benchmarks/results'/name
    row=json.loads(path.read_text())
    return {'path':str(path.relative_to(ROOT)),'sha256':sha256(path),'status':row.get('status'),
        'scope':row.get('scope','See original report scope and checks')}


def main(args):
    if args.output.exists():
        raise RuntimeError('Use a new inventory report path')
    manifest=json.loads((ROOT/'models/manifest.json').read_text())
    config=json.loads((ROOT/'configs/runtime.yaml').read_text())
    before=services.state(list(services.ORDER),health=True)
    if before['backend_status']!='READY':
        raise RuntimeError('Inventory requires all managed backends ready')
    report={'created_at':stamp(),'status':'RUNNING','scope':__doc__,
        'source_sha256':sha256(Path(__file__)),
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'runtime_sha256':sha256(ROOT/'configs/runtime.yaml'),
        'model_manifest_sha256':sha256(ROOT/'models/manifest.json'),
        'functional_model_count':7,'model_counting':'VLM, ASR, TTS chain (Matcha+Vocos), detection, pose, embedding, VAD. TTS includes two neural architectures.',
        'models':{},'managed_backends':before}
    models=report['models']
    traced=inventory()
    for key,component,backend in [('vlm','llama','llama.cpp CUDA'),('detection','detection','ORT TensorRT'),('pose','pose','ORT TensorRT')]:
        models[key]={'backend':backend,'device':'GPU','artifacts':[r for r in traced if r['component']==component]}
    for key,port in [('vlm',8088),('asr',8089)]:
        entry=manifest['models'][key]
        files=[artifact(CACHE/'models'/item['name'],expected=item['sha256'],role='projector' if item['name'].startswith('mmproj') else 'model') for item in entry['files']]
        props=request_json(f'http://127.0.0.1:{port}/props',timeout=5)
        model=next(r for r in files if r['role']=='model')
        if Path(props['model_path']).resolve()!=Path(model['path']):
            raise RuntimeError('Native endpoint model differs from selected files: '+key)
        models[key]={'backend':'llama.cpp CUDA','device':'GPU','artifacts':files,
            'native_properties':{k:props[k] for k in ('model_path','model_alias','modalities','total_slots','build_info') if k in props},
            'execution_evidence':evidence('llama-smoke.json' if key=='vlm' else 'asr-probe.json')}
    for key in ('detection','pose'):
        params_path=DATA/'config'/f'ros-{key}.json'
        params=next(iter(json.loads(params_path.read_text()).values()))['ros__parameters']
        if (Path(params['detection.model_path']).resolve()!=
                (CACHE/'models/yolo'/manifest['models'][key]['onnx']['name']).resolve()
                or params['detection.provider']!='tensorrt' or params['detection.trt_fp16_enable']):
            raise RuntimeError('Live ROS configuration differs from qualified profile')
        models[key].update(runtime_parameters=artifact(params_path),precision=manifest['models'][key]['precision'],
            execution_evidence=evidence(f'vision-{key}-provider.json'))

    speech=Path(CACHE/'models/matcha-host')
    host=json.loads((speech/'host-manifest.json').read_text())
    profile=json.loads((DATA/'config/ovs-host-profile.json').read_text())
    if (Path(profile['env']['MATCHA_MODEL_BASE']).resolve()!=speech.resolve()
            or profile['env']['MATCHA_ACOUSTIC_EP']!='SPLIT_TRT'
            or Path(profile['env']['MATCHA_SPLIT_ENCODER_ONNX']).resolve()!=(speech/'onnx/matcha_encoder_trt.onnx').resolve()
            or Path(profile['env']['MATCHA_SPLIT_ESTIMATOR_ENGINE']).resolve()!=(speech/'engines/matcha_estimator_step0_bf16.engine').resolve()
            or Path(profile['env']['VOCOS_ENGINE']).resolve()!=(speech/'engines/vocos_fp16.engine').resolve()):
        raise RuntimeError('OVS selected artifacts differ from host build')
    engines=[]
    for name,item in host['engines'].items():
        engines.append(dict(artifact(speech/'engines'/name,expected=item['engine_sha256'],role='TensorRT engine'),
            build_identity=item['identity']))
    if len(engines)!=4:
        raise RuntimeError('Expected three Matcha steps and one Vocos engine')
    models['tts']={'backend':'OVS/voxedge Matcha split TensorRT + Vocos TensorRT',
        'device':'CPU encoder; GPU three estimator steps and Vocos', 'tensorrt':host['tensorrt'],
        'artifacts':[artifact(speech/'onnx/matcha_encoder_trt.onnx',role='CPU ONNX encoder'),*engines],
        'support_files':[artifact(speech/name,role='TTS text processing') for name in ('tokens.txt','lexicon.txt','date-zh.fst','phone-zh.fst','number-zh.fst')],
        'host_manifest':artifact(speech/'host-manifest.json'),
        'execution_evidence':evidence('tts-smoke.json')}
    selected=config['memory']['embedding_profile']
    directory=model_path(selected)
    exported=json.loads((ROOT/'benchmarks/results/bge-int8-export.json').read_text())
    models['embedding']={'backend':'Mem0/Sentence Transformers ORT CPU','device':'CPU',
        'profile':selected,'embedding_threads':config['memory']['embedding_threads'],
        'binding_fingerprint':embedding_fingerprint(directory),
        'artifacts':[artifact(directory/'onnx/model.onnx',expected=exported['output_sha256'],role='dynamic int8 embedding')],
        'execution_evidence':evidence('embedding-threads4-public-voice.json'),
        'lifecycle':'Loaded by assistant processes on demand, not a standalone resident service'}
    vad=importlib.metadata.distribution('pipecat-ai').locate_file('pipecat/audio/vad/data/silero_vad.onnx')
    models['vad']={'backend':'Pipecat Silero ORT CPU','device':'CPU',
        'artifacts':[artifact(vad,expected=manifest['models']['vad']['artifact_sha256'],role='VAD')],
        'execution_evidence':evidence('voice-assistant-smoke.json'),
        'lifecycle':'Loaded by voice application processes on demand'}
    active_engines={r['path'] for row in models.values() for r in row['artifacts'] if r['path'].endswith('.engine')}
    report['active_engine_count']=len(active_engines)
    report['inactive_engine_variants']=[artifact(p,role='not selected by runtime')
        for p in sorted((CACHE/'models').rglob('*.engine')) if str(p.resolve()) not in active_engines]
    libraries={}
    for name,row in before['components'].items():
        values=set()
        for pid in row['cgroup_pids']:
            try:
                for line in Path(f'/proc/{pid}/maps').read_text().splitlines():
                    if any(x in line for x in ('libnvinfer','libcudart','libonnxruntime','libggml-cuda')):
                        values.add(line.split(None,5)[-1])
            except FileNotFoundError:
                pass
        libraries[name]=sorted(values)
    report['observed_loaded_libraries']=libraries
    report['environment']={'l4t':Path('/etc/nv_tegra_release').read_text().strip(),
        'architecture':subprocess.check_output(['uname','-m'],text=True).strip(),
        'upstreams_lock':artifact(ROOT/'deploy/upstreams.lock.json'),
        'vision_source_lock':artifact(ROOT/'deploy/vision/source-lock.json')}
    after=services.state(list(services.ORDER),health=True)
    checks={'seven_model_roles':len(models)==7,'six_selected_tensorrt_engines':len(active_engines)==6,
        'all_backends_ready':after['backend_status']=='READY',
        'same_service_pids':all(before['components'][k]['main_pid']==after['components'][k]['main_pid'] for k in services.ORDER),
        'runtime_unchanged':report['runtime_sha256']==sha256(ROOT/'configs/runtime.yaml'),
        'model_manifest_unchanged':report['model_manifest_sha256']==sha256(ROOT/'models/manifest.json')}
    report.update(checks=checks,status='PASS' if all(checks.values()) else 'FAIL',completed_at=stamp())
    write_json(args.output,report)
    print(json.dumps({'status':report['status'],'checks':checks,'active_engines':len(active_engines),
        'inactive_engines':len(report['inactive_engine_variants'])}),flush=True)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(main(parser.parse_args()))
