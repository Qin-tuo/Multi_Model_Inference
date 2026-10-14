"""Run a frozen public visual quality comparison using maintained frameworks."""
import argparse
from contextlib import redirect_stdout
import io
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.project import CACHE,sha256,stamp,write_json

VISION_PY=Path('/home/nvidia/AI_Stack/environments/vision/bin/python')
EXECUTABLE=Path('/home/nvidia/AI_Stack/builds/vision-eval/vision-eval')
PROTOCOL=ROOT/'benchmarks/manifests/vision-quality-v2.json'


def reference(args):
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(2)
    manifest=json.loads((ROOT/'models/manifest.json').read_text())['models'][args.task]
    checkpoint=CACHE/'models/yolo'/manifest['checkpoint']['name']
    if sha256(checkpoint)!=manifest['checkpoint']['sha256']:raise RuntimeError('Checkpoint drift')
    if args.predictions.exists():raise RuntimeError('Preserve existing predictions')
    model=YOLO(str(checkpoint))
    if next(model.model.parameters()).dtype!=torch.float32:raise RuntimeError('Reference must be FP32')
    with args.inputs.open() as inputs,args.predictions.open('w') as output:
        for count,line in enumerate(inputs,1):
            image_id,path=line.rstrip('\n').split('\t')
            started=time.monotonic_ns()
            result=model.predict(path,imgsz=640,rect=False,conf=.5,iou=.45,max_det=100,
                device='cpu',half=False,verbose=False,save=False,batch=1)[0]
            values=[]
            boxes=result.boxes.data.cpu().numpy()
            points=result.keypoints.data.cpu().numpy() if result.keypoints is not None else None
            for index,box in enumerate(boxes):
                x1,y1,x2,y2,score,category=map(float,box)
                row={'class_id':int(category),'score':score,'bbox':[x1,y1,x2-x1,y2-y1]}
                if points is not None:row['keypoints']=points[index].reshape(-1).tolist()
                values.append(row)
            record={'image_id':int(image_id),'elapsed_ms':(time.monotonic_ns()-started)/1e6,'predictions':values}
            output.write(json.dumps(record,allow_nan=False)+'\n');output.flush()
            if count%100==0:print(json.dumps({'completed':count,'backend':'torch','task':args.task}),flush=True)
    return 0


def score(args):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if args.output.exists():raise RuntimeError('Preserve previous metrics')
    data=json.loads(args.dataset.read_text())
    annotation=Path(data['annotations'][args.task]['path'])
    if sha256(annotation)!=data['annotations'][args.task]['sha256']:raise RuntimeError('Annotation drift')
    output_log=io.StringIO()
    with redirect_stdout(output_log):gt=COCO(str(annotation))
    ids=[int(line.split('\t')[0]) for line in Path(data['ordered_inputs']).read_text().splitlines()]
    categories=sorted(gt.getCatIds())
    rows=[json.loads(line) for line in args.predictions.read_text().splitlines()]
    if [r['image_id'] for r in rows]!=ids:raise RuntimeError('Missing, duplicated or reordered images')
    predictions=[]
    masked=[]
    for row in rows:
        if len(row['predictions'])>100:raise RuntimeError('Publication cap differs')
        for value in row['predictions']:
            cls=value['class_id']
            if type(cls) is not int or not 0<=cls<len(categories):raise RuntimeError('Invalid class')
            numbers=[*value['bbox'],value['score'],*value.get('keypoints',[])]
            if not all(math.isfinite(x) for x in numbers):raise RuntimeError('Non-finite prediction')
            if len(value['bbox'])!=4 or min(value['bbox'][2:])<0 or not 0<=value['score']<=1:
                raise RuntimeError('Invalid detection values')
            result={'image_id':row['image_id'],'category_id':categories[cls],'bbox':value['bbox'],'score':value['score']}
            if args.task=='pose':
                if len(value['keypoints'])!=51:raise RuntimeError('Expected17keypoints')
                result['keypoints']=value['keypoints']
                other=dict(result,keypoints=list(value['keypoints']))
                for index in range(17):
                    if other['keypoints'][index*3+2]<.5:other['keypoints'][index*3:index*3+3]=[0.,0.,0.]
                masked.append(other)
            predictions.append(result)
    if not predictions:raise RuntimeError('No predictions on full corpus')

    def evaluate(values,kind):
        with redirect_stdout(output_log):
            dt=gt.loadRes(values);evaluator=COCOeval(gt,dt,kind)
            evaluator.params.imgIds=ids
            evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
        return {'AP_percent':float(evaluator.stats[0])*100,'stats':evaluator.stats.tolist(),
            'max_detections':evaluator.params.maxDets,'iou_thresholds':evaluator.params.iouThrs.tolist()}

    metrics={'bbox':evaluate(predictions,'bbox')}
    if args.task=='pose':
        metrics['keypoints']=evaluate(predictions,'keypoints')
        metrics['keypoints_publication_filter']=evaluate(masked,'keypoints')
    report={'created_at':stamp(),'status':'COMPLETE','task':args.task,'images':len(ids),
        'predictions':len(predictions),'prediction_sha256':sha256(args.predictions),
        'annotation_sha256':sha256(annotation),'metrics':metrics,
        'scope':'Public COCO regression under deployment thresholds; not standard low-confidence Ultralytics leaderboard settings or independent recorded scenes.'}
    write_json(args.output,report)
    args.output.with_suffix('.log').write_text(output_log.getvalue())
    print(json.dumps({'task':args.task,'images':len(ids),'metrics':metrics}),flush=True)
    return 0


def hashes(protocol=PROTOCOL):
    names=['configs/runtime.yaml','models/manifest.json',
        'deploy/vision/source-lock.json','deploy/vision/serve.py','deploy/vision/eval/runner.cpp','tools/vision_quality.py']
    result={name:sha256(ROOT/name) for name in names}
    result[str(protocol.resolve())]=sha256(protocol)
    for relative in ('workspaces/vision/install/yolo_ros/lib/libyolo_ros_core.so',
            'install/onnxruntime-1.28.0-cuda13-sm87/lib/libonnxruntime.so'):
        path=CACHE/relative
        result[str(path.resolve())]=sha256(path)
    return result


def run(args):
    from deploy import services
    from deploy.resource_guard import hold_model_lease
    from tools.memory_capacity import MemoryMonitor
    if args.run_dir.exists() or args.output.exists():raise RuntimeError('Use new run paths')
    data=json.loads(args.dataset.read_text())
    if data['status']!='SEALED' or data['images']!=5000:raise RuntimeError('Full sealed corpus required')
    inputs=Path(data['ordered_inputs'])
    if sha256(inputs)!=data['ordered_inputs_sha256']:raise RuntimeError('Dataset input ordering drift')
    index=Path(data['image_index'])
    if sha256(index)!=data['image_index_sha256']:raise RuntimeError('Image manifest drift')
    for line in index.read_text().splitlines():
        row=json.loads(line)
        if sha256(Path(data['root'])/'val2017'/row['file_name'])!=row['sha256']:
            raise RuntimeError('Image differs from frozen index')
    reused=None
    if args.reference_run:
        reused=json.loads(args.reference_run.read_text())
        if (reused['dataset_sha256']!=sha256(args.dataset) or reused['errors']
                or not reused['checks']['source_unchanged'] or not reused['checks']['both_models_complete']
                or any(reused['source_hashes'][name]!=sha256(ROOT/name) for name in ('configs/runtime.yaml','models/manifest.json'))):
            raise RuntimeError('Reference run is incomplete or uses different data/models/runtime')
    before=services.state(list(services.ORDER),health=True)
    if before['backend_status']!='READY':raise RuntimeError('All backends must be ready')
    model=json.loads((ROOT/'models/manifest.json').read_text())['models']
    engine_paths=sorted((CACHE/'models/yolo/ort-trt-fp32').rglob('*.engine'))
    engines={str(p.resolve()):sha256(p) for p in engine_paths}
    if len(engines)!=2:raise RuntimeError('Expected two selected vision engines')
    args.run_dir.mkdir(parents=True,mode=0o700)
    (args.run_dir/'source.py').write_bytes(Path(__file__).read_bytes())
    (args.run_dir/'runner.cpp').write_bytes((ROOT/'deploy/vision/eval/runner.cpp').read_bytes())
    report={'created_at':stamp(),'status':'RUNNING','source_hashes':hashes(args.protocol),
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'dataset_manifest':str(args.dataset),'dataset_sha256':sha256(args.dataset),
        'verified_image_hashes':5000,
        'native_binary':str(EXECUTABLE),'native_binary_sha256':sha256(EXECUTABLE),
        'engines_before':engines,'backends_before':before,'results':{},'errors':[],
        'artifacts':str(args.run_dir.resolve()),'scope':json.loads(args.protocol.read_text())['scope'],
        'protocol':str(args.protocol),'protocol_sha256':sha256(args.protocol)}
    if reused:
        report['reference_reuse']={'report':str(args.reference_run),'sha256':sha256(args.reference_run),
            'reason':'Reuse unchanged frozen FP32 predictions after a native publication-only fix; rescore all5000images withsameCOCOeval andprimaryreference.',
            'original_source_commit':reused['source_commit'],'original_source_hashes':reused['source_hashes']}
    write_json(args.output,report)
    lease=hold_model_lease()
    monitor=MemoryMonitor();monitor.__enter__()
    stopped=False
    try:
        stopped=True;services.stop(['vision','pose'])
        for task in ('detection','pose'):
            result={}
            for backend in ('torch','tensorrt'):
                name=task+'-'+backend;predictions=args.run_dir/(name+'.jsonl')
                if backend=='torch':
                    command=[str(VISION_PY),str(Path(__file__).resolve()),'reference','--task',task,
                        '--inputs',str(inputs),'--predictions',str(predictions)]
                else:
                    onnx=CACHE/'models/yolo'/model[task]['onnx']['name']
                    if sha256(onnx)!=model[task]['onnx']['sha256']:raise RuntimeError('ONNX drift')
                    native=[str(EXECUTABLE),str(onnx),'detect' if task=='detection' else 'pose',
                        'tensorrt',str(CACHE/'models/yolo/ort-trt-fp32'/task),str(inputs),str(predictions)]
                    setup=' && '.join('source '+shlex.quote(str(p)) for p in
                        (Path('/opt/ros/jazzy/setup.bash'),CACHE/'workspaces/vision/install/setup.bash'))
                    command=['bash','-c',setup+' && exec '+shlex.join(native)]
                env=os.environ.copy()
                env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1')
                env['LD_LIBRARY_PATH']=str(CACHE/'install/onnxruntime-1.28.0-cuda13-sm87/lib')+':'+env.get('LD_LIBRARY_PATH','')
                report['current']={'task':task,'backend':backend,'command':command};write_json(args.output,report)
                if backend=='torch' and reused:
                    old=Path(reused['artifacts'])/(name+'.jsonl')
                    if sha256(old)!=reused['results'][task]['torch']['prediction_sha256']:
                        raise RuntimeError('Frozen reference predictions changed')
                    shutil.copyfile(old,predictions)
                    report['current']['execution']='reused frozen predictions; no new Torch inference'
                    report['current']['reference_source']=str(old)
                    write_json(args.output,report)
                else:
                    with (args.run_dir/(name+'.log')).open('w') as log:
                        subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                            check=True,timeout=7200,pass_fds=(lease.fileno(),))
                scored=args.run_dir/(name+'-metrics.json')
                command=[str(VISION_PY),str(Path(__file__).resolve()),'score','--task',task,
                    '--dataset',str(args.dataset.resolve()),'--predictions',str(predictions),'--output',str(scored)]
                with (args.run_dir/(name+'-score.log')).open('w') as log:
                    subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
                result[backend]=json.loads(scored.read_text())
                report['results'][task]=result;write_json(args.output,report)
                print(json.dumps({'task':task,'backend':backend,'completed':5000,'metrics':result[backend]['metrics']}),flush=True)
    except Exception as error:
        report['errors'].append({'type':type(error).__name__,'error':str(error)})
    finally:
        if stopped:
            try:report['restored']=services.start(['vision','pose'])
            except Exception as error:report['errors'].append({'stage':'restore','error':str(error)})
        lease.close()
        report['backends_after']=services.state(list(services.ORDER),health=True)
        monitor.__exit__();report['memory_observation']=monitor.report()
        comparisons={}
        for task,result in report['results'].items():
            if set(result)!={'torch','tensorrt'}:continue
            key='bbox' if task=='detection' else 'keypoints'
            reference=result['torch']['metrics'][key]['AP_percent'];candidate=result['tensorrt']['metrics'][key]['AP_percent']
            comparisons[task]={'reference_AP_percent':reference,'native_AP_percent':candidate,
                'drop_percentage_points':reference-candidate,'within_one_percentage_point':reference-candidate<=1.0}
            if task=='pose':comparisons[task]['reference_publication_filtered_AP_percent']=result['torch']['metrics']['keypoints_publication_filter']['AP_percent']
        checks={'no_errors':not report['errors'],'both_models_complete':len(comparisons)==2,
            'source_unchanged':hashes(args.protocol)==report['source_hashes'],
            'engines_unchanged':{str(p.resolve()):sha256(p) for p in sorted((CACHE/'models/yolo/ort-trt-fp32').rglob('*.engine'))}==engines,
            'backends_ready':report['backends_after']['backend_status']=='READY',
            'quality_drop_within_one_point':len(comparisons)==2 and all(r['within_one_percentage_point'] for r in comparisons.values())}
        report.update(checks=checks,comparisons=comparisons,status='PASS' if all(checks.values()) else 'FAIL',completed_at=stamp())
        report['raw_artifacts']=[{'path':str(p.resolve()),'sha256':sha256(p),'bytes':p.stat().st_size}
            for p in sorted(args.run_dir.iterdir()) if p.is_file()]
        write_json(args.output,report)
    print(json.dumps({'status':report['status'],'checks':checks,'comparisons':comparisons,'errors':report['errors']}),flush=True)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    ref=sub.add_parser('reference');ref.add_argument('--task',choices=['detection','pose'],required=True)
    ref.add_argument('--inputs',type=Path,required=True);ref.add_argument('--predictions',type=Path,required=True)
    sc=sub.add_parser('score');sc.add_argument('--task',choices=['detection','pose'],required=True)
    sc.add_argument('--dataset',type=Path,required=True);sc.add_argument('--predictions',type=Path,required=True);sc.add_argument('--output',type=Path,required=True)
    full=sub.add_parser('run');full.add_argument('--dataset',type=Path,required=True);full.add_argument('--run-dir',type=Path,required=True);full.add_argument('--output',type=Path,required=True)
    full.add_argument('--protocol',type=Path,default=PROTOCOL)
    full.add_argument('--reference-run',type=Path,help='Completed frozen run whose unchanged FP32 predictions can be reused')
    args=parser.parse_args()
    raise SystemExit({'reference':reference,'score':score,'run':run}[args.action](args))
