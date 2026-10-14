from pathlib import Path
import json,numpy as np,torch,cv2
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes
import sys
sys.path.insert(0,'/home/nvidia/Multi_Model_Inference')
from tools.project import sha256,stamp,write_json
root=Path('/home/nvidia/AI_Stack/runtime-data/benchmarks/vision-input-diagnostic-v1')
torch.set_num_threads(2)
source=Path('/home/nvidia/AI_Stack/environments/vision/lib/python3.12/site-packages/ultralytics/assets/bus.jpg')
model=YOLO('/home/nvidia/AI_Stack/models/yolo/yolov8n.pt')
standard=model.predict(str(source),imgsz=640,rect=False,conf=.5,iou=.45,max_det=100,device='cpu',half=False,verbose=False,save=False)[0]
pixels=cv2.imread(str(root/'letterboxed.png'))
inputs={'reference_float':np.load(root/'ultralytics-input.npy'),
 'native_float':np.fromfile(root/'input.bin',np.float32).reshape(1,3,640,640),
 'native_pixels_torch_scaling':torch.from_numpy(pixels).unsqueeze(0).permute(0,3,1,2).flip(1).contiguous().float().div_(255).numpy()}
results={}
with torch.inference_mode():
 for name,array in inputs.items():
  raw=model.predictor.model(torch.from_numpy(array.copy()))
  preds=non_max_suppression(raw,conf_thres=.5,iou_thres=.45,nc=80,max_det=100)[0]
  preds[:,:4]=scale_boxes((640,640),preds[:,:4],(1080,810))
  rows=[]
  for p in preds.cpu().numpy():
   x1,y1,x2,y2,score,cls=map(float,p)
   rows.append({'class_id':int(cls),'score':score,'bbox':[x1,y1,x2-x1,y2-y1]})
  results[name]=rows
report={'created_at':stamp(),'status':'DIAGNOSTIC','source_sha256':sha256(Path(__file__)),
 'scope':'Onepublicbus image, samealreadyfusedPyTorchFP32 model+UltralyticsNMS. Onlyinputtensor changes. Doesnotmodify runtime,fullcorpusreference orscoring thresholds;auxiliaryCPU work overlaps ongoingaccuracyrun,whichisnotaperformancebenchmark.',
 'predictions':results}
write_json(root/'inference-comparison.json',report)
print(json.dumps({name:[r for r in rows if r['class_id']==5] for name,rows in results.items()},indent=2))
