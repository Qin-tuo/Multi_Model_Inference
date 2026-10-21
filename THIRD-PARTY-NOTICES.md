# Third-party notices

The application retains NVIDIA copyright and Apache-2.0 notices from its upstream assistant code. Current source and model identities are recorded in `03_models/` and `01_src/deploy/speech/source-lock.json`. The application license does not replace the separate licenses of bundled components.

| Current component | Upstream license / retained notice |
|---|---|
| llama.cpp | MIT; `licenses/llama.cpp-MIT.txt` |
| OpenVoiceStream | MIT; `05_runtime/frameworks/openvoicestream/LICENSE` |
| TensorRT sample buffer helper | Apache-2.0; `licenses/TensorRT-Apache-2.0.txt` and the helper directory |
| YOLOv8n-seg and adapted segmentation code | AGPL-3.0; `licenses/Ultralytics-AGPL-3.0.txt` |
| MMPose RTMW and geometry/SimCC helpers | Apache-2.0; `05_runtime/frameworks/vision/mmpose/LICENSE` |
| OpenCV image/face transforms | Apache-2.0; `licenses/OpenCV-Apache-2.0.txt` |
| YuNet weights | MIT; `licenses/YuNet-MIT.txt` |
| ArcFace PyTorch implementation | MIT; `licenses/ArcFace-Pytorch-MIT.txt` |
| ProgressiveTeacher expression model / OpenCV Zoo alignment | Apache-2.0 declared in the upstream model README; `05_runtime/frameworks/vision/opencv_zoo/LICENSE` retains the upstream root notice |
| OMZ head-pose, landmark and gaze models | Apache-2.0; `licenses/OpenModelZoo-Apache-2.0.txt` |
| OSNet / Torchreid | MIT; `licenses/Torchreid-MIT.txt` |
| CLIP | MIT; `licenses/CLIP-MIT.txt` |
| Silero VAD asset distributed by faster-whisper | MIT; `05_runtime/models/vad/faster-whisper-1.2.1/LICENSE` |
| SQLAlchemy / APScheduler | MIT; `licenses/SQLAlchemy-MIT.txt`, `licenses/APScheduler-MIT.txt` |

The expression model's own LICENSE link was absent in the selected model subdirectory; its declaration and upstream root notice are retained rather than replaced with a stronger license claim.

The temporal open-hand rules in `01_src/app/vision/gestures.py` adapt the OpenCV Zoo hand-pose demo's five-finger angle/wrist-distance predicate (revision `4b2a0b446e5cf2f11fb6b2c7251091c035d2c1f7`, Apache-2.0). Local adaptations add finite-value guards and temporal interaction rules.

Runtime Python dependencies include NumPy, SciPy, OpenCV, Pillow, ONNX Runtime, FastAPI/Starlette, Uvicorn, Pydantic, HTTPX, PyYAML, Typer, Rich, psutil, Norfair, FilterPy and RealSense bindings. Installed distribution licenses remain in the two environment directories; version pins are in `requirements.lock` and `01_src/deploy/speech/requirements.lock`.

The speech runtime uses VoxEdge, Piper/espeak-ng and their transitive libraries. Their installed copyright and license texts remain with the packages; espeak-ng includes GPL-licensed code. Model weights, tokenizer/phoneme dictionaries and runtime libraries retain their own upstream terms and notices.
