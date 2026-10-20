# Third-Party Software Notices

This project uses the following third-party open source software.

## Direct Dependencies

| Package | License | URL |
|---------|---------|-----|
| PyYAML | MIT | https://github.com/yaml/pyyaml |
| Rich | MIT | https://github.com/Textualize/rich |
| Typer | MIT | https://github.com/fastapi/typer |
| psutil | BSD-3-Clause | https://github.com/giampaolo/psutil |
| sounddevice | MIT | https://github.com/spatialaudio/python-sounddevice |
| Silero VAD model (bundled by faster-whisper) | MIT | https://github.com/snakers4/silero-vad |
| httpx | BSD-3-Clause | https://github.com/encode/httpx |
| faster-whisper | MIT | https://github.com/SYSTRAN/faster-whisper |
| kokoro-onnx | MIT | https://github.com/thewh1teagle/kokoro-onnx |
| opencv-python-headless | Apache-2.0 | https://github.com/opencv/opencv-python |
| ChromaDB | Apache-2.0 | https://github.com/chroma-core/chroma |
| FastAPI | MIT | https://github.com/fastapi/fastapi |
| Uvicorn | BSD-3-Clause | https://github.com/encode/uvicorn |

## Separately Installed Dependencies

| Package | License | URL |
|---------|---------|-----|
| onnxruntime-gpu | MIT | https://github.com/microsoft/onnxruntime |
| CTranslate2 | MIT | https://github.com/OpenNMT/CTranslate2 |
| NumPy | BSD-3-Clause | https://github.com/numpy/numpy |
| reachy-mini | Apache-2.0 | https://github.com/pollen-robotics/reachy_mini |

## Key Transitive Dependencies

| Package | License | URL |
|---------|---------|-----|
| Transformers | Apache-2.0 | https://github.com/huggingface/transformers |
| sentence-transformers | Apache-2.0 | https://github.com/UKPLab/sentence-transformers |
| Starlette | BSD-3-Clause | https://github.com/encode/starlette |
| Pydantic | MIT | https://github.com/pydantic/pydantic |
| espeakng-loader | MIT | https://github.com/thewh1teagle/espeakng-loader |

## Development and Release Validation Dependencies

| Package | License | URL |
|---------|---------|-----|
| ONNX | Apache-2.0 | https://github.com/onnx/onnx |
| pytest | MIT | https://github.com/pytest-dev/pytest |
| SciPy | BSD-3-Clause | https://github.com/scipy/scipy |
| build | MIT | https://github.com/pypa/build |
| wheel | MIT | https://github.com/pypa/wheel |
| setuptools | MIT | https://github.com/pypa/setuptools |
| pybind11 | BSD-3-Clause | https://github.com/pybind/pybind11 |

## GPL-Licensed TTS Dependencies

The following GPL-licensed packages are transitive dependencies of `kokoro-onnx` (MIT). They run in the separate `01_src/app/tts_worker.py` process and communicate with the main application over JSON stdin/stdout pipes. Distributions that include them must provide the required license notices and corresponding source material.

| Package | License | URL |
|---------|---------|-----|
| phonemizer-fork | GPL-3.0 | https://github.com/thewh1teagle/phonemizer |
| espeak-ng | GPL-3.0 | https://github.com/espeak-ng/espeak-ng |

## JetPack 7.2 Native Wheel Distribution

The optional JetPack 7.2 wheels are local builds of unmodified CTranslate2 4.8.1 and patched ONNX Runtime 1.28.0 source. The wheels do not bundle the CUDA or cuDNN system libraries. Any published release must preserve the upstream license and third-party-notice files embedded in the wheels.

ONNX Runtime's third-party notices include components under several permissive licenses and Eigen under MPL-2.0. The release evidence archives the exact source tags, the repository's ONNX Runtime patch, build commands, final wheel hashes, licenses, and third-party notices.

## External Services (Process-Isolated)

The following run as separate Docker containers and communicate via HTTP API:

| Software | License | URL |
|----------|---------|-----|
| llama.cpp | MIT | https://github.com/ggerganov/llama.cpp |

## Model Licenses

| Model | License | URL |
|-------|---------|-----|
| Cosmos-Reason2-2B | Apache-2.0 | https://huggingface.co/nvidia/Cosmos-Reason2-2B |
| Gemma 4 E2B (Unsloth GGUF) | Apache-2.0 | https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF |
| faster-whisper (small.en) | MIT | https://huggingface.co/Systran/faster-whisper-small.en |
| Kokoro v1.0 | Apache-2.0 | https://huggingface.co/hexgrad/Kokoro-82M |
| YuNet face detection | MIT | https://huggingface.co/opencv/face_detection_yunet |
| FER+ int8 emotion | MIT | https://huggingface.co/onnxmodelzoo/emotion-ferplus-12-int8 |
| reachy-mini-emotions-library | Apache-2.0 | https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library |
| bge-small-en-v1.5 | MIT | https://huggingface.co/BAAI/bge-small-en-v1.5 |

The current Gemma 4 distributor metadata labels the GGUF repository Apache-2.0 and also links Google's Gemma 4 terms at https://ai.google.dev/gemma/docs/gemma_4_license. Capture and review both the model-card metadata and linked terms before redistributing model artifacts.

### Kokoro model data attributions

The Kokoro v1.0 model card identifies the following Creative Commons Attribution audio in its training data. Preserve these credits when distributing the model or a bundle containing it.

| Audio data | License | Source |
|---|---|---|
| Koniwa `tnc` | CC-BY-3.0 | https://github.com/koniwa/koniwa |
| SIWIS | CC-BY-4.0 | https://datashare.ed.ac.uk/items/1de74991-eede-4b48-8fbe-6c2abaed88d8 |

See the upstream Kokoro model card for its complete acknowledgements and training-data statement: https://huggingface.co/hexgrad/Kokoro-82M.

## Dependency-resolution requirement

The tables above are a human-readable notice index. Keep an exact dependency lock for every supported installation profile and archive the applicable upstream license and notice texts with each release.


## Local perception adaptation

`01_src/app/body_detector.py` derives the array preprocessing/postprocessing from
Ultralytics' YOLOv8 OpenCV ONNX example, commit
`00cd4b8b187b6c7bc34776a6fe25a000f08b5fcc`, under AGPL-3.0.
The upstream original and license are retained in the directory recorded by
`01_src/deploy/perception-sources.lock.json`. The TensorRT buffer/stream helper is
NVIDIA's `common_runtime.py` at commit `16ed0636df41e4a27d46f82d9028fc50fadc7200`
(Apache-2.0), loaded from the checked external source. OpenCV Zoo palm and
hand-pose source/model identities and Apache-2.0 notices are recorded in
`03_models/perception.json`. These components retain their own notices.

`01_src/app/gestures.py` adapts the five-finger angle and wrist-distance predicate
from OpenCV Zoo's `GestureClassification` in the handpose demo at the same
pinned Hugging Face revision `4b2a0b446e5cf2f11fb6b2c7251091c035d2c1f7`
(Apache-2.0). The original demo, digest and download URL are retained in
`03_models/perception.json`; the demo is not imported because it parses CLI
arguments and downloads a model at module import. Local adaptations add
finite-vector guards and temporal interaction rules.

## Local actions

The optional notes/reminders module uses SQLAlchemy 2.1.4 and APScheduler
3.11.3, both MIT licensed. Exact installed versions and license digests are
recorded in `01_src/deploy/actions-dependencies.lock.json`; license copies are in
`licenses/SQLAlchemy-MIT.txt` and `licenses/APScheduler-MIT.txt`.
Application transaction, idempotency and output-receipt rules adapt the
previous project's SQLAlchemy/ReminderService contracts, without importing
its Pipecat, memory or identity layers.

## B1 visual runtime

The selected artifacts and exact upstream revisions are recorded in
`03_models/candidates/vision-b1.json`; engine and runtime contracts are in
`03_models/vision-b1-runtime.json`. Upstream license declarations remain separate
from the application license. These records describe selected public models,
not the unavailable proprietary models or their original output taxonomy.

| Component | Declared upstream license | Retained notice/source |
|---|---|---|
| YOLOv8n-seg and adapted segmentation preprocessing/postprocessing | AGPL-3.0 | `licenses/Ultralytics-AGPL-3.0.txt`; pinned Ultralytics sources in the candidate manifest |
| RTMW3D-X and MMPose geometry/SimCC decoding | Apache-2.0 | Fixed MMPose revision and `frameworks/b1/mmpose/LICENSE` in the runtime bundle |
| YuNet model | MIT | `licenses/YuNet-MIT.txt`, copied from model revision `3cc26e7f1014a5ee5d74a42acee58bafc9d0a310` |
| YuNet decoder reference and OpenCV | Apache-2.0 | `licenses/OpenCV-Apache-2.0.txt`; pinned `face_detect.cpp` reference |
| SFace | Apache-2.0 | `frameworks/b1/sface/LICENSE` and pinned model card |
| ProgressiveTeacher facial expression model | Apache-2.0 declaration in model README | Model card and `frameworks/b1/expression/UPSTREAM-LICENSE`; the model-specific LICENSE link was unavailable, as recorded in the candidate manifest |
| Open Model Zoo head pose, 35-point landmarks and gaze cascade | Apache-2.0 | `frameworks/b1/open-model-zoo/LICENSE`, model cards and `model.yml` records |
| OSNet person appearance | MIT | `frameworks/b1/torchreid/LICENSE` and model card |
| CLIP image appearance | MIT | `frameworks/b1/clip/LICENSE` and model card |
| Norfair / FilterPy | BSD-3-Clause / MIT | License metadata in the packaged Python distributions; versions in the B1 runtime requirements lock |

Runtime paths in this table are relative to `05_runtime/`. PyTorch, ONNX Runtime
and OpenVINO reference environments are development-only and excluded from the
prepared B1 runtime bundle. The existing 4B and speech backends retain their
previous source/model notices.

## B2 standard replacements

`03_models/candidates/vision-b2.json` pins the three standard model replacements.
The other seven B1 visual models retain their original artifacts and notices.

- RTMPose-l COCO-WholeBody 2D uses the unmodified MMPose CSPNeXt/RTMCC
  network, official 133-point checkpoint and Apache-2.0 sources at revision
  `759b39c13fea6ba094afc1fa932f51dc1b11cbf9`. The runtime NumPy decoder is
  extracted verbatim from `mmpose/codecs/utils/post_processing.py` so unused
  Torch helpers are not application dependencies; its source and derived
  hashes are retained.
- OSNet x1.0 uses the official Torchreid MSMT17 combine-all checkpoint and
  model definition at `f8cd150fdf77e8d9e1ed143b7f308c2c609ded50`, under
  the existing Torchreid MIT notice.
- ArcFace-MobileNetV1 uses the published v1.1 checkpoint and unmodified
  inference source at `92ec13d5768d46614fbcac4c6de7c9d94a9d32ae` from
  `bubbliiiing/arcface-pytorch`. Its MIT notice is retained in
  `licenses/ArcFace-Pytorch-MIT.txt`. The existing OpenCV five-point alignment
  frontend remains separately attributed above.

These are public pretrained checkpoints, not the uploaded customized ONNX
files. No retraining or learned-layer replacement is performed.
