"""Launch the pinned OVS HTTP service with local host-compatible backends."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "01_src"))
from deploy.runtime import config, paths, port, sha256, write_json, verify_speech_sources  # noqa: E402 — direct script entrypoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    cfg = config(args.profile)
    p = paths(cfg)
    from deploy.resource_guard import hold_model_lease

    _model_lease = hold_model_lease(p["data"] / "systemd")
    upstream = p["ovs"]
    verify_speech_sources(cfg)
    model = p["models"] / "matcha-host"
    manifest = json.loads((model / "host-manifest.json").read_text())
    import tensorrt as trt

    if manifest["tensorrt"] != trt.__version__:
        raise RuntimeError("Matcha engine/runtime TensorRT mismatch")
    for name, entry in manifest["engines"].items():
        if sha256(model / "engines" / name) != entry["engine_sha256"]:
            raise RuntimeError("Matcha host engine hash mismatch: " + name)
    profile = {
        "name": "orin-host-llama-asr-matcha",
        "asr_backend": "jetson.llama_qwen3_asr",
        "asr_model_id": cfg.stt.model,
        "tts_backend": "jetson.matcha_trt",
        "tts_model_id": cfg.tts.model,
        "local_llama_asr": {
            "base_url": cfg.stt.native_base_url,
            "model": cfg.backends.asr_model_id,
            "context": cfg.backends.asr_context,
        },
        "execution_policy": {
            "mode": "concurrent",
            "shared_resource": "gpu",
            "cross_modal_overlap": True,
        },
        "max_concurrent_sessions": 1,
        "env": {
            "LANGUAGE_MODE": "zh_en",
            "MATCHA_MODEL_BASE": str(model),
            "MATCHA_ACOUSTIC_EP": "SPLIT_TRT",
            "MATCHA_SPLIT_ENCODER_ONNX": str(model / "onnx/matcha_encoder_trt.onnx"),
            "MATCHA_SPLIT_ESTIMATOR_ENGINE": str(
                model / "engines/matcha_estimator_step0_bf16.engine"
            ),
            "VOCOS_ENGINE": str(model / "engines/vocos_fp16.engine"),
            "OVS_TTS_STREAM_MAX_WORKERS": "1",
            "OVS_TTS_SAMPLE_RATE": "16000",
            "OVS_TTS_CHANNELS": "1",
            "LAZY_TTS": "0",
        },
    }
    path = p["data"] / "config/ovs-host-profile.json"
    write_json(path, profile)
    os.environ["OVS_PROFILE_JSON"] = str(path)
    os.environ["MODEL_DIR"] = str(p["models"])
    os.environ["OVS_AUTO_DOWNLOAD_ARTIFACTS"] = "0"
    bundle_alias = p["models"] / "matcha-icefall-zh-en"
    if not bundle_alias.exists():
        bundle_alias.symlink_to(model.name, target_is_directory=True)
    elif bundle_alias.resolve() != model.resolve():
        raise RuntimeError("Matcha bundle alias points at another model")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["OTEL_SDK_DISABLED"] = "true"
    from deploy.speech.cuda_compat import install_cuda_runtime_alias
    from deploy.speech.piper_compat import install_piper_alias

    install_cuda_runtime_alias()
    install_piper_alias()
    sys.path.insert(0, str(upstream))
    from server.core.asr_backend import _ASR_REGISTRY

    _ASR_REGISTRY["jetson.llama_qwen3_asr"] = (
        "deploy.speech.llama_asr",
        "LlamaASRBackend",
    )
    # The pinned diagnostic probes a root-only RK debugfs path before Jetson.
    # Preserve its probes, but handle that specific permission failure using
    # the actual host marker. This does not bypass model compatibility checks.
    from server.core import gpu_watchdog

    original_detect = gpu_watchdog._detect_platform

    def detect_platform():
        try:
            return original_detect()
        except PermissionError:
            if Path("/etc/nv_tegra_release").is_file():
                return gpu_watchdog._PLATFORM_JETSON
            raise

    gpu_watchdog._detect_platform = detect_platform
    import uvicorn
    from server.main import app

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port(cfg.tts.base_url, suffix="/v1"),
        log_level="info",
    )


if __name__ == "__main__":
    main()
