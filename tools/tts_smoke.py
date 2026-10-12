"""Exercise upstream VoxEdge/Matcha with host-built TensorRT engines and save WAV fixtures."""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.project import CACHE, DATA, stamp, write_json
from deploy.speech.cuda_compat import install_cuda_runtime_alias


def main():
    import numpy as np
    runtime = install_cuda_runtime_alias()
    code, cuda_version = runtime.cudaRuntimeGetVersion()
    if int(code) != 0: raise RuntimeError(f"CUDA runtime error {code}")
    from voxedge.backends.jetson.matcha_trt import MatchaTRTBackend, MatchaTRTConfig
    model = CACHE / "models/matcha-host"
    if not (model / "host-manifest.json").exists(): raise RuntimeError("Host Matcha build manifest is missing")
    backend = MatchaTRTBackend(MatchaTRTConfig(model_base=str(model), acoustic_ep="SPLIT_TRT",
                                            stream_max_workers=1, model_id="matcha-icefall-zh-en"))
    logging.basicConfig(level=logging.INFO)
    started = time.perf_counter()
    backend.preload()
    acoustic_mode = backend._acoustic_mode
    estimator_count = len(backend._split_estimator_engines)
    load_seconds = time.perf_counter()-started
    texts = ["你好，这是本地语音助手。", "我已经记住你喜欢简短的中文回答。", "五分钟后提醒你休息。"]
    results = []
    directory = DATA / "fixtures/tts"
    directory.mkdir(parents=True, exist_ok=True)
    try:
        for i, text in enumerate(texts):
            started = time.perf_counter()
            audio, metadata = backend.synthesize(text)
            elapsed = time.perf_counter()-started
            with wave.open(io.BytesIO(audio), "rb") as wav:
                samples = wav.getnframes()
                rate = wav.getframerate()
                channels = wav.getnchannels()
                width = wav.getsampwidth()
                signal = np.frombuffer(wav.readframes(samples), dtype="<i2").astype(np.float32)/32768
            rms = float(np.sqrt(np.mean(signal**2))) if signal.size else 0
            clip_fraction = float(np.mean(np.abs(signal) >= 0.999)) if signal.size else 1
            good = rate == 16000 and channels == 1 and width == 2 and samples > 1600 and rms > 0.0001 and clip_fraction < 0.05
            destination = directory / f"matcha-{i}.wav"
            destination.write_bytes(audio)
            results.append({"text": text, "status": "PASS" if good else "FAIL", "wav": str(destination),
                            "sample_rate": rate, "channels": channels, "duration_seconds": samples/rate,
                            "synthesis_seconds": elapsed, "rtf": elapsed/(samples/rate) if samples else None,
                            "rms": rms, "clip_fraction": clip_fraction, "metadata": metadata})
    finally:
        backend.unload()
    report = {"created_at": stamp(), "status": "PASS" if all(x["status"]=="PASS" for x in results) and acoustic_mode=="split_trt" and estimator_count==3 else "FAIL",
              "scope": "three synthetic Chinese TTS files; signal/format checks only, intelligibility and physical playback pending",
              "backend": "voxedge_matcha_split_tensorrt", "cuda_runtime": cuda_version,
              "load_seconds": load_seconds, "acoustic_mode": acoustic_mode, "gpu_estimator_count": estimator_count,
              "results": results}
    write_json(ROOT / "benchmarks/results/tts-smoke.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
