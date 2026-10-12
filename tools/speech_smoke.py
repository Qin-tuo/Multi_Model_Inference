"""Check real OVS TTS→ASR HTTP round trips using synthetic text fixtures."""
from __future__ import annotations

import io
import argparse
import json
from pathlib import Path
import re
import sys
import time
import wave

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.project import DATA, stamp, write_json


def normalized(text):
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).casefold()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-audio", action="store_true", help="Use the same saved WAVs for an ASR-only regression")
    args = parser.parse_args()
    results = []
    base = "http://127.0.0.1:8621"
    texts = ["一加一等于几？", "请记住我喜欢简短的中文回答。", "项目使用 Jetson 和 TensorRT。"]
    directory = DATA / "fixtures/roundtrip"
    directory.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=90) as client:
        ready = client.get(base + "/readyz"); ready.raise_for_status()
        response = client.get(base + "/v1/capabilities"); response.raise_for_status()
        caps = response.json()
        rate = caps["tts"]["audio"]["sample_rate"]
        channels = caps["tts"]["audio"]["channels"]
        for index, text in enumerate(texts):
            path = directory / f"request-{index}.wav"
            if args.reuse_audio:
                wav_data = path.read_bytes()
                with wave.open(io.BytesIO(wav_data), "rb") as saved:
                    pcm = saved.readframes(saved.getnframes())
                tts_seconds = None
            else:
                started = time.perf_counter()
                audio = client.post(base + "/v1/audio/speech", json={"model":"matcha-icefall-zh-en", "voice":"0",
                                    "input":text,"response_format":"pcm"})
                audio.raise_for_status()
                tts_seconds = time.perf_counter()-started
                pcm = audio.content
                if not pcm or len(pcm) % (2*channels): raise RuntimeError("Invalid PCM payload")
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, "wb") as wav:
                    wav.setnchannels(channels); wav.setsampwidth(2); wav.setframerate(rate); wav.writeframes(pcm)
                wav_data = wav_buffer.getvalue()
                path.write_bytes(wav_data)
            started = time.perf_counter()
            response = client.post(base + "/v1/audio/transcriptions", data={"model":"qwen3-asr", "response_format":"json"},
                                   files={"file":(path.name,wav_data,"audio/wav")})
            response.raise_for_status()
            result = response.json()
            actual = result["text"]
            results.append({"input":text,"transcript":actual,"exact_normalized_match":normalized(text)==normalized(actual),
                            "tts_seconds":tts_seconds,"asr_seconds":time.perf_counter()-started,
                            "audio_seconds":len(pcm)/(2*channels*rate),"sample_rate":rate,"channels":channels,
                            "fixture_wav":str(path)})
            print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    report = {"created_at":stamp(),"status":"PASS" if all(r["exact_normalized_match"] for r in results) else "FAIL",
              "scope":"three synthetic HTTP speech round trips; not held-out ASR quality or physical playback/AEC",
              "capabilities":caps,"cases":results,"reused_fixture_audio":args.reuse_audio}
    write_json(ROOT / "benchmarks/results/speech-roundtrip.json", report)
    return 0 if report["status"]=="PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
