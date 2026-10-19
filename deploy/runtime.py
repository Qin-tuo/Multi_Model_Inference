# SPDX-License-Identifier: Apache-2.0
"""Native backend commands and locked local artifacts for the Reachy application.

systemd owns supervision. This module constructs existing upstream entrypoints;
it does not schedule inference, download weights or alter CUDA/TensorRT.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def config(path=None):
    from app.config import Config

    return Config.load(path)


def paths(cfg):
    stack = Path(cfg.backends.stack_root).expanduser().resolve()
    return {
        "stack": stack,
        "data": stack / "runtime-data/reachy-headless",
        "app_python": stack / "environments/reachy-headless/bin/python",
        "speech_python": stack / "environments/speech/bin/python",
        "llama": stack / "builds/llama/bin/llama-server",
        "models": stack / "models",
        "ovs": stack / "frameworks/openvoicestream",
    }


def port(url, *, suffix=""):
    value = urlsplit(url)
    if (
        value.scheme != "http"
        or value.hostname != "127.0.0.1"
        or value.path.rstrip("/") != suffix
        or not value.port
    ):
        raise ValueError(
            "Managed endpoints require explicit 127.0.0.1 ports and expected paths: "
            + url
        )
    return value.port


def ports(cfg):
    result = {
        "llama": port(cfg.llm.base_url),
        "asr": port(cfg.stt.native_base_url),
        "speech": port(cfg.tts.base_url, suffix="/v1"),
        "app": cfg.web.port,
    }
    if (
        cfg.stt.base_url != cfg.tts.base_url
        or cfg.web.host != "127.0.0.1"
        or len(set(result.values())) != 4
    ):
        raise ValueError(
            "Managed profile needs four distinct local ports and one shared speech endpoint"
        )
    return result


def native_command(component, cfg, profile):
    """把一份冻结配置转换为上游程序参数；不隐式下载、构建或调优。"""
    p = paths(cfg)
    listen = ports(cfg)
    b = cfg.backends
    if component == "app":
        return [
            str(p["app_python"]),
            str(ROOT / "run_web_vision_chat.py"),
            "--config",
            str(profile),
        ]
    if component == "speech":
        return [
            str(p["speech_python"]),
            str(ROOT / "deploy/speech/serve.py"),
            "--profile",
            str(profile),
        ]
    models = json.loads((ROOT / "models/manifest.json").read_text())["models"]
    model = models["vlm" if component == "llama" else "asr"]
    main = next(row for row in model["files"] if not row["name"].startswith("mmproj"))
    projector = next(row for row in model["files"] if row["name"].startswith("mmproj"))
    if component == "llama":
        command = [
            str(p["llama"]),
            "-lv",
            "4",
            "--jinja",
            "--alias",
            cfg.llm.model,
            "-m",
            str(p["models"] / main["name"]),
            "--mmproj",
            str(p["models"] / projector["name"]),
            "-ngl",
            str(b.llm_gpu_layers),
            "-c",
            str(cfg.llm.context_tokens),
            "-t",
            str(b.llm_threads),
            "-np",
            str(b.llm_parallel),
            "--host",
            "127.0.0.1",
            "--port",
            str(listen["llama"]),
            "--cache-ram",
            str(b.llm_cache_ram_mib),
        ]
        if template := model.get("chat_template"):
            file = (
                p["stack"] / "frameworks/llama.cpp" / template["relative_source_path"]
            )
            if sha256(file) != template["sha256"]:
                raise RuntimeError("Chat template hash mismatch")
            command += ["--chat-template-file", str(file)]
        return command + [
            "--chat-template-kwargs",
            json.dumps(cfg.llm.chat_template_kwargs, separators=(",", ":")),
        ]
    if component == "asr":
        return [
            str(p["llama"]),
            "-lv",
            "4",
            "--alias",
            b.asr_model_id,
            "-m",
            str(p["models"] / main["name"]),
            "--mmproj",
            str(p["models"] / projector["name"]),
            "-ngl",
            str(b.asr_gpu_layers),
            "-c",
            str(b.asr_context_tokens),
            "-t",
            str(b.asr_threads),
            "-np",
            "1",
            "--host",
            "127.0.0.1",
            "--port",
            str(listen["asr"]),
            "--cache-ram",
            str(b.asr_cache_ram_mib),
        ]
    raise ValueError("Unknown managed component: " + component)


def artifact_files(cfg, components):
    p = paths(cfg)
    models = json.loads((ROOT / "models/manifest.json").read_text())["models"]
    files = {}

    def add(path, digest):
        files[str(Path(path).expanduser())] = digest

    lock = json.loads((ROOT / "deploy/runtime-artifacts.lock.json").read_text())
    if set(components) & {"llama", "asr"}:
        for name, row in lock["native_files"].items():
            add(p["stack"] / name, row["sha256"])
    for component, role in [("llama", "vlm"), ("asr", "asr")]:
        if component in components:
            for row in models[role]["files"]:
                add(p["models"] / row["name"], row["sha256"])
    if "speech" in components:
        for name, row in lock["speech_files"].items():
            add(p["stack"] / name, row["sha256"])
    if "app" in components:
        perception = json.loads((ROOT / "models/perception.json").read_text())
        if cfg.face.enabled:
            row = perception["face"]
            add(
                Path(row["directory"]) / row["filename"],
                row["files"][row["filename"]]["sha256"],
            )
        if cfg.perception.enabled and cfg.perception.body_enabled:
            row = perception["body"]
            add(row["engine_path"], row["engine_sha256"])
            add(row["runtime_helper"], row["runtime_helper_sha256"])
        if cfg.perception.enabled and cfg.perception.hands_enabled:
            for role in ["palm", "hand"]:
                row = perception[role]
                for name in [
                    Path(row["model_path"]).name,
                    row.get("runtime_source_file", row["source_file"]),
                ]:
                    entry = row["files"][name]
                    add(entry["path"], entry["sha256"])
        if cfg.audio.enabled:
            row = json.loads((ROOT / "models/speech.json").read_text())["vad"]
            add(
                Path(row["directory"]) / row["filename"],
                row["files"][row["filename"]]["sha256"],
            )
    return files


def verify(cfg, components):
    p = paths(cfg)
    ports(cfg)
    if not p["app_python"].is_file() or (
        "speech" in components and not p["speech_python"].is_file()
    ):
        raise RuntimeError("Prepared isolated Python environment is missing")
    result = {}
    for name, expected in artifact_files(cfg, components).items():
        file = Path(name)
        if not file.is_file():
            raise RuntimeError("Missing prepared artifact: " + name)
        actual = sha256(file)
        if actual != expected:
            raise RuntimeError("Prepared artifact hash mismatch: " + name)
        result[name] = {"sha256": actual, "bytes": file.stat().st_size}
    if "speech" in components:
        expected = json.loads((ROOT / "deploy/speech/source-lock.json").read_text())[
            "upstream"
        ]["commit"]
        observed = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=p["ovs"], text=True
        ).strip()
        if observed != expected:
            raise RuntimeError("OVS revision mismatch")
        if subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=p["ovs"], text=True
        ).strip():
            raise RuntimeError("OVS source has unrecorded local modifications")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=["llama", "asr", "speech", "app"])
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    cfg = config(args.profile)
    if args.component in {"llama", "asr", "app"}:
        from deploy.resource_guard import hold_model_lease

        _lease = hold_model_lease(paths(cfg)["stack"] / "cache")
    verify(cfg, [args.component])
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OVS_AUTO_DOWNLOAD_ARTIFACTS="0",
        OTEL_SDK_DISABLED="true",
    )
    command = native_command(args.component, cfg, args.profile)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
