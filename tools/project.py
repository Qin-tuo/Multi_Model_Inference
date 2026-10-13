#!/usr/bin/env python3
"""Reproducible project operations. No dependency installs happen in doctor."""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import glob
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = Path(os.environ.get("ASSISTANT_CACHE_DIR", Path.home() / ".cache/multi-model-assistant"))
DATA = Path(os.environ.get("ASSISTANT_DATA_DIR", Path.home() / ".local/share/multi-model-assistant"))


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def run(command, timeout=20, cwd=None):
    try:
        r = subprocess.run(command, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        return {"command": command, "exit_code": r.returncode,
                "stdout": r.stdout[-24000:], "stderr": r.stderr[-4000:]}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"command": command, "exit_code": None, "error": str(e)}


def read(path):
    try:
        return Path(path).read_text().strip().replace("\x00", "")
    except (OSError, UnicodeError):
        return None


def meminfo():
    result = {}
    for line in (read("/proc/meminfo") or "").splitlines():
        fields = line.split()
        if len(fields) >= 2:
            result[fields[0].rstrip(":")] = int(fields[1]) * 1024
    return result


def cuda_probe():
    """Use the installed driver directly; no Python CUDA package is required."""
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        rc = lib.cuInit(0)
        if rc:
            return {"status": "FAIL", "cuInit": rc}
        count = ctypes.c_int()
        lib.cuDeviceGetCount(ctypes.byref(count))
        devices = []
        for index in range(count.value):
            device = ctypes.c_int()
            lib.cuDeviceGet(ctypes.byref(device), index)
            name = ctypes.create_string_buffer(256)
            major, minor = ctypes.c_int(), ctypes.c_int()
            lib.cuDeviceGetName(name, 256, device)
            lib.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), device)
            devices.append({"name": name.value.decode(), "compute_capability": f"{major.value}.{minor.value}"})
        return {"status": "PASS" if devices else "FAIL", "devices": devices,
                "evidence": "CUDA driver enumeration; not yet a model/kernel test"}
    except (OSError, AttributeError) as e:
        return {"status": "FAIL", "error": str(e)}


def doctor(_args):
    tools = {}
    for name, argv in {"cmake": ["cmake", "--version"], "ninja": ["ninja", "--version"],
                       "compiler": ["g++", "--version"], "cuda": ["nvcc", "--version"],
                       "docker": ["docker", "--version"], "nsys": ["nsys", "--version"]}.items():
        tools[name] = run(argv)
    packages = run(["dpkg-query", "-W", "-f=${Package}\t${Version}\n",
                    "nvidia-l4t-core", "libnvinfer10", "libcudnn9-cuda-13", "nvidia-jetpack"])
    video = sorted(glob.glob("/dev/video*"))
    capture = run(["arecord", "-l"])
    playback = run(["aplay", "-l"])
    report = {"schema_version": 1, "created_at": stamp(), "device_model": read("/proc/device-tree/model"),
              "architecture": platform.machine(), "kernel": platform.release(),
              "os": read("/etc/os-release"), "l4t": read("/etc/nv_tegra_release"),
              "memory_bytes": meminfo(), "disk": dict(zip(("total", "used", "free"), shutil.disk_usage(ROOT))),
              "tools": tools, "packages": packages, "cuda_driver": cuda_probe(),
              "ros": {"jazzy_setup_exists": Path("/opt/ros/jazzy/setup.bash").exists()},
              "devices": {"video_nodes": video, "audio_capture": capture, "audio_playback": playback,
                          "audio_evidence": "enumeration only; physical microphone/speaker must be verified"},
              "power_mode": run(["/usr/sbin/nvpmodel", "-q"]),
              "thermal": [{"type": read(f"{p}/type"), "millidegrees": read(f"{p}/temp")}
                          for p in glob.glob("/sys/class/thermal/thermal_zone*")]}
    critical = [name for name in ("cmake", "compiler", "cuda") if tools[name].get("exit_code") != 0]
    gaps = critical[:]
    if report["cuda_driver"]["status"] != "PASS": gaps.append("CUDA driver")
    if not report["ros"]["jazzy_setup_exists"]: gaps.append("ROS 2 Jazzy not installed")
    if not video: gaps.append("RGB-D camera not enumerated")
    gaps.append("physical audio capture/playback not verified")
    report["gaps"] = gaps
    report["status"] = "FAIL" if critical or report["cuda_driver"]["status"] != "PASS" else "PARTIAL"
    raw = DATA / "reports/doctor.json"
    write_json(raw, report)
    # Exclude device serials, hostnames, USB IDs and complete environment variables.
    summary = {k: report[k] for k in ("schema_version", "created_at", "status", "device_model", "architecture",
                                     "kernel", "memory_bytes", "disk", "cuda_driver", "ros", "gaps")}
    summary["package_versions"] = packages.get("stdout", "")
    write_json(ROOT / "benchmarks/results/environment.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Local report: {raw}")
    return 2 if gaps else 0


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url, destination, expected_sha, expected_size):
    """Resume only if server confirms the requested offset; verify before publish."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == expected_size and sha256(destination) == expected_sha:
            print(f"Verified cache: {destination.name}", flush=True)
            return
        raise RuntimeError(f"Existing artifact failed verification: {destination}")
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == expected_size:
            break
        headers = {"User-Agent": "multi-model-assistant/0.1"}
        if offset: headers["Range"] = f"bytes={offset}-"
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                resumed = response.status == 206
                if resumed:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise RuntimeError("Server returned unexpected Content-Range")
                mode = "ab" if resumed else "wb"
                total = offset if resumed else 0
                last = time.monotonic()
                with partial.open(mode) as output:
                    while chunk := response.read(4 * 1024 * 1024):
                        output.write(chunk)
                        total += len(chunk)
                        if total > expected_size: raise RuntimeError("Artifact exceeds locked size")
                        if time.monotonic() - last >= 15:
                            print(f"Download {destination.name}: {total}/{expected_size}", flush=True)
                            last = time.monotonic()
            if total == expected_size: break
        except (OSError, urllib.error.URLError) as e:
            print(f"Download attempt {attempt+1}: {type(e).__name__}: {e}", flush=True)
            if attempt == 4: raise
            time.sleep(2)
    if partial.stat().st_size != expected_size or sha256(partial) != expected_sha:
        raise RuntimeError(f"Artifact hash/size mismatch: {partial}")
    partial.replace(destination)
    print(f"Downloaded and verified: {destination.name}", flush=True)


def manifests():
    return (json.loads((ROOT / "deploy/upstreams.lock.json").read_text()),
            json.loads((ROOT / "models/manifest.json").read_text()))


def prepare(args):
    if args.component == "speech":
        _, models = manifests()
        model = models["models"]["asr"]
        for file in model["files"]:
            download_file(f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{file['name']}",
                          CACHE / "models" / file["name"], file["sha256"], file["size"])
        subprocess.run(["python3", str(ROOT / "deploy/speech/build_matcha_host.py")], check=True)
        return 0
    if args.component == "memory":
        subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "tools/memory_model.py"), "prepare"], check=True)
        return 0
    if args.component != "llama":
        raise RuntimeError(f"prepare for {args.component} has not been implemented")
    lock, models = manifests()
    source = CACHE / "upstreams/llama.cpp"
    build = CACHE / "build/llama"
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        subprocess.run(["git", "clone", "--depth", "1", lock["llama"]["url"], str(source)], check=True)
    dirty = run(["git", "status", "--porcelain"], cwd=source)
    if dirty.get("stdout", "").strip(): raise RuntimeError("Upstream checkout is modified; refusing overwrite")
    wanted = lock["llama"]["commit"]
    if run(["git", "rev-parse", "HEAD"], cwd=source).get("stdout", "").strip() != wanted:
        subprocess.run(["git", "fetch", "--depth", "1", "origin", wanted], cwd=source, check=True)
        subprocess.run(["git", "checkout", "--detach", wanted], cwd=source, check=True)
    if not args.download_only:
        subprocess.run(["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                        "-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=87", "-DLLAMA_BUILD_TESTS=OFF",
                        "-DLLAMA_OPENSSL=OFF"], check=True)
        subprocess.run(["cmake", "--build", str(build), "--target", "llama-server", "llama-cli",
                        "-j", str(args.jobs)], check=True)
    if not args.build_only:
        model = models["models"]["vlm"]
        for file in model["files"]:
            url = f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{file['name']}"
            download_file(url, CACHE / "models" / file["name"], file["sha256"], file["size"])
    return 0


def request_json(url, data=None, timeout=120):
    encoded = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, encoded, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def server_command(port=None):
    _, models = manifests()
    model = models["models"]["vlm"]
    config = json.loads((ROOT / "configs/runtime.yaml").read_text())["llama"]
    main = next(x for x in model["files"] if not x["name"].startswith("mmproj"))
    projector = next(x for x in model["files"] if x["name"].startswith("mmproj"))
    binary = CACHE / "build/llama/bin/llama-server"
    for path in (binary, CACHE / "models" / main["name"], CACHE / "models" / projector["name"]):
        if not path.exists(): raise RuntimeError(f"Missing prepared artifact: {path}")
    command = [str(binary), "-lv", "4", "--jinja", "--alias", model.get("server_model_id", "assistant-vlm"),
            "-m", str(CACHE / "models" / main["name"]), "--mmproj",
            str(CACHE / "models" / projector["name"]), "-ngl", str(config["gpu_layers"]),
            "-c", str(config["context_tokens"]), "-t", str(config["threads"]),
            "-np", str(config["parallel"]), "--host", "127.0.0.1", "--port", str(port or config["port"])]
    if template := model.get("chat_template"):
        path = CACHE / "upstreams/llama.cpp" / template["relative_source_path"]
        if sha256(path) != template["sha256"]: raise RuntimeError("Chat template differs from the model manifest")
        command += ["--chat-template-file", str(path)]
    return command


def wait_ready(process, url, seconds=240):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        if process.poll() is not None: raise RuntimeError(f"llama-server exited: {process.returncode}")
        try:
            if request_json(url + "/health", timeout=2).get("status") == "ok": return
        except (urllib.error.URLError, TimeoutError, OSError): pass
        time.sleep(1)
    raise TimeoutError("llama-server readiness timed out")


def smoke_llama(args):
    import base64
    import struct
    import zlib
    # Synthetic test fixture; does not pretend to be a physical camera capture.
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind+data)&0xffffffff)
    pixels = (b"\x00" + b"\xff\x00\x00"*224)*224
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB",224,224,8,2,0,0,0)) + chunk(b"IDAT",zlib.compress(pixels)) + chunk(b"IEND",b"")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
    command = server_command(port)
    run_id = dt.datetime.now(dt.timezone.utc).strftime("llama-smoke-%Y%m%dT%H%M%SZ")
    log = DATA / "logs" / (run_id + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    cases = [("text", "只用中文回答：一加一等于几？"), ("image", [
        {"type": "text", "text": "这张图片主要是什么颜色？用简短中文回答。"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,"+base64.b64encode(png).decode()}}])]
    results = []
    start = time.monotonic()
    with log.open("w") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        try:
            url = f"http://127.0.0.1:{port}"
            wait_ready(process, url)
            load_seconds = time.monotonic()-start
            for name, content in cases:
                started = time.monotonic()
                response = request_json(url+"/v1/chat/completions", {
                    "messages": [{"role": "user", "content": content}], "max_tokens": 64,
                    "temperature": 0, "stream": False})
                answer = response["choices"][0]["message"].get("content", "")
                correct = ("二" in answer or "2" in answer) if name == "text" else "红" in answer
                results.append({"case": name, "answer": answer, "seconds": time.monotonic()-started,
                                "usage": response.get("usage"), "nonempty": bool(answer.strip()),
                                "sanity_answer_correct": correct})
            status = read(f"/proc/{process.pid}/status") or ""
            memory = {line.split(":")[0]: line.split(":",1)[1].strip() for line in status.splitlines()
                      if line.startswith(("VmRSS:", "VmHWM:"))}
        finally:
            process.terminate()
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
    backend_log = log.read_text()
    cuda_evidence = [line for line in backend_log.splitlines() if "CUDA" in line or "offloaded" in line]
    gpu_confirmed = any("offloaded" in line and not "offloaded 0/" in line for line in cuda_evidence)
    report = {"run_id": run_id, "created_at": stamp(), "backend": "real", "input_mode": "synthetic_fixture",
              "status": "PASS" if gpu_confirmed and all(x["sanity_answer_correct"] for x in results) else "FAIL",
              "scope": "GPU load and text/image functionality only; not quality or latency acceptance",
              "load_seconds": load_seconds, "memory": memory, "command": command,
              "cuda_evidence": cuda_evidence[-30:], "results": results, "local_log": str(log)}
    write_json(ROOT / "benchmarks/results/llama-smoke.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


def smoke_memory(_args):
    subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "tools/memory_model.py"), "smoke"], check=True)
    return 0


def smoke(args):
    if args.component == "llama": return smoke_llama(args)
    if args.component == "memory": return smoke_memory(args)
    if args.component == "speech":
        subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "tools/speech_smoke.py")], check=True)
        return 0
    raise RuntimeError(f"Real smoke for {args.component} is not implemented yet")


def serve(args):
    if args.component=='qdrant':
        from deploy.memory.runtime import serve as serve_memory
        return serve_memory(DATA/'qdrant')
    model_lease = None
    if args.component in {"llama", "asr", "speech", "vision", "pose"}:
        from deploy.resource_guard import hold_model_lease
        model_lease = hold_model_lease()
    if args.component in {"camera", "vision", "pose"}:
        component = "detection" if args.component == "vision" else args.component
        command = [sys.executable, str(ROOT / "deploy/vision/serve.py"), component]
        os.execv(command[0], command)
    if args.component in {"llama", "asr", "speech"}:
        for proc in Path("/proc").glob("[0-9]*"):
            try: argv = (proc / "cmdline").read_bytes().split(b"\0")
            except OSError: continue
            if b"--build-ort" in argv and any(arg.endswith(b"deploy/vision/prepare.py") for arg in argv):
                raise RuntimeError("ONNX Runtime GPU build is still active; start model services after this heavy build completes")
    if args.component == "asr":
        _, models = manifests()
        model = models["models"]["asr"]
        main = next(x for x in model["files"] if not x["name"].startswith("mmproj"))
        projector = next(x for x in model["files"] if x["name"].startswith("mmproj"))
        command = [str(CACHE / "build/llama/bin/llama-server"), "-lv", "4", "--alias", "assistant-asr",
                   "-m", str(CACHE / "models" / main["name"]), "--mmproj", str(CACHE / "models" / projector["name"]),
                   "-ngl", "99", "-c", str(model["context_tokens"]), "-t", "2", "-np", "1",
                   "--host", "127.0.0.1", "--port", str(model["port"])]
        os.execv(command[0], command)
    if args.component == "speech":
        python = str(ROOT / ".venv-speech/bin/python")
        os.execv(python, [python, str(ROOT / "deploy/speech/serve.py")])
    if args.component != "llama":
        raise RuntimeError("Only the llama model service is ready; the complete assistant profile is pending")
    command = server_command()
    os.execv(command[0], command)


def tests(args):
    patterns = {"core": "test_*.py", "framework": "test_pipecat_pipeline.py",
                "memory": "test_mem0_adapter.py", "interaction": "test_interaction*.py"}
    pattern = patterns[args.suite]
    if not list((ROOT / "tests").glob(pattern)):
        raise RuntimeError(f"Test suite {args.suite} is not implemented; no empty success")
    python = ROOT / ".venv/bin/python"
    if not python.exists(): raise RuntimeError("Project venv is missing")
    subprocess.run([str(python), str(ROOT / "tools/run_tests.py"), "--pattern", pattern], check=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("doctor")
    prep = sub.add_parser("prepare")
    prep.add_argument("--component", required=True, choices=["llama", "vision", "speech", "memory"])
    exclusive = prep.add_mutually_exclusive_group()
    exclusive.add_argument("--build-only", action="store_true")
    exclusive.add_argument("--download-only", action="store_true")
    prep.add_argument("--jobs", type=int, default=2, choices=range(1,5))
    sm = sub.add_parser("smoke")
    sm.add_argument("--component", required=True, choices=["llama", "vision", "speech", "memory"])
    srv = sub.add_parser("serve", help="Run a model service in the foreground")
    srv.add_argument("--component", choices=["qdrant", "llama", "asr", "speech", "camera", "vision", "pose"], required=True)
    test = sub.add_parser("test")
    test.add_argument("--suite", choices=["core", "framework", "memory", "interaction"], default="core")
    bench = sub.add_parser('bench', help='Measured workloads; full B0-B3 scenarios remain separate work')
    bench.add_argument('--suite', choices=['memory-capacity'], required=True)
    bench.add_argument('--tier', type=int, choices=[1000,10000,50000], required=True)
    bench.add_argument('--run-dir', type=Path, required=True)
    bench.add_argument('--output', type=Path, required=True)
    bench.add_argument('--seed-method', choices=['adapter','sdk-batch'], default='adapter')
    interact = sub.add_parser("interact", help="Persistent Pipecat text conversation")
    interact.add_argument("--user", default="owner")
    interact.add_argument("--data-dir", type=Path, default=DATA)
    interact.add_argument("--session-id")
    interact.add_argument("--input-file", type=Path)
    interact.add_argument("--no-vision", action="store_true")
    interact.add_argument('--visual-feedback', choices=['on','off'])
    voice = sub.add_parser("voice", help="Persistent Pipecat speech, memory, tools and vision")
    voice.add_argument("--user", default="owner")
    voice.add_argument("--data-dir", type=Path, default=DATA)
    voice.add_argument("--session-id")
    voice.add_argument("--no-vision", action="store_true")
    voice.add_argument('--visual-feedback', choices=['on','off'])
    voice.add_argument("--input-wav", type=Path, nargs="+")
    voice.add_argument("--output-wav", type=Path)
    voice.add_argument("--report", type=Path)
    voice.add_argument("--input-device", type=int)
    voice.add_argument("--output-device", type=int)
    notifications = sub.add_parser("notifications", help="Text reminder delivery; independent of model services")
    notifications.add_argument("--user", default="owner")
    notifications.add_argument("--data-dir", type=Path, default=DATA)
    notifications.add_argument("--seconds", type=float, default=0)
    for action in ("up", "down", "status", "install-services"):
        lifecycle = sub.add_parser(action, help="Managed backend lifecycle via user systemd")
        lifecycle.add_argument("--components", default="all")
        lifecycle.add_argument("--timeout", type=float, default=300)
    logs = sub.add_parser("logs", help="Read one managed backend's rotated log")
    logs.add_argument("--component", choices=["qdrant", "llama", "asr", "speech", "camera", "vision", "pose"], required=True)
    logs.add_argument("--lines", type=int, default=50)
    args = parser.parse_args()
    try:
        if args.action == 'bench':
            env = os.environ.copy()
            env.setdefault('OPENBLAS_NUM_THREADS','1')
            env.setdefault('OMP_NUM_THREADS','2')
            return subprocess.call([str(ROOT/'.venv/bin/python'),str(ROOT/'tools/memory_capacity.py'),
                '--tier',str(args.tier),'--run-dir',str(args.run_dir),'--output',str(args.output),
                '--seed-method',args.seed_method],cwd=ROOT,env=env)
        if args.action == "logs":
            from deploy.services import main as services_main
            return services_main(["logs", "--components", args.component, "--lines", str(args.lines)])
        if args.action in {"up", "down", "status", "install-services"}:
            from deploy.services import main as services_main
            return services_main(["install" if args.action == "install-services" else args.action,
                                  "--components", args.components, "--timeout", str(args.timeout)])
        if args.action == "notifications":
            return subprocess.call([str(ROOT / ".venv/bin/python"), "-m", "app.reminder_console", "--user", args.user,
                                    "--data-dir", str(args.data_dir), "--seconds", str(args.seconds)], cwd=ROOT)
        if args.action in {"interact", "voice"}:
            application_env=os.environ.copy()
            application_env.setdefault('OPENBLAS_NUM_THREADS','1')
            application_env.setdefault('OMP_NUM_THREADS','2')
            module = "app.interact" if args.action == "interact" else "app.voice_cli"
            command = [str(ROOT / ".venv/bin/python"), "-m", module, "--user", args.user,
                       "--data-dir", str(args.data_dir)]
            if args.session_id: command += ["--session-id", args.session_id]
            if args.visual_feedback is not None: command += ['--visual-feedback', args.visual_feedback]
            if args.action == "interact":
                if args.input_file: command += ["--input-file", str(args.input_file)]
            else:
                if args.input_wav: command += ["--input-wav", *map(str, args.input_wav)]
                for field in ("output_wav", "input_device", "output_device", "report"):
                    if (value := getattr(args, field)) is not None:
                        command += ["--" + field.replace("_", "-"), str(value)]
            if args.no_vision:
                command += ["--no-vision"]
                return subprocess.call(command, cwd=ROOT,env=application_env)
            from deploy.vision.serve import ros_environment
            setup = Path("/opt/ros/jazzy/setup.bash")
            if not setup.is_file(): raise RuntimeError("ROS is unavailable; use --no-vision for text-only operation")
            overlay = CACHE / 'workspaces/vision/install/setup.bash'
            if not overlay.is_file(): raise RuntimeError('Vision workspace is unavailable; prepare vision or use --no-vision')
            script = "source " + shlex.quote(str(setup)) + ' && source ' + shlex.quote(str(overlay)) + " && exec " + shlex.join(command)
            ros_env=ros_environment()
            ros_env.update({key:application_env[key] for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')})
            return subprocess.call(["bash", "-c", script], cwd=ROOT, env=ros_env)
        return {"doctor": doctor, "prepare": prepare, "smoke": smoke, "test": tests, "serve": serve}[args.action](args)
    except (OSError, RuntimeError, TimeoutError, subprocess.CalledProcessError, ValueError) as e:
        print(json.dumps({"status": "FAIL", "error": str(e)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
