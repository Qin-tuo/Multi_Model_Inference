"""Small bootstrap/health CLI; systemd owns processes, restart and cgroup cleanup."""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.project import CACHE, DATA, meminfo, server_command, stamp, write_json

ORDER = ("llama", "asr", "speech", "camera", "vision", "pose")
DEPENDENCIES = {"speech": ("asr",), "vision": ("camera",), "pose": ("camera",)}
PORTS = {"llama": 8088, "asr": 8089, "speech": 8621}
UNIT_DIR = DATA / "systemd"
LINK_DIR = Path.home() / ".config/systemd/user"


def unit(component): return f"mmassistant-{component}.service"


def command(argv, *, timeout=30, check=True, env=None):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
    if check and result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip()[-1800:])
    return result


def quote(value, *, executable=False):
    value = str(value)
    if any(char in value for char in "\n\r\x00"): raise ValueError("Unit values cannot contain control characters")
    value = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    if executable: value = value.replace("$", "$$")
    return '"' + value + '"'


def selected(value, *, dependencies=True):
    names = set(value.split(",")) if value != "all" else set(ORDER)
    if not names or names - set(ORDER): raise ValueError("Unknown backend component")
    if dependencies:
        for name in list(names): names.update(DEPENDENCIES.get(name, ()))
    return [name for name in ORDER if name in names]


def properties(component):
    fields = ["LoadState", "ActiveState", "SubState", "MainPID", "ControlGroup", "FragmentPath", "NRestarts", "Result"]
    result = command(["systemctl", "--user", "show", unit(component), "--no-pager", *["--property=" + field for field in fields]], check=False)
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    values["main_pid"] = int(values.get("MainPID", 0))
    path = values.get("ControlGroup", "")
    values["cgroup_pids"] = []
    if path:
        group = Path("/sys/fs/cgroup") / path.lstrip("/")
        try:
            values["cgroup_pids"] = sorted({int(pid) for file in group.rglob("cgroup.procs") for pid in file.read_text().split()})
        except FileNotFoundError: pass
    return values


def render(component):
    template = (ROOT / "deploy/systemd/backend.service.in").read_text()
    deps = [unit(name) for name in DEPENDENCIES.get(component, ())]
    env = {"ASSISTANT_CACHE_DIR": str(CACHE), "ASSISTANT_DATA_DIR": str(DATA),
           "ASSISTANT_ROS_DOMAIN_ID": os.environ.get("ASSISTANT_ROS_DOMAIN_ID", "42"),
           "PYTHONUNBUFFERED": "1"}
    return template.format(project=ROOT, component=component, after=" ".join(deps), wants=" ".join(deps),
        working_directory=str(ROOT).replace("%", "%%"), command=" ".join(quote(arg, executable=True) for arg in
            ("/usr/bin/python3", ROOT / "deploy/logged_service.py", component)),
        environment="\n".join("Environment=" + quote(key + "=" + value) for key, value in env.items()))


def install():
    command(["systemctl", "--user", "show", "--property=Version"])
    UNIT_DIR.mkdir(parents=True, exist_ok=True)
    LINK_DIR.mkdir(parents=True, exist_ok=True)
    old_path = UNIT_DIR / "managed.json"
    old = json.loads(old_path.read_text()) if old_path.exists() else {}
    files = {unit(name): render(name) for name in ORDER}
    files["mmassistant.target"] = "[Unit]\nDescription=Local assistant backend group\n"
    for name, content in files.items():
        path, linked = UNIT_DIR / name, LINK_DIR / name
        if linked.exists() or linked.is_symlink():
            if not linked.is_symlink() or linked.resolve() != path.resolve():
                raise RuntimeError("Refusing to replace an unrelated user unit: " + str(linked))
        if path.exists() and path.read_text() != content:
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            if old.get(name) != observed:
                raise RuntimeError("Managed unit has local edits; preserve/review them before regenerating: " + str(path))
    with tempfile.TemporaryDirectory(prefix="unit-check-", dir=UNIT_DIR) as temporary:
        staged = Path(temporary)
        for name, content in files.items(): (staged / name).write_text(content)
        command(["systemd-analyze", "--user", "verify", *[str(staged / name) for name in files]])
        for name in files: (staged / name).replace(UNIT_DIR / name)
    for name in files:
        path = LINK_DIR / name
        if not path.is_symlink(): path.symlink_to(UNIT_DIR / name)
    command(["systemctl", "--user", "daemon-reload"])
    write_json(old_path, {name: hashlib.sha256(content.encode()).hexdigest() for name, content in files.items()})


def build_running():
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try: args = path.read_bytes().split(b"\0")
        except OSError: continue
        if b"--build-ort" in args and any(arg.endswith(b"deploy/vision/prepare.py") for arg in args):
            return int(path.parent.name)
    return None


def preflight(names):
    building = build_running()
    if building and any(name != "camera" for name in names):
        raise RuntimeError(f"GPU ORT build PID {building} is active; keep model services stopped until it completes")
    if "llama" in names: server_command()
    if "asr" in names:
        item = json.loads((ROOT / "models/manifest.json").read_text())["models"]["asr"]
        if not all((CACHE / "models" / entry["name"]).is_file() for entry in item["files"]):
            raise RuntimeError("ASR model artifacts are missing")
    if "speech" in names and not (CACHE / "models/matcha-host/host-manifest.json").is_file():
        raise RuntimeError("Host-compatible speech engines are missing")
    if any(name in names for name in ("vision", "pose")):
        prefix = CACHE / "install/onnxruntime-1.28.0-cuda13-sm87"
        for suffix in (".ort-build-info", "lib/libonnxruntime_providers_cuda.so", "lib/libonnxruntime_providers_tensorrt.so"):
            if not (prefix / suffix).is_file(): raise RuntimeError("Full GPU ORT installation is incomplete")
        if not (CACHE / "workspaces/vision/install/yolo_ros/lib/yolo_ros/yolo_node").is_file():
            raise RuntimeError("ROS vision executable is not installed")
    for name in names:
        owned = set(properties(name)["cgroup_pids"])
        if name in {"camera", "vision", "pose"}:
            for path in Path("/proc").glob("[0-9]*/cmdline"):
                if int(path.parent.name) in owned: continue
                try: args = path.read_bytes().split(b"\0")
                except OSError: continue
                if not args: continue
                executable = Path(os.fsdecode(args[0])).name
                matches = ((name == "camera" and executable == "realsense2_camera_node") or
                           (name in {"vision", "pose"} and executable == "yolo_node" and f"__ns:=/{name}".encode() in args))
                if matches:
                    raise RuntimeError(f"Unmanaged {name} process PID {path.parent.name} is already present")
        port = PORTS.get(name)
        if port and properties(name).get("ActiveState") != "active":
            with socket.socket() as sock:
                sock.settimeout(.2)
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    raise RuntimeError(f"Port {port} is occupied by an unmanaged process; no process was replaced")


def owns_port(component, pids):
    """Match the loopback listening socket to a process in the unit cgroup."""
    port = PORTS[component]
    inodes = set()
    for line in Path("/proc/net/tcp").read_text().splitlines()[1:]:
        fields = line.split()
        address, number = fields[1].split(":")
        if int(number, 16) == port and fields[3] == "0A" and address in {"0100007F", "00000000"}:
            inodes.add(fields[9])
    for pid in pids:
        try:
            for fd in (Path("/proc") / str(pid) / "fd").iterdir():
                try: target = os.readlink(fd)
                except OSError: continue
                if target.startswith("socket:[") and target[8:-1] in inodes: return True
        except FileNotFoundError: pass
    return False


def probe(component):
    if component in PORTS:
        suffix = "/v1/capabilities" if component == "speech" else "/health"
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORTS[component]}" + suffix, timeout=3) as response:
                data = json.load(response)
            ready = all(data[kind]["ready"] for kind in ("asr", "tts")) if component == "speech" else data.get("status") == "ok"
            return {"ready": bool(ready), "scope": "local backend readiness"}
        except (OSError, ValueError, KeyError): return {"ready": False, "reason": "health_endpoint_unavailable"}
    setups = ["/opt/ros/jazzy/setup.bash"]
    if component != "camera": setups.append(str(CACHE / "workspaces/vision/install/setup.bash"))
    script = " && ".join("source " + shlex.quote(path) for path in setups)
    script += " && " + shlex.join(["/usr/bin/python3", str(ROOT / "tools/ros_health.py"), component])
    env = os.environ.copy()
    env.update(ROS_DOMAIN_ID=env.get("ASSISTANT_ROS_DOMAIN_ID", "42"), ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST")
    result = command(["bash", "-c", script], timeout=8, check=False, env=env)
    try: return json.loads(result.stdout)
    except ValueError: return {"ready": False, "reason": "ROS_probe_failed"}


def state(names, *, health=False):
    rows = {}
    for name in names:
        row = properties(name)
        row["managed_by_project"] = bool(row.get("FragmentPath") and
            Path(row["FragmentPath"]).resolve() == (UNIT_DIR / unit(name)).resolve())
        row["health"] = probe(name) if health and row.get("ActiveState") == "active" else {"ready": False}
        if name in PORTS and row["health"]["ready"]:
            row["health"]["owned_listening_socket"] = owns_port(name, row["cgroup_pids"])
            row["health"]["ready"] = row["health"]["owned_listening_socket"]
        rows[name] = row
    ready = bool(rows) and all(row["managed_by_project"] and row["main_pid"] > 0 and row["health"]["ready"] for row in rows.values())
    stopped = all(row.get("LoadState") == "not-found" or row.get("ActiveState") == "inactive" for row in rows.values())
    return {"created_at": stamp(), "components": rows, "backend_status": "READY" if ready else "STOPPED" if stopped else "DEGRADED",
            "health_checked": health, "assistant_acceptance": "NOT_VERIFIED",
            "scope": "backend supervision/readiness only; interactive or device pipelines have separate acceptance"}


def stop(names):
    # Explicit shutdown also stops consumers of the selected dependencies.
    expanded = set(names)
    for name, deps in DEPENDENCIES.items():
        if expanded.intersection(deps): expanded.add(name)
    for name in reversed(ORDER):
        if name not in expanded: continue
        before = properties(name)
        if before.get("LoadState") == "not-found": continue
        fragment = before.get("FragmentPath")
        if fragment and Path(fragment).resolve() != (UNIT_DIR / unit(name)).resolve():
            raise RuntimeError("Refusing to stop an unrelated unit with the same name")
        command(["systemctl", "--user", "stop", unit(name)], timeout=30)
        after = properties(name)
        if after["main_pid"] or after["cgroup_pids"]:
            raise RuntimeError("A backend process remains after systemd stop: " + name)
    if all(properties(name).get("ActiveState") in {"inactive", "failed", None} for name in ORDER):
        command(["systemctl", "--user", "stop", "mmassistant.target"], check=False)
    return state([name for name in ORDER if name in expanded])


def start(names, *, timeout=300):
    preflight(names)
    install()
    existing = {name for name in names if properties(name).get("ActiveState") == "active"}
    started = []
    try:
        command(["systemctl", "--user", "start", "mmassistant.target"])
        for name in names:
            if name not in existing:
                if name != "camera" and meminfo().get("MemAvailable", 0) < 2 * 1024**3:
                    raise RuntimeError("Insufficient free memory to admit another model service")
                command(["systemctl", "--user", "reset-failed", unit(name)], check=False)
                command(["systemctl", "--user", "start", unit(name)])
                started.append(name)
            deadline = time.monotonic() + timeout
            while True:
                row = properties(name)
                if row.get("ActiveState") == "failed": raise RuntimeError("Backend failed: " + name)
                if row.get("ActiveState") == "active" and probe(name)["ready"]: break
                if time.monotonic() >= deadline: raise RuntimeError("Readiness timeout: " + name)
                time.sleep(.25)
        result = state(names, health=True)
        if result["backend_status"] != "READY": raise RuntimeError("A backend stopped being ready during startup")
        if meminfo().get("MemAvailable", 0) < 2 * 1024**3:
            raise RuntimeError("Started profile has less than the initial 2 GiB memory headroom")
        return result
    except BaseException:
        for name in reversed(started): command(["systemctl", "--user", "stop", unit(name)], timeout=30, check=False)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["install", "up", "down", "status", "logs"])
    parser.add_argument("--components", default="all")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--lines", type=int, default=50)
    args = parser.parse_args(argv)
    names = selected(args.components, dependencies=args.action == "up")
    if args.action == "logs":
        if len(names) != 1 or not 1 <= args.lines <= 1000: raise ValueError("logs requires one component and 1..1000 lines")
        path = DATA / "logs/backends" / (names[0] + ".log")
        with path.open(encoding="utf-8", errors="replace") as stream:
            print("".join(deque(stream, maxlen=args.lines)), end="")
        return 0
    if args.action == "install": install(); result = state(names)
    elif args.action == "up": result = start(names, timeout=args.timeout)
    elif args.action == "down": result = stop(names)
    else: result = state(names, health=True)
    write_json(DATA / "reports/services-latest.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
