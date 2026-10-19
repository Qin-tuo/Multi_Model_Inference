# SPDX-License-Identifier: Apache-2.0
"""Thin systemd CLI for the selected services; upstream servers own inference."""

import argparse
from collections import deque
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy import runtime  # noqa: E402 — direct script entrypoint

# 按依赖先启动后端，最后启动应用；停止时反向处理依赖。
# 这里管理进程生命周期，不参与 GPU 推理或应用轮次调度。
ORDER = ("llama", "asr", "speech", "app")
DEPENDENCIES = {"speech": ("asr",), "app": ("llama", "asr", "speech")}
PREFIX = "reachy-headless"
LINK_DIR = Path.home() / ".config/systemd/user"


def unit(name):
    return f"{PREFIX}-{name}.service"


def command(argv, timeout=45, check=True):
    value = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if check and value.returncode:
        raise RuntimeError((value.stderr or value.stdout)[-2000:])
    return value


def selected(value, dependencies=False):
    names = set(ORDER if value == "all" else value.split(","))
    if not names or names - set(ORDER):
        raise ValueError("Unknown service component")
    if dependencies:
        for name in list(names):
            names.update(DEPENDENCIES.get(name, ()))
    return [name for name in ORDER if name in names]


def quote(value, executable=False):
    value = str(value)
    if any(c in value for c in "\n\r\x00"):
        raise ValueError("Invalid systemd argument")
    value = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    if executable:
        value = value.replace("$", "$$")
    return '"' + value + '"'


def directory(cfg):
    return runtime.paths(cfg)["data"] / "systemd"


def profile_path(cfg):
    return directory(cfg) / "settings.json"


def properties(name):
    fields = [
        "LoadState",
        "ActiveState",
        "SubState",
        "MainPID",
        "ControlGroup",
        "FragmentPath",
        "NRestarts",
        "Result",
    ]
    value = command(
        [
            "systemctl",
            "--user",
            "show",
            unit(name),
            *[f"--property={f}" for f in fields],
        ],
        check=False,
    )
    result = dict(
        line.split("=", 1) for line in value.stdout.splitlines() if "=" in line
    )
    if not result.get("LoadState"):
        raise RuntimeError(
            "Cannot observe systemd state for "
            + unit(name)
            + ": "
            + (value.stderr or value.stdout)[-1000:]
        )
    result["main_pid"] = int(result.get("MainPID") or 0)
    result["cgroup_pids"] = []
    if group := result.get("ControlGroup"):
        try:
            result["cgroup_pids"] = sorted(
                {
                    int(pid)
                    for file in (Path("/sys/fs/cgroup") / group.lstrip("/")).rglob(
                        "cgroup.procs"
                    )
                    for pid in file.read_text().split()
                }
            )
        except FileNotFoundError:
            pass
    return result


def owns_port(port, pids):
    inodes = set()
    for line in Path("/proc/net/tcp").read_text().splitlines()[1:]:
        row = line.split()
        address, number = row[1].split(":")
        if address == "0100007F" and int(number, 16) == port and row[3] == "0A":
            inodes.add(row[9])
    for pid in pids:
        try:
            for fd in (Path("/proc") / str(pid) / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if target.startswith("socket:[") and target[8:-1] in inodes:
                    return True
        except FileNotFoundError:
            pass
    return False


def health(name, cfg):
    from app.request_io import control_json

    listen = runtime.ports(cfg)
    if name in {"llama", "asr"}:
        base = f"http://127.0.0.1:{listen[name]}"
        models = control_json(base + "/v1/models", 1)
        expected = cfg.llm.model if name == "llama" else cfg.backends.asr_model_id
        if expected not in {m.get("id") for m in models.get("data", [])}:
            return {"ready": False, "reason": "model_alias_mismatch"}
        slots = control_json(base + "/slots", 1)
        context = (
            cfg.llm.context_tokens
            if name == "llama"
            else cfg.backends.asr_context_tokens
        )
        ready = (
            isinstance(slots, list)
            and len(slots) == 1
            and slots[0].get("n_ctx") == context
            and type(slots[0].get("is_processing")) is bool
        )
        return {
            "ready": ready,
            "model": expected,
            "context_tokens": context,
            "slots": len(slots),
            "processing": sum(bool(s.get("is_processing")) for s in slots),
        }
    if name == "speech":
        caps = control_json(cfg.tts.base_url + "/capabilities", 1)
        valid = True
        rows = {}
        for role, model, backend in [
            ("asr", cfg.stt.model, cfg.stt.backend),
            ("tts", cfg.tts.model, cfg.tts.backend),
        ]:
            value = caps.get(role, {})
            good = (
                value.get("ready") is True
                and value.get("model_id") == model
                and value.get("backend") == backend
            )
            good = (
                good
                and value.get("audio", {}).get("sample_rate") == 16000
                and value.get("concurrency", {}).get("admission_limit") == 1
            )
            valid = valid and good
            rows[role] = {
                "ready": good,
                "model": value.get("model_id"),
                "backend": value.get("backend"),
                "concurrency": value.get("concurrency"),
            }
        return {"ready": valid, "capabilities": rows}
    value = control_json(
        f"http://127.0.0.1:{listen['app']}/api/state", 2, max_bytes=1024 * 1024
    )
    ready = value.get("ready") is True and value.get("camera_ready") is True
    if cfg.perception.enabled:
        ready = ready and value.get("perception", {}).get("ready") is True
    if cfg.audio.enabled:
        ready = ready and value.get("speech_ready") is True
    if cfg.actions.enabled:
        ready = (
            ready
            and value.get("actions", {}).get("enabled") is True
            and not value["actions"].get("error")
        )
    ready = ready and value.get("effective_config") == cfg.effective()
    return {
        "ready": bool(ready),
        "camera": value.get("camera"),
        "phase": value.get("phase"),
        "models": value.get("models"),
        "config_matches": value.get("effective_config") == cfg.effective(),
    }


def state(names, cfg, with_health=True):
    rows = {}
    listen = runtime.ports(cfg)
    for name in names:
        row = properties(name)
        fragment = row.get("FragmentPath")
        row["managed_by_project"] = bool(
            fragment
            and Path(fragment).resolve() == (directory(cfg) / unit(name)).resolve()
        )
        row["health"] = {"ready": False}
        if with_health and row.get("ActiveState") == "active":
            try:
                row["health"] = health(name, cfg)
            except Exception as exc:
                row["health"] = {"ready": False, "error": str(exc)}
            row["owned_listening_socket"] = owns_port(listen[name], row["cgroup_pids"])
            row["health"]["ready"] = (
                row["health"]["ready"] and row["owned_listening_socket"]
            )
        rows[name] = row
    ready = bool(rows) and all(
        r["managed_by_project"] and r["main_pid"] and r["health"]["ready"]
        for r in rows.values()
    )
    return {
        "created_at": runtime.stamp(),
        "services": rows,
        "status": "READY"
        if ready
        else "STOPPED"
        if all(
            not r["cgroup_pids"]
            and r.get("ActiveState") in {"inactive", None}
            and r.get("LoadState") != "error"
            for r in rows.values()
        )
        else "DEGRADED",
        "profile": str(profile_path(cfg)),
        "scope": "Managed software readiness; physical audio is deferred, not qualified by this check",
    }


def render(name, cfg):
    p = runtime.paths(cfg)
    profile = profile_path(cfg)
    deps = DEPENDENCIES.get(name, ())
    argv = [
        p["app_python"],
        ROOT / "deploy/logged_service.py",
        name,
        "--profile",
        profile,
    ]
    pre = ""
    if deps:
        waiting = [
            p["app_python"],
            ROOT / "deploy/services.py",
            "wait",
            "--config",
            profile,
            "--components",
            ",".join(deps),
            "--timeout",
            "180",
        ]
        pre = "ExecStartPre=" + " ".join(quote(a, True) for a in waiting) + "\n"
    return (
        f"# Managed by Reachy headless for {ROOT}\n[Unit]\nDescription=Reachy headless {name}\n"
        f"PartOf={PREFIX}.target\nAfter={' '.join(unit(d) for d in deps)}\nWants={' '.join(unit(d) for d in deps)}\n"
        "StartLimitIntervalSec=120\nStartLimitBurst=3\n\n[Service]\nType=exec\n"
        f"WorkingDirectory={str(ROOT).replace('%', '%%')}\n{pre}ExecStart={' '.join(quote(a, True) for a in argv)}\n"
        'Environment="PYTHONUNBUFFERED=1" "HF_HUB_OFFLINE=1" "TRANSFORMERS_OFFLINE=1" "OTEL_SDK_DISABLED=true"\n'
        "Restart=on-failure\nRestartSec=5\nSuccessExitStatus=130\nKillMode=control-group\nKillSignal=SIGINT\n"
        "TimeoutStartSec=240\nTimeoutStopSec=30\nSendSIGKILL=yes\nMemoryAccounting=yes\nCPUAccounting=yes\nUMask=0077\n"
        f"StandardOutput=journal\nStandardError=journal\nSyslogIdentifier={PREFIX}-{name}\n"
    )


def install(cfg):
    folder = directory(cfg)
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    LINK_DIR.mkdir(parents=True, exist_ok=True)
    record = folder / "managed.json"
    old = json.loads(record.read_text()) if record.exists() else {}
    files = {unit(name): render(name, cfg) for name in ORDER}
    files[PREFIX + ".target"] = (
        "[Unit]\nDescription=Reachy headless selected local services\n"
    )
    profile = json.dumps(cfg.effective(), ensure_ascii=False, indent=2) + "\n"
    changed = profile_path(cfg).exists() and profile_path(cfg).read_text() != profile
    changed = changed or any(
        (folder / name).exists() and (folder / name).read_text() != body
        for name, body in files.items()
    )
    if changed and any(properties(name)["cgroup_pids"] for name in ORDER):
        raise RuntimeError(
            "Stop managed services before replacing their profile or entrypoint"
        )
    for name, body in files.items():
        link = LINK_DIR / name
        path = folder / name
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve() != path.resolve():
                raise RuntimeError("Refusing unrelated unit: " + str(link))
        if (
            path.exists()
            and path.read_text() != body
            and old.get(name) != runtime.sha256(path)
        ):
            raise RuntimeError("Managed unit has local edits: " + str(path))
    with tempfile.TemporaryDirectory(prefix="unit-check-", dir=folder) as temporary:
        staged = Path(temporary)
        for name, body in files.items():
            (staged / name).write_text(body)
        command(
            [
                "systemd-analyze",
                "--user",
                "verify",
                *[str(staged / name) for name in files],
            ]
        )
        for name in files:
            (staged / name).replace(folder / name)
    runtime.write_json(profile_path(cfg), cfg.effective())
    for name in files:
        if not (LINK_DIR / name).is_symlink():
            (LINK_DIR / name).symlink_to(folder / name)
    command(["systemctl", "--user", "daemon-reload"])
    runtime.write_json(
        record,
        {
            name: hashlib.sha256(body.encode()).hexdigest()
            for name, body in files.items()
        },
    )


def preflight(names, cfg):
    listen = runtime.ports(cfg)
    for name in names:
        row = properties(name)
        if row["cgroup_pids"]:
            if not row.get("FragmentPath") or Path(
                row["FragmentPath"]
            ).resolve() != directory(cfg) / unit(name):
                raise RuntimeError("Unowned service: " + name)
            continue
        with socket.socket() as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", listen[name])) == 0:
                raise RuntimeError(
                    f"Port {listen[name]} is occupied outside {unit(name)}; no process was stopped"
                )
    if "app" in names:
        camera = Path(cfg.vision.camera_device)
        if not camera.exists():
            raise RuntimeError("Configured camera is unavailable")
        import psutil

        target = str(camera.resolve())
        owned = set(properties("app")["cgroup_pids"])
        for process in psutil.process_iter(["pid"]):
            if process.pid in owned:
                continue
            try:
                for fd in (Path("/proc") / str(process.pid) / "fd").iterdir():
                    try:
                        link = os.readlink(fd)
                    except OSError:
                        continue
                    if link == target:
                        raise RuntimeError(
                            f"Camera is held by PID {process.pid}; no process was stopped"
                        )
            except (FileNotFoundError, PermissionError):
                continue
    return runtime.verify(cfg, names)


def wait_ready(names, cfg, seconds):
    deadline = time.monotonic() + seconds
    while True:
        value = state(names, cfg)
        if value["status"] == "READY":
            return value
        if any(
            row.get("ActiveState") == "failed" for row in value["services"].values()
        ):
            raise RuntimeError(
                "Managed service entered failed state: "
                + json.dumps(value, ensure_ascii=False)
            )
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Readiness timeout: " + json.dumps(value, ensure_ascii=False)
            )
        time.sleep(0.25)


def require_headroom():
    # Preserve the previous deployment admission floor; this is not a memory
    # reservation and does not qualify peak multi-model capacity.
    import psutil

    if psutil.virtual_memory().available < 2 * 1024**3:
        raise RuntimeError(
            "Less than 2 GiB available memory; another model service was not admitted"
        )


def start(names, cfg, seconds):
    """校验工件与归属，顺序启动并检查真实就绪；失败回收本次新启动项。"""
    artifacts = preflight(names, cfg)
    install(cfg)
    started = []
    try:
        command(["systemctl", "--user", "start", PREFIX + ".target"])
        for name in names:
            if properties(name).get("ActiveState") != "active":
                require_headroom()
                command(
                    ["systemctl", "--user", "reset-failed", unit(name)], check=False
                )
                command(["systemctl", "--user", "start", "--no-block", unit(name)])
                started.append(name)
            wait_ready([name], cfg, seconds)
        result = state(names, cfg)
        if result["status"] != "READY":
            raise RuntimeError("A managed service lost readiness during startup")
        require_headroom()
        result["verified_artifact_count"] = len(artifacts)
        return result
    except BaseException:
        for name in reversed(started):
            command(
                ["systemctl", "--user", "stop", unit(name)], timeout=45, check=False
            )
        raise


def stop(names, cfg):
    """连同消费者一起停止指定组件，核验对应 cgroup 中没有残留进程。"""
    expanded = set(names)
    for name, deps in DEPENDENCIES.items():
        if expanded.intersection(deps):
            expanded.add(name)
    for name in reversed(ORDER):
        if name not in expanded:
            continue
        row = properties(name)
        if row.get("LoadState") == "not-found":
            continue
        if row.get("FragmentPath") and Path(row["FragmentPath"]).resolve() != directory(
            cfg
        ) / unit(name):
            raise RuntimeError("Refusing to stop an unrelated service")
        command(["systemctl", "--user", "stop", unit(name)])
        if properties(name)["cgroup_pids"]:
            raise RuntimeError("Service processes remain: " + name)
    if all(not properties(name)["cgroup_pids"] for name in ORDER):
        command(["systemctl", "--user", "stop", PREFIX + ".target"], check=False)
    return state([name for name in ORDER if name in expanded], cfg, False)


@contextmanager
def operation(cfg):
    directory(cfg).mkdir(parents=True, exist_ok=True)
    with (directory(cfg) / "operation.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("Another deployment operation is active") from None
        yield


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["install", "up", "down", "status", "wait", "logs", "check"]
    )
    parser.add_argument("--config")
    parser.add_argument("--components", default="all")
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--lines", type=int, default=50)
    args = parser.parse_args(argv)
    cfg = runtime.config(args.config)
    if (
        not args.config
        and args.action in {"status", "down", "logs", "wait"}
        and profile_path(cfg).is_file()
    ):
        cfg = runtime.config(profile_path(cfg))
    names = selected(args.components, args.action == "up")
    if not 1 <= args.timeout <= 600:
        raise ValueError("Timeout must be 1..600 seconds")
    if args.action == "logs":
        if len(names) != 1 or not 1 <= args.lines <= 1000:
            raise ValueError("logs requires one component and 1..1000 lines")
        with (runtime.paths(cfg)["data"] / "logs" / f"{names[0]}.log").open() as stream:
            print("".join(deque(stream, maxlen=args.lines)), end="")
        return 0
    if args.action in {"install", "up", "down"}:
        with operation(cfg):
            if args.action == "install":
                runtime.verify(cfg, names)
                install(cfg)
                result = state(names, cfg, False)
            elif args.action == "up":
                result = start(names, cfg, args.timeout)
            else:
                result = stop(names, cfg)
    elif args.action == "wait":
        result = wait_ready(names, cfg, args.timeout)
    elif args.action == "check":
        result = {
            "status": "ARTIFACTS_VERIFIED",
            "files": runtime.verify(cfg, names),
            "commands": {
                n: runtime.native_command(n, cfg, profile_path(cfg)) for n in names
            },
        }
    else:
        result = state(names, cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return (
        0
        if result["status"] in {"READY", "STOPPED", "ARTIFACTS_VERIFIED"}
        or args.action == "install"
        else 1
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(
            json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        raise SystemExit(1)
