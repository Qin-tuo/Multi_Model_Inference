"""Measurement-only helpers; no runtime scheduling or inference changes."""

import csv
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time

import psutil

from deploy import runtime

ROOT = Path(__file__).resolve().parents[2]


def run_worker(argv, log, seconds=300):
    """Each benchmark owns one process group, including its tegrastats child."""
    child = subprocess.Popen(
        argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    try:
        return child.wait(seconds)
    finally:
        # On an error/timeout, also reap helpers whose parent failed before its
        # normal finally block. No managed/runtime service is in this group.
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if child.poll() is None:
            try:
                child.wait(10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(10)


def distribution(values):
    ordered = sorted(values)
    if any(not math.isfinite(v) or v < 0 for v in ordered):
        raise ValueError("Invalid duration")

    def percentile(q):
        return ordered[max(0, math.ceil(len(ordered) * q) - 1)] if ordered else None

    return {
        "count": len(ordered),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "max": max(ordered) if ordered else None,
        "method": "nearest-rank",
    }


def jsonable(value):
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def matching(left, right):
    """Numerical parity, not a semantic/accuracy ground truth."""
    left, right = jsonable(left), jsonable(right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            matching(left[k], right[k]) for k in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            matching(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return math.isclose(left, right, rel_tol=1e-5, abs_tol=1e-5)
    return left == right


class Events:
    def __init__(self, path):
        self.stream = Path(path).open("w")
        self.lock = threading.Lock()

    def write(self, **row):
        with self.lock:
            self.stream.write(
                json.dumps(
                    {"observed_monotonic": time.monotonic(), **jsonable(row)},
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )

    def close(self):
        self.stream.close()


class CallTimes:
    def __init__(self):
        self.calls = []

    def wrap(self, label, function):
        def measured(*args, **kwargs):
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                end = time.perf_counter()
                self.calls.append(
                    {
                        "label": label,
                        "start": start,
                        "end": end,
                        "ms": (end - start) * 1000,
                    }
                )

        return measured


class NativeProxy:
    """Time selected bound methods, forwarding every other attribute unchanged."""

    def __init__(self, target, times, labels):
        self.target, self.times, self.labels = target, times, labels

    def __getattr__(self, name):
        value = getattr(self.target, name)
        return (
            self.times.wrap(self.labels[name], value) if name in self.labels else value
        )



def environment(out, *, source_root=None):
    source_root = ROOT if source_root is None else Path(source_root).resolve()

    def command(args):
        result = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {
            "argv": args,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    sources = [
        p
        for folder in ("01_src/app", "01_src/benchmarks", "01_src/deploy")
        for p in (source_root / folder).glob("**/*.py")
    ]
    sources += list((source_root / "02_config").glob("*.yaml"))
    sources += list((source_root / "scripts").glob("*.py"))
    if (source_root / "01_src/tooling.py").is_file():
        sources.append(source_root / "01_src/tooling.py")
    sources += [
        source_root / "01_src/project_paths.py",
        source_root / "run_web_vision_chat.py",
        source_root / "main.py",
    ]
    result = {
        "created_at": runtime.stamp(),
        "pid": os.getpid(),
        "kernel": command(["uname", "-a"]),
        "power": command(["sudo", "-n", "nvpmodel", "-q"]),
        "source_root": str(source_root),
        "git_commit": command(["git", "-C", str(source_root), "rev-parse", "HEAD"]),
        "l4t": Path("/etc/nv_tegra_release").read_text(),
        "python": os.sys.version,
        "cuda_version_manifest": json.loads(
            Path("/usr/local/cuda/version.json").read_text()
        ),
        "opencv": command(
            [
                os.sys.executable,
                "-c",
                "import cv2;print(cv2.__version__);print(cv2.getNumThreads())",
            ]
        ),
        "background_processes": [
            {k: p.info[k] for k in ("pid", "ppid", "name", "create_time")}
            for p in psutil.process_iter(["pid", "ppid", "name", "create_time"])
        ],
        "source_sha256": {
            str(p.relative_to(source_root)): runtime.sha256(p) for p in sources
        },
    }
    coordinator_sources = []
    if source_root != ROOT:
        coordinator_sources = [
            Path(__file__),
            ROOT / "01_src/benchmarks/vision_modules.py",
            ROOT / "01_src/benchmarks/vision_protocol.py",
            ROOT / "01_src/deploy/vision/build.py",
        ]
    if coordinator_sources:
        result["coordinator_source_sha256"] = {
            str(p.relative_to(ROOT)): runtime.sha256(p) for p in coordinator_sources
        }
    runtime.write_json(out / "environment.json", result)
    for p in sources:
        target = out / "source" / p.relative_to(source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    for p in coordinator_sources:
        target = out / "coordinator-source" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    return result


class Resources:
    """One-second host/process samples and owned tegrastats output."""

    def __init__(self, out, groups):
        self.out, self.groups = out, groups
        self.stop = threading.Event()
        self.phase = "setup"
        self.errors = []
        self.thread = None
        self.stats = None
        self.stats_log = None

    def start(self):
        self.clock_anchor = {
            "monotonic": time.monotonic(),
            "wall_time_unix": time.time(),
            "utc": runtime.stamp(),
        }
        self.stats_log = (self.out / "tegrastats.log").open("w")
        self.stats = subprocess.Popen(
            ["tegrastats", "--interval", "1000"],
            stdout=self.stats_log,
            stderr=subprocess.STDOUT,
        )
        self.thread = threading.Thread(
            target=self._loop, name="baseline-resources", daemon=True
        )
        self.thread.start()
        return self

    def _loop(self):
        fields = [
            "monotonic",
            "phase",
            "mem_available",
            "swap_used",
            "cpu_percent",
            "gpu_load_percent",
        ]
        for name in self.groups:
            fields += [
                name + "_" + metric
                for metric in ("pss", "rss", "fds", "threads", "cpu_seconds")
            ]
        with (self.out / "resources.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            psutil.cpu_percent()
            while not self.stop.is_set():
                started = time.monotonic()
                try:
                    row = {
                        "monotonic": started,
                        "phase": self.phase,
                        "mem_available": psutil.virtual_memory().available,
                        "swap_used": psutil.swap_memory().used,
                        "cpu_percent": psutil.cpu_percent(),
                        "gpu_load_percent": None,
                    }
                    for path in [
                        "/sys/devices/platform/gpu.0/load",
                        "/sys/devices/platform/17000000.gpu/load",
                        "/sys/devices/gpu.0/load",
                    ]:
                        if Path(path).exists():
                            row["gpu_load_percent"] = int(Path(path).read_text()) / 10
                            break
                    for name, pids in self.groups.items():
                        totals = dict(pss=0, rss=0, fds=0, threads=0, cpu_seconds=0.0)
                        for pid in pids:
                            process = psutil.Process(pid)
                            mem = process.memory_full_info()
                            cpu = process.cpu_times()
                            for key, value in dict(
                                pss=mem.pss,
                                rss=mem.rss,
                                fds=process.num_fds(),
                                threads=process.num_threads(),
                                cpu_seconds=cpu.user + cpu.system,
                            ).items():
                                totals[key] += value
                        row.update({name + "_" + k: v for k, v in totals.items()})
                    writer.writerow(row)
                except Exception as exc:
                    self.errors.append({"monotonic": started, "error": repr(exc)})
                self.stop.wait(max(0, 1 - (time.monotonic() - started)))

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(5)
            if self.thread.is_alive():
                raise RuntimeError("Resource sampler failed to release")
        if self.stats:
            self.stats.terminate()
            self.stats.wait(5)
            self.stats_log.close()
        runtime.write_json(
            self.out / "monitor.json",
            {
                "errors": self.errors,
                "clock_anchor": self.clock_anchor,
                "clock_end": {
                    "monotonic": time.monotonic(),
                    "wall_time_unix": time.time(),
                },
                "process_groups": self.groups,
                "scope": "Host unified memory and stable process PSS/RSS are separate. GPU/temperature/clocks in raw tegrastats; no summing RSS with GPU memory.",
            },
        )
