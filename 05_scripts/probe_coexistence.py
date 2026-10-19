#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Native coexistence/pressure check with isolated data and live D435i perception.

Temporarily replaces only the managed app; native backends stay resident. Uses
production inference/admission limits. It records every outcome and restores
the managed app in finally. No physical audio or performance baseline claim.
"""

import argparse
import asyncio
from collections import deque
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import httpx
import psutil
from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "01_src"))
from deploy import runtime, services  # noqa: E402


def backend_identity(name, cfg):
    row = services.properties(name)
    port = runtime.ports(cfg)[name]
    listeners = {
        c.pid
        for c in psutil.net_connections(kind="tcp")
        if c.status == psutil.CONN_LISTEN
        and c.laddr.ip == "127.0.0.1"
        and c.laddr.port == port
    }
    if len(listeners) != 1 or not listeners <= set(row["cgroup_pids"]):
        raise RuntimeError("Native listener has no unambiguous owned process")
    stable = sorted({row["main_pid"], *listeners})
    return {
        "identity": {
            "main_pid": row["main_pid"],
            "restarts": row["NRestarts"],
            "processes": {
                str(pid): psutil.Process(pid).create_time() for pid in stable
            },
        },
        "stable_pids": stable,
        "observed_cgroup_pids": row["cgroup_pids"],
    }


def process_sample(groups):
    rows = {}
    for name, pids in groups.items():
        total = dict(pss=0, rss=0, threads=0, fds=0, cpu_seconds=0.0)
        for pid in pids:
            process = psutil.Process(pid)
            memory = process.memory_full_info()
            cpu = process.cpu_times()
            total["pss"] += memory.pss
            total["rss"] += memory.rss
            total["threads"] += process.num_threads()
            total["fds"] += process.num_fds()
            total["cpu_seconds"] += cpu.user + cpu.system
        rows[name] = total
    return rows


async def exercise(cfg, child, out, report, input_file):
    base = f"http://127.0.0.1:{cfg.web.port}"
    report["backend_identity_before"] = {
        name: backend_identity(name, cfg) for name in ("llama", "asr", "speech")
    }
    groups = {
        name: row["stable_pids"]
        for name, row in report["backend_identity_before"].items()
    }
    report["process_resource_scope"] = (
        "Stable native listener and supervisor PIDs; transient OVS watchdog children excluded from per-service sums, included in host memory and tegrastats."
    )
    groups["app"] = [child.pid]
    report["observed_pids"] = groups
    events = deque(maxlen=512)
    stop = asyncio.Event()
    phase = "warmup"
    monitor_errors = []
    received_frames = 0
    output_events = (out / "events.jsonl").open("w")
    state_samples = (out / "state-samples.jsonl").open("w")
    resource_file = (out / "resources.csv").open("w", newline="")
    fields = ["monotonic", "phase", "mem_available", "swap_used"]
    for name in groups:
        fields.extend(
            f"{name}_{metric}"
            for metric in ("pss", "rss", "threads", "fds", "cpu_seconds")
        )
    writer = csv.DictWriter(resource_file, fieldnames=fields)
    writer.writeheader()

    def save():
        runtime.write_json(out / "report.json", report)

    async with httpx.AsyncClient(base_url=base, timeout=5, trust_env=False) as client:

        async def state():
            return (await client.get("/api/state")).raise_for_status().json()

        async def wait_for(predicate, seconds=150):
            async with asyncio.timeout(seconds):
                while True:
                    if child.poll() is not None:
                        raise RuntimeError("Isolated app exited")
                    value = await predicate()
                    if value:
                        return value
                    await asyncio.sleep(0.02)

        async def idle():
            value = await state()
            return value if value["active_turn"] is None else None

        async def monitor():
            next_tick = time.monotonic()
            while not stop.is_set():
                tick = time.monotonic()
                try:
                    value = await state()
                    scene = value["perception"]
                    sample = {
                        "monotonic": tick,
                        "phase": phase,
                        "camera": value["camera"],
                        "web": value["web"],
                        "web_reads": value["web_reads"],
                        "turn_phase": value["phase"],
                        "pending": value["pending"],
                        "session": value["session"],
                        "perception": {
                            k: scene.get(k)
                            for k in (
                                "ready",
                                "error",
                                "frame",
                                "completed_monotonic",
                                "elapsed_ms",
                                "stages",
                                "processed",
                                "skipped_capture_frames",
                                "failures",
                            )
                        },
                    }
                    state_samples.write(json.dumps(sample) + "\n")
                    if (
                        value["pending"] > 1
                        or value["camera"]["buffer_bytes"]
                        > cfg.vision.buffer_bytes_limit
                    ):
                        raise RuntimeError(
                            "Application queue/buffer exceeded configured bound"
                        )
                    if value["session"]["retained_turns"] > cfg.session.max_turns:
                        raise RuntimeError("Session exceeded retained-turn bound")
                    resources = await asyncio.to_thread(process_sample, groups)
                    row = {
                        "monotonic": tick,
                        "phase": phase,
                        "mem_available": psutil.virtual_memory().available,
                        "swap_used": psutil.swap_memory().used,
                    }
                    for name, values in resources.items():
                        row.update(
                            {f"{name}_{key}": value for key, value in values.items()}
                        )
                    writer.writerow(row)
                    if row["mem_available"] < 1024**3:
                        raise RuntimeError(
                            "Pressure probe stopped at 1 GiB available-memory guard"
                        )
                except Exception as exc:
                    monitor_errors.append({"monotonic": tick, "error": repr(exc)})
                    stop.set()
                    return
                next_tick += 1
                await asyncio.sleep(max(0, next_tick - time.monotonic()))

        async def complete_turn(kind, index):
            await wait_for(idle)
            if kind == "audio":
                response = await client.post(
                    "/api/audio",
                    content=input_file.read_bytes(),
                    headers={"Content-Type": "audio/wav"},
                )
            else:
                response = await client.post(
                    "/api/chat",
                    json={
                        "text": "用一句话描述当前画面，不执行工具。"
                        if kind == "camera"
                        else f"只回答：第{index}轮通过。",
                        "use_image": kind == "camera",
                    },
                )
            key = response.raise_for_status().json()["turn_id"]
            result = (await wait_for(idle))["last_result"]
            if (
                result["turn_id"] != key
                or result["status"] != "complete"
                or not result["resources_released"]
            ):
                raise RuntimeError(
                    "Completion/release or turn identity failed: "
                    + json.dumps(result, ensure_ascii=False)
                )
            reply = (await client.get(result["audio_url"])).raise_for_status()
            if not reply.content.startswith(b"RIFF"):
                raise RuntimeError("Completed reply is not a WAV")
            return result

        try:
            # Lazy HTTP workers and native allocations get an explicit warmup.
            await asyncio.gather(state(), state())
            await complete_turn("text", 0)
            await complete_turn("camera", 0)
            await complete_turn("audio", 0)
            await asyncio.sleep(5)
            report["before"] = process_sample(groups)
            report["initial_state"] = await state()
            observer = asyncio.create_task(monitor())
            async with connect(
                base.replace("http://", "ws://") + "/ws", max_size=2**21
            ) as ws:

                async def receive():
                    nonlocal received_frames
                    async for raw in ws:
                        message = json.loads(raw)
                        if message["type"] == "frame":
                            received_frames += 1
                        elif message.get("turn_id"):
                            events.append(message)
                            output_events.write(
                                json.dumps(
                                    {"received_monotonic": time.monotonic(), **message},
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )

                receiver = asyncio.create_task(receive())
                slow = []
                try:
                    for _ in range(3):
                        slow.append(
                            await connect(
                                base.replace("http://", "ws://") + "/ws",
                                max_size=2**21,
                                max_queue=1,
                            )
                        )
                    try:
                        extra = await connect(base.replace("http://", "ws://") + "/ws")
                    except Exception as exc:
                        response = getattr(exc, "response", None)
                        report["extra_viewer_status"] = getattr(
                            response, "status_code", None
                        )
                        if report["extra_viewer_status"] != 403:
                            raise
                    else:
                        await extra.close()
                        raise RuntimeError("Fifth viewer was unexpectedly admitted")
                    phase = "mixed_pressure"
                    for index in range(1, 21):
                        if stop.is_set() or receiver.done():
                            raise RuntimeError("Observer/control connection failed")
                        print(f"coexistence cycle {index}/20", flush=True)
                        await wait_for(idle)
                        response = await client.post(
                            "/api/chat",
                            json={
                                "text": "请详细介绍十种学习编程的方法，每条解释具体步骤，不执行任何工具。",
                                "use_image": index % 2 == 0,
                            },
                        )
                        key = response.raise_for_status().json()["turn_id"]

                        async def native_busy():
                            slots = (
                                (await client.get(cfg.llm.base_url + "/slots"))
                                .raise_for_status()
                                .json()
                            )
                            return any(slot["is_processing"] for slot in slots)

                        await wait_for(native_busy, 20)
                        burst = await asyncio.gather(
                            *(
                                client.post(
                                    "/api/chat",
                                    json={"text": "不应被接纳的并发测试请求"},
                                )
                                for _ in range(4)
                            )
                        )
                        codes = [reply.status_code for reply in burst]
                        if codes != [409] * 4:
                            raise RuntimeError(
                                "Unexpected admission under load: " + str(codes)
                            )
                        tick = time.monotonic()
                        await ws.send(json.dumps({"type": "cancel", "turn_id": key}))
                        final = await wait_for(idle, 60)
                        result = final["last_result"]
                        if (
                            result["turn_id"] != key
                            or result["status"] != "cancelled"
                            or not result["resources_released"]
                        ):
                            raise RuntimeError(
                                "Cancellation/release failed: "
                                + json.dumps(result, ensure_ascii=False)
                            )
                        if (await client.get("/api/audio/" + key)).status_code != 404:
                            raise RuntimeError("Cancelled output was published")
                        owned = [
                            event for event in events if event.get("turn_id") == key
                        ]
                        cancelled = next(
                            i
                            for i, event in enumerate(owned)
                            if event["type"] == "cancelled"
                        )
                        if any(
                            event["type"] in {"token", "transcript"}
                            for event in owned[cancelled + 1 :]
                        ):
                            raise RuntimeError("Late output after cancellation")
                        case = {
                            "cycle": index,
                            "rejected_statuses": codes,
                            "cancel_to_idle_seconds": time.monotonic() - tick,
                            "cancelled": result,
                            "cancelled_audio_404": True,
                            "late_output_after_cancel": False,
                        }
                        report["cases"].append(case)
                        case["completed"] = await complete_turn(
                            "camera" if index % 2 == 0 else "text", index
                        )
                        case["file_speech"] = await complete_turn("audio", index)
                        save()
                finally:
                    for viewer in slow:
                        await viewer.close()
                    receiver.cancel()
                    await asyncio.gather(receiver, return_exceptions=True)
            phase = "cooldown"
            print("cooldown observation: 30 seconds", flush=True)
            await asyncio.sleep(30)
            stop.set()
            await observer
            report["monitor_errors"] = monitor_errors
            report["after"] = process_sample(groups)
            report["final_state"] = await state()
            report["received_frames"] = received_frames
            if monitor_errors:
                raise RuntimeError("Resource observer reported errors")
            if report["final_state"]["web"]["clients"] != 0:
                raise RuntimeError("Viewers remain after cleanup")
            report["backend_identity_after"] = {
                name: backend_identity(name, cfg) for name in ("llama", "asr", "speech")
            }
            if any(
                report["backend_identity_before"][name]["identity"] != after["identity"]
                for name, after in report["backend_identity_after"].items()
            ):
                raise RuntimeError("A native backend restarted during pressure")
            report["pressure_status"] = "PASS"
        finally:
            stop.set()
            if "observer" in locals():
                await asyncio.gather(observer, return_exceptions=True)
            for stream in (output_events, state_samples, resource_file):
                stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    original = runtime.config(services.profile_path(runtime.config()))
    before = services.state(services.ORDER, original)
    if before["status"] != "READY":
        raise RuntimeError("Managed stack must be ready before this probe")
    cfg = runtime.config(services.profile_path(original))
    cfg.web.port = 8093
    cfg.actions.database = str(out / "actions.sqlite")
    cfg.audio.output_directory = str(out / "audio")
    cfg.validate()
    runtime.write_json(out / "settings.json", cfg.effective())
    report = {
        "created_at": runtime.stamp(),
        "scope": __doc__,
        "status": "RUNNING",
        "cases": [],
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "probe_sha256": runtime.sha256(__file__),
        "input_sha256": runtime.sha256(args.input),
        "config": cfg.effective(),
        "before_services": before,
    }
    child = stats = None
    log = (out / "app.log").open("w")
    stats_log = (out / "tegrastats.log").open("w")
    failed = None
    try:
        with services.operation(original):
            services.stop(["app"], original)
        services.preflight(["app"], cfg)
        child = subprocess.Popen(
            runtime.native_command("app", cfg, out / "settings.json"),
            cwd=ROOT,
            start_new_session=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        report["tegrastats_start_monotonic"] = time.monotonic()
        stats = subprocess.Popen(
            ["tegrastats", "--interval", "1000"],
            stdout=stats_log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 120
        while True:
            if child.poll() is not None:
                raise RuntimeError("Isolated application failed at startup")
            try:
                if services.health("app", cfg)["ready"]:
                    break
            except (httpx.HTTPError, OSError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("Isolated app startup timed out")
            time.sleep(0.25)
        asyncio.run(exercise(cfg, child, out, report, args.input))
    except BaseException as exc:
        failed = exc
        report["error"] = repr(exc)
    finally:
        if child is not None:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(10)
                    report["forced_probe_shutdown"] = True
                    failed = failed or RuntimeError("Probe needed forced shutdown")
            report["app_exit_code"] = child.returncode
            if child.returncode != 0:
                failed = failed or RuntimeError(
                    "Isolated app did not shut down cleanly"
                )
        if stats is not None:
            stats.terminate()
            stats.wait(10)
        log.close()
        stats_log.close()
        try:
            with services.operation(original):
                report["restored"] = services.start(["app"], original, 240)
        except BaseException as exc:
            report["restore_error"] = repr(exc)
            failed = failed or exc
        report["status"] = "FAIL" if failed else "PASS"
        report["finished_at"] = runtime.stamp()
        runtime.write_json(out / "report.json", report)
    if failed:
        raise failed
    print(json.dumps({"status": "PASS", "report": str(out / "report.json")}))


if __name__ == "__main__":
    main()
