#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Detach/rebind only configured D435i RGB UVC interface; verify the real app.

Requires explicit opt-in and passwordless sudo for the two sysfs writes. Stops
only this project's legacy camera service, owns an isolated application, then
restores its driver/service and verifies fresh legacy RGB plus aligned depth.
Saves frame metadata and aggregates, never camera images or facial landmarks.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_src"))
import httpx
import psutil
import uvicorn
from websockets.asyncio.client import connect

from app.config import Config
import run_web_vision_chat as entry


async def command(*args, timeout=45):
    child = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out, err = await asyncio.wait_for(child.communicate(), timeout)
    if child.returncode:
        raise RuntimeError(f"{args[0]} failed: " + err.decode().strip())
    return out.decode().strip()


async def probe(args):
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": __doc__,
        "status": "RUNNING",
    }
    cfg = Config.load()
    cfg.gestures.greetings_enabled = False
    device = Path(cfg.vision.camera_device)
    interface = (
        Path("/sys/class/video4linux") / device.resolve().name / "device"
    ).resolve()
    usb = interface.parent
    driver = Path("/sys/bus/usb/drivers/uvcvideo")
    serial = (usb / "serial").read_text().strip()
    assert (usb / "idVendor").read_text().strip() == "8086" and (
        usb / "idProduct"
    ).read_text().strip() == "0b3a"
    assert serial in str(device) and (interface / "driver").resolve() == driver
    assert (interface / "bInterfaceClass").read_text().strip() == "0e"
    await command("sudo", "-n", "true")
    active = (
        await command("systemctl", "--user", "is-active", "mmassistant-camera.service")
        == "active"
    )
    report["device"] = {
        "by_id": str(device),
        "interface": interface.name,
        "serial": serial,
        "speed_mbps": (usb / "speed").read_text().strip(),
    }
    detached = False
    server = running = listener = None
    legacy_stopped = False
    camera = None

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    async def driver_write(action):
        await command(
            "sudo",
            "-n",
            "/usr/bin/python3",
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2])",
            str(driver / action),
            interface.name,
        )

    async def wait(test, seconds=30):
        async with asyncio.timeout(seconds):
            while True:
                value = await test()
                if value:
                    return value
                await asyncio.sleep(0.05)

    def process():
        p = psutil.Process()
        return {
            "fds": p.num_fds(),
            "threads": p.num_threads(),
            "pss_bytes": p.memory_full_info().pss,
        }

    try:
        await command("systemctl", "--user", "stop", "mmassistant-camera.service")
        legacy_stopped = True
        factory = entry.Camera

        def capture(**kwargs):
            nonlocal camera
            camera = factory(**kwargs)
            return camera

        entry.Camera = capture
        try:
            app = entry.build_app(cfg, no_audio=True, no_actions=True)
        finally:
            entry.Camera = factory
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(32)
        port = listener.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=port,
                log_level="warning",
                ws_max_size=cfg.web.ws_message_bytes,
                ws_max_queue=4,
                limit_concurrency=16,
                timeout_graceful_shutdown=10,
            )
        )
        running = asyncio.create_task(server.serve(sockets=[listener]))

        async def started():
            if running.done():
                running.result()
                raise RuntimeError("Application exited before ready")
            return server.started

        await wait(started, 60)
        base = f"http://127.0.0.1:{port}"
        async with httpx.AsyncClient(
            base_url=base, timeout=5, trust_env=False
        ) as client:

            async def state():
                return (await client.get("/api/state")).raise_for_status().json()

            async def scene():
                value = await state()
                return (
                    value
                    if value["camera_ready"] and value["perception"]["ready"]
                    else None
                )

            initial = await wait(scene)
            before = camera.snapshot()
            owner = camera._thread
            report["initial"] = {
                "camera": initial["camera"],
                "frame": before.metadata(),
                "providers": initial["perception"]["providers"],
                "resources": process(),
            }
            frames = []
            offline = []
            async with connect(
                base.replace("http://", "ws://") + "/ws", max_size=2**21
            ) as ws:

                async def receive():
                    async for raw in ws:
                        event = json.loads(raw)
                        if event["type"] == "frame":
                            frames.append(event["frame"])
                        elif event["type"] == "camera_state":
                            offline.append(event)
                        if len(frames) > 1000:
                            raise RuntimeError(
                                "Diagnostic frame history exceeded budget"
                            )

                receiver = asyncio.create_task(receive())
                try:

                    async def saw_frame():
                        return bool(frames)

                    await wait(saw_frame)
                    await driver_write("unbind")
                    detached = True

                    async def unavailable():
                        value = await state()
                        return (
                            value
                            if not value["camera_ready"]
                            and not value["perception"]["ready"]
                            and offline
                            else None
                        )

                    lost = await wait(unavailable, 15)
                    report["disconnected"] = {
                        "camera": lost["camera"],
                        "perception_ready": lost["perception"]["ready"],
                        "face_detected": lost["face"]["face_detected"],
                        "old_frame_invalidated": before.invalidated.is_set(),
                        "preview_events": offline[-1:],
                    }
                    assert (
                        before.invalidated.is_set()
                        and camera.snapshot() is None
                        and camera.buffer_bytes == 0
                    )
                    assert not lost["face"]["face_detected"]
                    rejected = await client.post(
                        "/api/chat", json={"text": "当前画面是什么", "use_image": True}
                    )
                    assert rejected.status_code == 400
                    report["offline_image_request_code"] = 400
                    key = (
                        (
                            await client.post(
                                "/api/chat", json={"text": "请只回答：文字通道正常。"}
                            )
                        )
                        .raise_for_status()
                        .json()["turn_id"]
                    )

                    async def done():
                        value = await state()
                        last = value.get("last_result")
                        return (
                            last
                            if not value["active_turn"]
                            and last
                            and last["turn_id"] == key
                            else None
                        )

                    text_result = await wait(done)
                    assert text_result["status"] == "complete"
                    report["offline_text_turn"] = {
                        "status": text_result["status"],
                        "resources_released": text_result["resources_released"],
                    }
                    await driver_write("bind")
                    detached = False
                    restored = await wait(scene, 30)
                    after = camera.snapshot()
                    assert (
                        after.source_epoch != before.source_epoch
                        and camera._thread is owner
                        and camera.status()["reconnects"] >= 1
                    )
                    assert (
                        not after.invalidated.is_set()
                        and restored["camera"]["error"] is None
                    )
                    assert (
                        restored["perception"]["frame"]["source_epoch"]
                        == after.source_epoch
                    )
                    report["restored"] = {
                        "camera": restored["camera"],
                        "frame": after.metadata(),
                        "providers": restored["perception"]["providers"],
                        "same_capture_thread": True,
                        "resources": process(),
                    }
                    key = (
                        (
                            await client.post(
                                "/api/chat",
                                json={
                                    "text": "只回答当前画面里是否有人，不识别身份，不转录文字。",
                                    "use_image": True,
                                },
                            )
                        )
                        .raise_for_status()
                        .json()["turn_id"]
                    )
                    visual = await wait(done, 90)
                    assert (
                        visual["status"] == "complete"
                        and visual["resources_released"]
                        and visual["frame"]["source_epoch"] == after.source_epoch
                    )
                    report["restored_visual_turn"] = {
                        "status": visual["status"],
                        "frame": visual["frame"],
                        "metadata": visual["metadata"],
                        "resources_released": visual["resources_released"],
                    }

                    async def saw_new_frame():
                        return any(
                            f["source_epoch"] == after.source_epoch for f in frames
                        )

                    await wait(saw_new_frame)
                    new_index = next(
                        i
                        for i, f in enumerate(frames)
                        if f["source_epoch"] == after.source_epoch
                    )
                    assert all(
                        f["source_epoch"] != before.source_epoch
                        for f in frames[new_index:]
                    )
                    report["web_frames"] = {
                        "received": len(frames),
                        "new_epoch_seen": True,
                        "old_epoch_after_new": False,
                    }
                    report["status"] = "PASSED_WITH_DECLARED_SCOPE"
                finally:
                    receiver.cancel()
                    await asyncio.gather(receiver, return_exceptions=True)
    except BaseException as exc:
        report.update(status="FAILED", error=repr(exc))
        raise
    finally:
        try:
            if detached:
                await driver_write("bind")
            if server:
                server.should_exit = True
            if running:
                await asyncio.wait_for(running, 30)
            if listener:
                listener.close()
            report["app_shutdown"] = {
                "camera": camera.status() if camera else None,
                "resources": process(),
            }
            if report["status"] == "PASSED_WITH_DECLARED_SCOPE":
                # Camera-only lifecycle qualification; full app/model/audio
                # startup repetition remains a separate R4 acceptance item.
                cycle_camera = factory(device=str(device), capture_fps=10)
                report["camera_lifecycle"] = {
                    "scope": "One warmup plus 20 real camera start/read/close cycles",
                    "cycles": [],
                }
                epochs = set()
                try:
                    for index in range(21):
                        assert await asyncio.to_thread(cycle_camera.start), (
                            cycle_camera.last_error
                        )
                        frame = cycle_camera.snapshot()
                        assert frame is not None
                        assert frame.source_epoch not in epochs
                        epochs.add(frame.source_epoch)
                        await asyncio.to_thread(cycle_camera.close)
                        assert (
                            frame.invalidated.is_set()
                            and not cycle_camera.status()["capture_worker_alive"]
                        )
                        assert cycle_camera.buffer_bytes == 0
                        resources = process()
                        if index == 0:
                            report["camera_lifecycle"]["baseline_after_warmup"] = (
                                resources
                            )
                        else:
                            assert (
                                resources["fds"]
                                == report["camera_lifecycle"]["baseline_after_warmup"][
                                    "fds"
                                ]
                            )
                            assert (
                                resources["threads"]
                                == report["camera_lifecycle"]["baseline_after_warmup"][
                                    "threads"
                                ]
                            )
                            report["camera_lifecycle"]["cycles"].append(
                                {
                                    "index": index,
                                    "frame": frame.metadata(),
                                    "after_close": resources,
                                }
                            )
                    report["camera_lifecycle"]["status"] = "PASSED"
                finally:
                    await asyncio.to_thread(cycle_camera.close)
        except BaseException as exc:
            report.update(status="FAILED_SHUTDOWN_OR_LIFECYCLE", error=repr(exc))
            raise
        finally:
            if active and legacy_stopped:
                await command(
                    "systemctl", "--user", "start", "mmassistant-camera.service"
                )
                restore = args.output.with_name(
                    args.output.stem + "-legacy-restore.json"
                )
                try:
                    await command(
                        "bash",
                        "-c",
                        'source /opt/ros/jazzy/setup.bash\nexport ROS_DOMAIN_ID=42 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST\nexec /usr/bin/python3 "$1" "$2"',
                        "camera-restore",
                        str(Path(__file__).with_name("probe_legacy_camera.py")),
                        str(restore),
                        timeout=50,
                    )
                    report["legacy_restore"] = json.loads(restore.read_text())
                except Exception as exc:
                    report["legacy_restore"] = {"fresh": False, "error": repr(exc)}
                    report["status"] = "FAILED_RESTORATION"
            save()
    if report["status"] != "PASSED_WITH_DECLARED_SCOPE":
        raise RuntimeError(report["status"])
    print(
        json.dumps(
            {
                "status": report["status"],
                "restored": report["restored"]["camera"],
                "legacy_fresh": report["legacy_restore"]["fresh"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detach-configured-camera", action="store_true", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(probe(parser.parse_args()))
