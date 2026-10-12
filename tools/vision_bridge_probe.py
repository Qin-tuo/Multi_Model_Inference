"""Live D435i → ROS → bounded RGB-D cache → native Pipecat image frames.

No raw camera image is written to disk, and no VLM/detector result is asserted.
The optional owned camera launch is stopped to exercise stale-frame refusal.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipecat.frames.frames import EndFrame, UserImageRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.workers.runner import WorkerRunner
from app.adapters.vision import RosVisionBridge
from app.vision_frames import VisionUnavailable
from app.vision_snapshot import VisionSnapshotProcessor
from tools.project import CACHE, stamp, write_json


class Capture(FrameProcessor):
    def __init__(self, provider):
        super().__init__(); self.provider = provider; self.results = []
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, UserImageRawFrame):
            now = time.monotonic_ns()
            self.results.append({"request_id": frame.metadata["request_id"], "frame_id": frame.metadata["frame_id"],
                "capture_to_consumer_ms": (now - frame.metadata["capture_time_ns"]) / 1e6,
                "question_frame_skew_ms": abs(frame.metadata["question_time_ns"] - frame.metadata["capture_time_ns"]) / 1e6,
                "size": list(frame.size), "bytes": len(frame.image), "format": frame.format})
            self.provider.release_request(frame.metadata["request_id"])
        await self.push_frame(frame, direction)


async def probe(args):
    bridge = RosVisionBridge.configured().start()
    provider = VisionSnapshotProcessor(bridge.frames)
    capture = Capture(provider)
    worker = PipelineWorker(Pipeline([provider, capture]), enable_rtvi=False, idle_timeout_secs=None)
    errors = []
    @worker.event_handler("on_pipeline_error")
    async def error(_worker, frame): errors.append(frame.error)
    runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    camera = None
    output = None
    old_lease = None
    try:
        if args.launch_camera:
            output = (Path.home() / "AI_Stack/logs/vision-bridge-camera.log").open("w")
            camera = subprocess.Popen(["python3", str(ROOT / "tools/project.py"), "serve", "--component", "camera"],
                cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 25
        while bridge.status()["state"] != "READY":
            if time.monotonic() > deadline: raise RuntimeError("No valid aligned RGB-D frame arrived")
            if camera is not None and camera.poll() is not None: raise RuntimeError("Owned camera launch exited before capture")
            await asyncio.sleep(.1)
        # Hold the stream long enough to populate and prune the two-second ring.
        await asyncio.sleep(2.2)
        count = 20
        for index in range(count):
            await worker.queue_frame(provider.request("live frame bridge probe", user_id="owner",
                turn_id=f"probe-{index}", question_time_ns=time.monotonic_ns()))
            await asyncio.sleep(.1)
        await asyncio.sleep(.2)
        active = bridge.status()
        stale_refused = None
        restart_verified = None
        stale_state = None
        if camera is not None:
            old_lease = bridge.frames.acquire()
            old_epoch = old_lease.frame.epoch
            os.killpg(camera.pid, signal.SIGINT)
            await asyncio.to_thread(camera.wait, 10)
            camera = None
            await asyncio.sleep(.7)
            try:
                lease = bridge.frames.acquire(); lease.close()
                stale_refused = False
            except VisionUnavailable as error:
                stale_refused = str(error) in {"CAMERA_STREAM_STALE", "NO_FRESH_FRAME"}
            stale_state = bridge.status()
            stale_refused = stale_refused and stale_state["state"] == "STALE"
            camera = subprocess.Popen(["python3", str(ROOT / "tools/project.py"), "serve", "--component", "camera"],
                cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 25
            while bridge.status()["state"] != "READY" or bridge.frames.epoch == old_epoch:
                if time.monotonic() > deadline: raise RuntimeError("Camera stream did not recover with a new epoch")
                await asyncio.sleep(.1)
            try:
                old_lease.validate()
                restart_verified = False
            except VisionUnavailable as error:
                restart_verified = "EPOCH" in str(error)
            finally:
                old_lease.close()
                old_lease = None
            await worker.queue_frame(provider.request("after driver restart", user_id="owner",
                turn_id="restart", question_time_ns=time.monotonic_ns()))
            await asyncio.sleep(.2)
            count += 1
        stopped = bridge.status()
        passed = (len(capture.results) == count and not errors and active["leased_frames"] == 0
            and active["peak_payload_bytes"] <= active["max_payload_bytes"]
            and all(row["bytes"] == 640 * 480 * 3 and row["question_frame_skew_ms"] <= 150 for row in capture.results)
            and (not args.launch_camera or (stale_refused and restart_verified)))
        report = {"created_at": stamp(), "status": "PASS" if passed else "FAIL",
            "scope": "live RGB-D transport, byte ownership, source-time selection and Pipecat image-request frames; not visual inference, image QA, or JT-03 acceptance",
            "raw_images_saved": False, "requests": count, "responses": len(capture.results), "errors": errors,
            "active": active, "final_stream": stopped, "stale_frame_refused": stale_refused,
            "while_camera_stopped": stale_state, "driver_restart_epoch_verified": restart_verified, "samples": capture.results}
        write_json(ROOT / "benchmarks/results/vision-bridge-smoke.json", report)
        print(json.dumps({key: value for key, value in report.items() if key != "samples"}, ensure_ascii=False, indent=2))
        return 0 if passed else 1
    finally:
        if old_lease is not None: old_lease.close()
        if camera is not None and camera.poll() is None:
            os.killpg(camera.pid, signal.SIGINT)
            try: await asyncio.to_thread(camera.wait, 10)
            except subprocess.TimeoutExpired:
                os.killpg(camera.pid, signal.SIGKILL)
                await asyncio.to_thread(camera.wait)
        if output: output.close()
        await worker.queue_frame(EndFrame())
        await running
        bridge.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-camera", action="store_true", help="Launch and stop an owned camera process for this probe")
    return asyncio.run(probe(parser.parse_args()))


if __name__ == "__main__": raise SystemExit(main())
