#!/usr/bin/env python3
"""E1-only input/observation adapters around the unchanged production app.

Only the camera driver is substituted with a fixed public image. All functional
modules, tool policies, capacities, rendering transport and cancellation remain
enabled. Hooks forward original arguments/returns and write measurement records.
"""

import argparse
from contextvars import ContextVar
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "01_src"))
from benchmarks.baseline_support import Events  # noqa: E402
from benchmarks.http_observer import Observer  # noqa: E402
from benchmarks.vision_stream import ReplayCapture  # noqa: E402
from deploy import runtime  # noqa: E402
from project_paths import load_artifact_manifest

TURN = ContextVar("benchmark_turn", default=None)


def instrument_app(cfg, out):
    import cv2
    import app.application as entry
    import app.turn_control as turns
    import app.actions.store as store
    from app.camera import Camera
    from app.llm import LLM
    from app.stt import STT
    from app.tts import LocalTTS
    from app.speech.output import SpeechOutput
    from deploy.resource_guard import hold_model_lease

    lease = hold_model_lease(runtime.paths(cfg)["stack"] / "cache")
    events = Events(out / "server-events.jsonl")
    rounds = {}

    def emit(kind, **row):
        row.setdefault("turn_id", TURN.get())
        events.write(kind=kind, **row)

    class FixedCamera(Camera):
        def _open(self):
            fixture = load_artifact_manifest(
                ROOT / "benchmarks/manifests/b0-e1-input-v1.json"
            )
            if runtime.sha256(fixture["image_path"]) != fixture["sha256"]:
                raise RuntimeError("E1 camera fixture changed")
            image = cv2.imread(fixture["image_path"])
            image.flags.writeable = False
            self._cap = ReplayCapture(
                [("e1-public-street.png", image)],
                self._stop,
                events,
                lambda: "application",
                self.capture_fps,
            )
            return True

        @staticmethod
        def _base64_jpeg(jpeg):
            start = time.monotonic()
            result = Camera._base64_jpeg(jpeg)
            if TURN.get():
                emit(
                    "image_base64",
                    started=start,
                    ended=time.monotonic(),
                    bytes=len(jpeg),
                )
            return result

    class MeasuredBroadcaster(entry.Broadcaster):
        benchmark_control = None

        def send(self, message, *args, **kwargs):
            now = time.monotonic()
            value = super().send(message, *args, **kwargs)
            if message.get("turn_id") or message.get("type") == "gesture":
                emit(
                    "server_event",
                    emitted_monotonic=now,
                    message=deepcopy(message),
                    turn_id=message.get("turn_id"),
                )
            if (
                message.get("type") == "status"
                and message.get("stage") == "idle"
                and self.benchmark_control is not None
            ):
                emit(
                    "final_result",
                    turn_id=message["turn_id"],
                    result=self.benchmark_control.state()["last_result"],
                )
            return value

    class MeasuredControl(entry.TurnController):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.broadcaster.benchmark_control = self
            original = self._queue.get

            def take(*a, **kw):
                request = original(*a, **kw)
                tick = time.monotonic()
                TURN.set(request["turn_id"])
                emit(
                    "dequeued",
                    admitted_monotonic=request["admitted_monotonic"],
                    dequeued_monotonic=tick,
                    input_kind=request["input_kind"],
                    request_id=request["request_id"],
                    use_image=request["use_image"],
                )
                return request

            self._queue.get = take

    class MeasuredLLM(LLM):
        def generate_stream(self, *args, **kwargs):
            key = TURN.get()
            rounds[key] = rounds.get(key, 0) + 1
            number = rounds[key]
            observer = Observer()
            first = None
            metadata = {}
            error = None
            start = time.monotonic()
            try:
                with observer.capture():
                    for text, meta in super().generate_stream(*args, **kwargs):
                        metadata.update(meta)
                        if text:
                            first = first or time.monotonic()
                        yield text, meta
            except BaseException as exc:
                error = repr(exc)
                raise
            finally:
                end = time.monotonic()
                # Deferred parsing/file export is outside the native-call clock;
                # its small overhead remains in the complete application result.
                try:
                    http = observer.export(out, f"{key}-round-{number}")
                except Exception as exc:
                    http = {"capture_error": repr(exc)}
                emit(
                    "llm_call",
                    round=number,
                    started=start,
                    ended=end,
                    first_adapter_content=first,
                    metadata=metadata,
                    error=error,
                    http=http,
                )

    class MeasuredSTT(STT):
        def transcribe(self, *args, **kwargs):
            start = time.monotonic()
            try:
                result = super().transcribe(*args, **kwargs)
                emit("asr_call", started=start, ended=time.monotonic(), result=result)
                return result
            except BaseException as exc:
                emit("asr_call", started=start, ended=time.monotonic(), error=repr(exc))
                raise

    class MeasuredTTS(LocalTTS):
        def synthesize(self, *args, **kwargs):
            start = time.monotonic()
            try:
                result = super().synthesize(*args, **kwargs)
                emit(
                    "tts_call",
                    started=start,
                    ended=time.monotonic(),
                    result={k: v for k, v in result.items() if k != "audio"},
                    first_pcm_estimated_monotonic=start + result["first_pcm_seconds"]
                    if result.get("first_pcm_seconds") is not None
                    else None,
                )
                return result
            except BaseException as exc:
                emit("tts_call", started=start, ended=time.monotonic(), error=repr(exc))
                raise

    class MeasuredOutput(SpeechOutput):
        def _run(self):
            token = TURN.set(self.turn_id)
            try:
                return super()._run()
            finally:
                TURN.reset(token)

        def publish(self):
            start = time.monotonic()
            result = super().publish()
            emit(
                "audio_published",
                turn_id=self.turn_id,
                started=start,
                ended=time.monotonic(),
                receipt=result,
            )
            return result

    class MeasuredStore(store.ActionStore):
        def execute(self, *args, **kwargs):
            start = time.monotonic()
            result = super().execute(*args, **kwargs)
            emit(
                "tool_completed",
                started=start,
                ended=time.monotonic(),
                request_id=args[0],
                name=args[2],
                arguments=args[3],
                result=result,
            )
            return result

    segment = turns.segment_pcm

    def measured_segment(*args, **kwargs):
        start = time.monotonic()
        count = 0
        try:
            for value in segment(*args, **kwargs):
                count += 1
                yield value
        finally:
            emit("file_vad", started=start, ended=time.monotonic(), segments=count)

    entry.Camera, entry.Broadcaster, entry.TurnController = (
        FixedCamera,
        MeasuredBroadcaster,
        MeasuredControl,
    )
    entry.LLM, entry.STT, entry.create_tts = MeasuredLLM, MeasuredSTT, MeasuredTTS
    turns.SpeechOutput, turns.segment_pcm = MeasuredOutput, measured_segment
    store.ActionStore = MeasuredStore
    return entry.build_app(cfg), events, lease


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = runtime.config(args.config)
    app, events, _lease = instrument_app(cfg, args.output)
    import uvicorn

    try:
        uvicorn.run(
            app,
            host=cfg.web.host,
            port=cfg.web.port,
            log_level="warning",
            ws_max_size=cfg.web.ws_message_bytes,
            ws_max_queue=4,
            limit_concurrency=16,
            timeout_graceful_shutdown=10,
        )
    finally:
        events.close()


if __name__ == "__main__":
    main()
