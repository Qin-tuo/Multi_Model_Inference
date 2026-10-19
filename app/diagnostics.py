# SPDX-License-Identifier: Apache-2.0
"""Bounded board-side observations, independent of inference decisions.

Durations use the board's monotonic clock. The live window is an operational
sample, not the frozen benchmark and not a CUDA kernel profile.
"""
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import threading
import time
import uuid

import psutil


class TurnTrace:
    """One turn, at most 64 completed spans; parallel TTS spans may overlap LLM."""

    def __init__(self, turn_id, admitted, clock=time.monotonic):
        self.clock, self.origin, self.turn_id = clock, admitted, turn_id
        self._lock = threading.Lock()
        self._spans = deque(maxlen=64)
        self._active = {}
        self._totals = {}
        self._marks = {}
        self._sequence = self._dropped = 0
        self._status = "running"
        self._end = None

    def mark(self, name):
        with self._lock:
            self._marks.setdefault(name, max(0, (self.clock() - self.origin) * 1000))

    @contextmanager
    def span(self, name):
        start = self.clock()
        with self._lock:
            self._sequence += 1
            identity = self._sequence
            self._active[identity] = {"name": name, "start_ms": (start - self.origin) * 1000}
        outcome = "complete"
        try:
            yield
        except BaseException:
            outcome = "interrupted"
            raise
        finally:
            end = self.clock()
            with self._lock:
                span = self._active.pop(identity)
                span.update(duration_ms=max(0, (end - start) * 1000), outcome=outcome)
                if len(self._spans) == self._spans.maxlen:
                    self._dropped += 1
                self._spans.append(span)
                total = self._totals.setdefault(name, {"count": 0, "duration_ms": 0.0})
                total["count"] += 1
                total["duration_ms"] += span["duration_ms"]

    def finish(self, status):
        with self._lock:
            self._status, self._end = status, self.clock()

    def snapshot(self):
        with self._lock:
            elapsed = max(0, ((self._end or self.clock()) - self.origin) * 1000)
            active = [{**s, "duration_ms": max(0, elapsed - s["start_ms"]),
                       "outcome": "running"} for s in self._active.values()]
            return {"turn_id": self.turn_id, "status": self._status,
                    "elapsed_ms": elapsed, "marks_ms": dict(self._marks),
                    "stages": deepcopy(self._totals),
                    "spans": [dict(s) for s in self._spans] + active,
                    "omitted_spans": self._dropped}


def _number(path, divisor=1):
    try:
        return int(path.read_text().strip()) / divisor
    except (OSError, ValueError, AttributeError):
        return None


class BoardDiagnostics:
    """Sampled by the existing stats worker, shared by all viewers."""

    def __init__(self, config, sysfs=Path("/sys")):
        self.config = config
        self.session = uuid.uuid4().hex
        self._lock = threading.Lock()
        self._latest = {"ready": False, "source": "board", "session": self.session}
        self._previous = None
        self._sequence = 0
        self.temperatures = {}
        for path in sysfs.glob("class/thermal/thermal_zone*/type"):
            try:
                name = path.read_text().strip()
                if name in {"cpu-thermal", "gpu-thermal", "tj-thermal"}:
                    self.temperatures[name] = path.with_name("temp")
            except OSError:
                pass
        self.gpu_clock = next(sysfs.glob("class/devfreq/*gpu*/cur_freq"), None)
        self.emc_clock = sysfs / "class/devfreq/bwmgr/cur_freq"
        self.power = None
        for path in sysfs.glob("class/hwmon/hwmon*/in*_label"):
            try:
                if path.read_text().strip() == "VDD_IN":
                    channel = path.name.split("_")[0][2:]
                    self.power = (path.with_name(f"in{channel}_input"),
                                  path.with_name(f"curr{channel}_input"))
                    break
            except OSError:
                pass

    def sample(self, current, stats, now=None):
        started = time.monotonic()
        now = started if now is None else now
        camera, scene = current.get("camera", {}), current.get("perception", {})
        epoch = camera.get("source_epoch")
        captured, processed = camera.get("frame_id"), scene.get("processed")
        capture_fps = perception_fps = interval = None
        if self._previous:
            old_time, old_epoch, old_capture, old_processed = self._previous
            interval = now - old_time
            if interval > 0 and old_epoch == epoch and epoch is not None:
                if captured is not None and old_capture is not None and captured >= old_capture:
                    capture_fps = (captured - old_capture) / interval
                if processed is not None and old_processed is not None and processed >= old_processed:
                    perception_fps = (processed - old_processed) / interval
        self._previous = now, epoch, captured, processed
        frame = scene.get("frame") or {}
        received = frame.get("received_monotonic")
        valid = bool(scene.get("ready") and received is not None and
                     frame.get("source_epoch") == epoch and
                     0 <= now - received <= self.config.vision.stale_after)
        memory, swap = psutil.virtual_memory(), psutil.swap_memory()
        power_w = None
        if self.power:
            voltage, current_ma = (_number(p) for p in self.power)
            if voltage is not None and current_ma is not None:
                power_w = voltage * current_ma / 1_000_000  # mV * mA -> W
        last = current.get("last_result") or {}
        chunks = last.get("speech", {}).get("chunks", [])
        first_pcm = chunks[0].get("first_pcm_seconds") if chunks else None
        self._sequence += 1
        value = {
            "schema_version": 1, "ready": True, "source": "board",
            "session": self.session, "sequence": self._sequence,
            "sampled_monotonic": now, "sampled_unix": time.time(),
            "interval_seconds": interval,
            "resources": {"cpu_percent": stats.cpu_percent, "gpu_percent": stats.gpu_percent,
                "ram_used_gib": (memory.total - memory.available) / 2**30,
                "ram_total_gib": memory.total / 2**30, "ram_available_gib": memory.available / 2**30,
                "swap_used_gib": swap.used / 2**30,
                "temperatures_c": {k: _number(p, 1000) for k, p in self.temperatures.items()},
                "gpu_mhz": _number(self.gpu_clock, 1_000_000),
                "emc_mhz": _number(self.emc_clock, 1_000_000), "module_input_w": power_w},
            "vision": {"camera_ready": camera.get("ready", False), "ready": valid,
                "source_epoch": epoch, "frame_id": frame.get("frame_id") if valid else None,
                "capture_fps": capture_fps, "perception_fps": perception_fps,
                "capture_target_fps": self.config.vision.capture_fps,
                "perception_target_fps": self.config.perception.fps,
                "frame_age_ms": max(0, (now - received) * 1000) if valid else None,
                "processing_ms": scene.get("elapsed_ms") if valid else None,
                "stages_ms": scene.get("stages", {}) if valid else {},
                "body_stages_ms": scene.get("body_stages_ms", {}) if valid else {},
                "processed": processed, "skipped_capture_frames": scene.get("skipped_capture_frames"),
                "failures": scene.get("failures"), "reconnects": camera.get("reconnects"),
                "buffer_mib": camera.get("buffer_bytes", 0) / 2**20,
                "error": scene.get("error") or camera.get("error")},
            "turn": {"active_turn": current.get("active_turn"), "phase": current.get("phase"),
                "pending": current.get("pending"), "trace": current.get("turn_trace"),
                "last": {"turn_id": last.get("turn_id"), "status": last.get("status"),
                    "first_content_ms": (last["first_content_seconds"] * 1000
                                         if last.get("first_content_seconds") is not None else None),
                    "worker_total_ms": (last["elapsed_seconds"] * 1000
                                        if last.get("elapsed_seconds") is not None else None),
                    "first_tts_chunk_pcm_ms": first_pcm * 1000 if first_pcm is not None else None,
                    "audio_seconds": last.get("speech", {}).get("duration"),
                    "output_kind": last.get("speech", {}).get("receipt", {}).get("kind"),
                    "error": last.get("error")}},
            "transport": current.get("web", {}),
        }
        value["collection_ms"] = (time.monotonic() - started) * 1000
        with self._lock:
            self._latest = value
        return value

    def snapshot(self):
        with self._lock:
            value = deepcopy(self._latest)
        if value.get("ready"):
            value["cache_age_seconds"] = max(0, time.monotonic() - value["sampled_monotonic"])
        return value
