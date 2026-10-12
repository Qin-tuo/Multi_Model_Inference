"""Bounded RGB-D payload ownership for the ROS/Pipecat adapter."""
from collections import Counter, deque
from dataclasses import dataclass
import threading
import time
import uuid


class VisionUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class RGBDFrame:
    frame_id: str
    epoch: str
    source_time_ns: int
    capture_time_ns: int
    arrival_time_ns: int
    width: int
    height: int
    rgb_step: int
    rgb_encoding: str
    rgb: bytes
    depth_step: int
    depth: bytes
    optical_frame: str

    @property
    def payload_bytes(self): return len(self.rgb) + len(self.depth)

    def packed_rgb(self):
        if self.rgb_encoding == "rgb8" and self.rgb_step == self.width * 3:
            return self.rgb
        from PIL import Image
        mode = "RGB" if self.rgb_encoding == "rgb8" else "BGR"
        return Image.frombytes("RGB", (self.width, self.height), self.rgb, "raw", mode, self.rgb_step, 1).tobytes()


class FrameLease:
    def __init__(self, store, frame):
        self.store, self.frame, self.closed = store, frame, False

    def validate(self):
        if self.closed: raise VisionUnavailable("FRAME_RELEASED")
        with self.store.lock:
            if self.frame.epoch != self.store.epoch: raise VisionUnavailable("CAMERA_EPOCH_CHANGED")
            if self.store.clock() - self.frame.capture_time_ns > self.store.retention_ns:
                raise VisionUnavailable("FRAME_EXPIRED")
        return self.frame

    def close(self):
        if not self.closed:
            self.store.release(self.frame.frame_id)
            self.closed = True

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


class VisionFrameStore:
    """A byte-limited data cache, not a task scheduler or conversation runtime.

    Retired frames remain in the byte accounting while a consumer holds a
    lease. The first profile permits one image request in flight.
    """
    def __init__(self, *, max_bytes=118 * 1024 * 1024, retention_seconds=2,
                 stale_seconds=.5, max_frame_bytes=4 * 1024 * 1024, max_leases=1,
                 clock=time.monotonic_ns):
        if min(max_bytes, max_frame_bytes, max_leases, retention_seconds, stale_seconds) <= 0:
            raise ValueError("Vision cache bounds must be positive")
        self.max_bytes, self.max_frame_bytes, self.max_leases = max_bytes, max_frame_bytes, max_leases
        self.retention_ns, self.stale_ns = int(retention_seconds * 1e9), int(stale_seconds * 1e9)
        self.clock, self.lock = clock, threading.RLock()
        self.epoch = uuid.uuid4().hex
        self.queue, self.resident, self.pins = deque(), {}, {}
        self.bytes = self.peak_bytes = 0
        self.last_arrival = self.last_source = None
        self.counts = Counter()

    def reset(self, reason):
        with self.lock:
            while self.queue: self._retire()
            self.epoch = uuid.uuid4().hex
            self.last_arrival = self.last_source = None
            self.counts["reset:" + reason] += 1

    def _retire(self):
        key = self.queue.popleft()
        if key not in self.pins:
            self.bytes -= self.resident.pop(key).payload_bytes

    def _prune(self, now):
        while self.queue and now - self.resident[self.queue[0]].capture_time_ns > self.retention_ns:
            self._retire()
            self.counts["expired"] += 1

    def add(self, *, source_time_ns, capture_time_ns, arrival_time_ns, width, height,
            rgb_step, rgb_encoding, rgb, depth_step, depth, optical_frame):
        size = len(rgb) + len(depth)
        valid = (0 < width <= 1920 and 0 < height <= 1080 and source_time_ns > 0
                 and rgb_encoding in {"rgb8", "bgr8"} and rgb_step >= width * 3 and depth_step >= width * 2
                 and len(rgb) == rgb_step * height and len(depth) == depth_step * height
                 and 0 < size <= self.max_frame_bytes and bool(optical_frame))
        with self.lock:
            if not valid:
                self.counts["invalid_payload"] += 1
                return False
            if capture_time_ns > arrival_time_ns or arrival_time_ns - capture_time_ns > self.retention_ns:
                self.counts["invalid_capture_clock_or_stale"] += 1
                return False
            if self.last_source is not None and source_time_ns <= self.last_source:
                if self.last_source - source_time_ns > self.stale_ns:
                    self.reset("source_clock_reset")
                else:
                    self.counts["out_of_order_or_duplicate"] += 1
                    return False
            if self.last_arrival is not None and arrival_time_ns - self.last_arrival > self.stale_ns:
                self.reset("capture_gap")
            self._prune(arrival_time_ns)
            while self.queue and self.bytes + size > self.max_bytes:
                self._retire()
                self.counts["capacity_eviction"] += 1
            if self.bytes + size > self.max_bytes:
                self.counts["capacity_rejection"] += 1
                return False
            key = f"{self.epoch}:{source_time_ns}"
            frame = RGBDFrame(key, self.epoch, source_time_ns, capture_time_ns, arrival_time_ns,
                width, height, rgb_step, rgb_encoding, bytes(rgb), depth_step, bytes(depth), optical_frame)
            self.resident[key] = frame
            self.queue.append(key)
            self.bytes += size
            self.peak_bytes = max(self.peak_bytes, self.bytes)
            self.last_source, self.last_arrival = source_time_ns, arrival_time_ns
            self.counts["accepted"] += 1
            return True

    def acquire(self, question_time_ns=None, *, max_skew_ms=150):
        now = self.clock()
        target = now if question_time_ns is None else question_time_ns
        if max_skew_ms <= 0 or target > now: raise VisionUnavailable("INVALID_QUESTION_TIME")
        with self.lock:
            self._prune(now)
            if not self.queue: raise VisionUnavailable("NO_FRESH_FRAME")
            if self.last_arrival is None or now - self.last_arrival > self.stale_ns:
                raise VisionUnavailable("CAMERA_STREAM_STALE")
            if sum(self.pins.values()) >= self.max_leases: raise VisionUnavailable("IMAGE_REQUEST_BUSY")
            frame = min((self.resident[key] for key in self.queue), key=lambda value: abs(value.capture_time_ns - target))
            if abs(frame.capture_time_ns - target) > max_skew_ms * 1e6:
                raise VisionUnavailable("NO_FRAME_NEAR_QUESTION")
            self.pins[frame.frame_id] = self.pins.get(frame.frame_id, 0) + 1
            return FrameLease(self, frame)

    def release(self, key):
        with self.lock:
            if key not in self.pins: return
            self.pins[key] -= 1
            if self.pins[key] == 0:
                del self.pins[key]
                if key not in self.queue:
                    self.bytes -= self.resident.pop(key).payload_bytes

    def status(self):
        now = self.clock()
        with self.lock:
            self._prune(now)
            fresh = bool(self.queue) and self.last_arrival is not None and now - self.last_arrival <= self.stale_ns
            return {"state": "READY" if fresh else "STALE", "epoch": self.epoch,
                    "frames": len(self.queue), "resident_payload_bytes": self.bytes, "peak_payload_bytes": self.peak_bytes,
                    "max_payload_bytes": self.max_bytes, "leased_frames": sum(self.pins.values()), "counts": dict(self.counts),
                    "last_arrival_age_ms": (now - self.last_arrival) / 1e6 if self.last_arrival is not None else None}
