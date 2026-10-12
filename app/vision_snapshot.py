"""Satisfy native Pipecat image requests from time-qualified ROS frame leases."""
import asyncio
import re
import time
import uuid

from pipecat.frames.frames import (CancelFrame, EndFrame, ErrorFrame, InterruptionFrame,
                                  UserImageRawFrame, UserImageRequestFrame)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from app.vision_frames import VisionUnavailable


def wants_visual_input(text):
    """Local intent hints; /look and the API visual flag provide explicit input."""
    return bool(re.search(r"摄像头|镜头|画面|图中|图片里|眼前|你(?:现在)?(?:看到|看见)|看(?:看)?(?:这个|这是什么|一下)|这(?:个|是什么).*(?:颜色|东西|物体)|面前有什么|^(?:请问[，, ]*)?[这那](?:个)?是什么[？?。！!]*$", text.strip()))


class VisionReplyReceiver(FrameProcessor):
    """Deliver the scoped native image response to its requesting session."""
    def __init__(self, receive):
        super().__init__()
        self.receive = receive

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, UserImageRawFrame) and frame.transport_source == "d435":
            await self.receive(frame)
            return
        await self.push_frame(frame, direction)


class VisionSnapshotProcessor(FrameProcessor):
    def __init__(self, frames):
        super().__init__()
        self.frames = frames
        self.generation = 0
        self.leases = {}
        self.packing = {}

    def request(self, text, *, user_id, turn_id, question_time_ns=None, append_to_context=False):
        frame = UserImageRequestFrame(user_id=user_id, text=text, video_source="d435", append_to_context=append_to_context)
        frame.metadata.update(metadata_schema_version=1, request_id=uuid.uuid4().hex, turn_id=turn_id,
            task_type="vision_snapshot", cancellation_generation=self.generation,
            question_time_ns=question_time_ns if question_time_ns is not None else time.monotonic_ns())
        return frame

    def release_request(self, request_id):
        lease = self.leases.pop(request_id, None)
        if lease: lease.close()

    def invalidate(self):
        self.generation += 1
        for key in list(self.leases):
            if key not in self.packing: self.release_request(key)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, (InterruptionFrame, CancelFrame, EndFrame)):
            self.invalidate()
        if isinstance(frame, UserImageRequestFrame) and frame.video_source == "d435":
            metadata = frame.metadata
            key = metadata.get("request_id")
            if key and metadata.get("cancellation_generation") != self.generation:
                with self.frames.lock: self.frames.counts["stale_requests_dropped"] += 1
                return
            try:
                if not key or not metadata.get("turn_id") or metadata.get("cancellation_generation") != self.generation:
                    raise VisionUnavailable("STALE_OR_UNSCOPED_IMAGE_REQUEST")
                lease = self.frames.acquire(metadata.get("question_time_ns"))
                self.leases[key] = lease
                # Requests are SystemFrames. Return immediately so image packing
                # cannot block the native interruption/control queue.
                self.packing[key] = self.create_task(self._respond(frame, lease))
            except (VisionUnavailable, ValueError, OSError) as error:
                if key: self.release_request(key)
                failure = ErrorFrame(error=str(error))
                failure.metadata.update(metadata)
                await self.push_error_frame(failure)
            return
        await self.push_frame(frame, direction)

    async def _respond(self, request, lease):
        metadata = request.metadata
        key = metadata["request_id"]
        work = asyncio.create_task(asyncio.to_thread(lease.frame.packed_rgb))
        delivered = False
        try:
            try: image = await asyncio.shield(work)
            except asyncio.CancelledError:
                await work  # Logical cancellation does not release a live buffer owner.
                raise
            if metadata["cancellation_generation"] != self.generation:
                return
            snapshot = lease.validate()
            response = UserImageRawFrame(image=image, size=(snapshot.width, snapshot.height), format="RGB",
                user_id=request.user_id, text=request.text, append_to_context=request.append_to_context, request=request)
            response.transport_source = "d435"
            response.metadata.update(metadata, frame_id=snapshot.frame_id, worker_epoch=snapshot.epoch,
                capture_time_ns=snapshot.capture_time_ns, arrival_time_ns=snapshot.arrival_time_ns,
                source_timestamp_ns=snapshot.source_time_ns,
                deadline_ns=snapshot.capture_time_ns + self.frames.retention_ns,
                payload_ref=f"vision-frame:{snapshot.frame_id}")
            await self.push_frame(response, FrameDirection.DOWNSTREAM)
            delivered = True
        except (VisionUnavailable, ValueError, OSError) as error:
            if metadata["cancellation_generation"] == self.generation:
                failure = ErrorFrame(error=str(error))
                failure.metadata.update(metadata)
                await self.push_error_frame(failure)
        finally:
            self.packing.pop(key, None)
            if not delivered or metadata["cancellation_generation"] != self.generation:
                self.release_request(key)

    async def cleanup(self):
        self.invalidate()
        if self.packing: await asyncio.gather(*list(self.packing.values()), return_exceptions=True)
        for key in list(self.leases): self.release_request(key)
        await super().cleanup()
