# SPDX-License-Identifier: Apache-2.0
"""One bounded request worker around the upstream VLM streaming sequence.

R1 owns text/image turns. Audio, history, tools and full cancellation are
integrated in later stages into this same owner.
"""
import queue
import threading
import time
import uuid


class TurnController:
    def __init__(self, config, llm, broadcaster, camera=None):
        self.config, self.llm, self.broadcaster, self.camera = config, llm, broadcaster, camera
        self._queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._active = None
        self._last = None

    def start(self):
        if self._thread and self._thread.is_alive(): return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="conversation", daemon=True)
        self._thread.start()

    def submit(self, text, use_image=False):
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError("Text must contain 1–4000 characters")
        if type(use_image) is not bool:
            raise ValueError("use_image must be boolean")
        with self._lock:
            if self._stop.is_set() or not self._thread or not self._thread.is_alive():
                raise RuntimeError("Conversation worker is not ready")
            if self._active is not None:
                raise RuntimeError("The current request is still running")
            if use_image and (self.camera is None or not self.camera.health_check()):
                raise ValueError("No fresh camera image is available")
            request = {"turn_id": uuid.uuid4().hex, "text": text.strip(), "use_image": use_image}
            self._active = request
            self._queue.put_nowait(request)
            return {"accepted": True, "turn_id": request["turn_id"]}

    def state(self):
        with self._lock:
            return {"ready": bool(self._thread and self._thread.is_alive() and not self._stop.is_set()),
                    "active_turn": self._active["turn_id"] if self._active else None,
                    "last_result": self._last, "pending": self._queue.qsize()}

    def _run(self):
        while not self._stop.is_set():
            try: request = self._queue.get(timeout=0.1)
            except queue.Empty: continue
            result = {"turn_id": request["turn_id"], "text": "", "metadata": {}, "mode": "camera_text" if request["use_image"] else "text"}
            def send(kind, **values):
                self.broadcaster.send({"type": kind, "turn_id": request["turn_id"], **values})
            start = time.monotonic()
            first = None
            try:
                if self._stop.is_set(): raise InterruptedError("Application stopping")
                images = None
                if request["use_image"]:
                    frame = self.camera.snapshot()
                    if frame is None: raise ValueError("Camera image expired before processing")
                    result["frame"] = frame.metadata()
                    images = [self.camera._base64_jpeg(frame.jpeg)]
                send("transcript", text=request["text"])
                send("status", stage="thinking")
                stream = self.llm.generate_stream(request["text"], images_b64=images,
                    system_prompt=self.config.vision.system_prompt if images else None)
                try:
                    for text, metadata in stream:
                        if self._stop.is_set(): raise InterruptedError("Application stopping")
                        result["metadata"].update(metadata)
                        if text:
                            first = first or time.monotonic()
                            result["text"] += text
                            send("token", text=text)
                finally:
                    stream.close()
                if not result["metadata"].get("done") or not result["text"].strip():
                    raise RuntimeError("Model did not produce a complete text answer")
                result["status"] = "complete"
            except Exception as exc:
                result.update(status="failed", error=str(exc))
                send("error", error=str(exc))
            finally:
                result.update(elapsed_seconds=time.monotonic() - start,
                              first_content_seconds=first - start if first else None)
                # The HTTP stream has closed before the next request is admitted.
                # Provider-release and cross-modality cancellation are R3 checks.
                with self._lock:
                    self._last = result
                    self._active = None
                self._queue.task_done()
                send("done", text=result["text"], status=result["status"],
                     tokens=result["metadata"].get("eval_count"),
                     ttft=result["first_content_seconds"], vlm_time=result["elapsed_seconds"])
                send("status", stage="idle")

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.config.llm.timeout + 5)
            if self._thread.is_alive():
                raise RuntimeError("Conversation worker has not released")
            self._thread = None
