"""Benchmark-only HTTPX hooks; capture exact SSE bytes without consuming twice."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import httpx


def safe_payload(value):
    if isinstance(value, dict):
        return {k: safe_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_payload(v) for v in value]
    if isinstance(value, str) and value.startswith("data:image/"):
        return {
            "data_url_sha256": hashlib.sha256(value.encode()).hexdigest(),
            "chars": len(value),
        }
    return value


class Tee(httpx.SyncByteStream):
    def __init__(self, stream, row):
        self.stream, self.row = stream, row

    def __iter__(self):
        for chunk in self.stream:
            now = time.monotonic()
            self.row["received_bytes"] += len(chunk)
            if self.row["received_bytes"] <= 1024 * 1024:
                self.row["chunks"].append((now, chunk))
            else:
                self.row["capture_truncated"] = True
            yield chunk

    def close(self):
        try:
            self.stream.close()
        finally:
            self.row["closed_monotonic"] = time.monotonic()


class Observer:
    def __init__(self):
        self.rows = []
        self.requests = []

    def request(self, request):
        if request.method != "POST" or not request.url.path.startswith(
            "/v1/chat/completions"
        ):
            return
        row = {
            "path": request.url.path,
            "sent_monotonic": time.monotonic(),
            "request_sha256": hashlib.sha256(request.content).hexdigest(),
            "request": safe_payload(json.loads(request.content)),
            "chunks": [],
            "received_bytes": 0,
            "capture_truncated": False,
        }
        self.requests.append(request)
        self.rows.append(row)

    def response(self, response):
        index = next(
            (i for i, r in enumerate(self.requests) if r is response.request), None
        )
        if index is None:
            return
        row = self.rows[index]
        row.update(headers_monotonic=time.monotonic(), status_code=response.status_code)
        if response.is_stream_consumed:
            row["chunks"].append((time.monotonic(), response.content))
            row["received_bytes"] = len(response.content)
        else:
            response.stream = Tee(response.stream, row)

    @contextmanager
    def capture(self):
        original = httpx.Client

        def factory(*args, **kwargs):
            hooks = kwargs.pop("event_hooks", {})
            kwargs["event_hooks"] = {
                "request": [*hooks.get("request", []), self.request],
                "response": [*hooks.get("response", []), self.response],
            }
            return original(*args, **kwargs)

        # Only this isolated benchmark worker performs model calls. Resource
        # monitoring does not use HTTPX. RequestIO extensions remain untouched.
        with patch.object(httpx, "Client", factory):
            yield self

    def export(self, directory, prefix):
        """Run after the timed model call; preserve raw bytes and native timings."""
        rows = []
        for index, source in enumerate(self.rows):
            body = b"".join(data for _, data in source["chunks"])
            file = Path(directory) / f"{prefix}-{index}.response"
            file.write_bytes(body)
            row = {k: v for k, v in source.items() if k != "chunks"}
            row.update(
                response_file=file.name,
                response_sha256=hashlib.sha256(body).hexdigest(),
                chunks=[
                    {"received_monotonic": tick, "bytes": len(data)}
                    for tick, data in source["chunks"]
                ],
            )
            if source["path"].endswith("input_tokens"):
                row["payload"] = json.loads(body) if body else None
            else:
                pending = b""
                native_timings = None
                first_content = None
                for tick, data in source["chunks"]:
                    pending += data
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        if (
                            not line.startswith(b"data:")
                            or line[5:].strip() == b"[DONE]"
                        ):
                            continue
                        item = json.loads(line[5:].strip())
                        if item.get("timings"):
                            native_timings = item["timings"]
                        if any(
                            c.get("delta", {}).get("content")
                            for c in item.get("choices", [])
                        ):
                            first_content = first_content or tick
                row["first_content_received_monotonic"] = first_content
                row["native_timings"] = native_timings
            rows.append(row)
        return rows
