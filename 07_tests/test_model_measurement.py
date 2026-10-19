import hashlib
import json

import httpx

from benchmarks.http_observer import Observer
from benchmarks.text_metrics import character_errors


def test_observer_preserves_fragmented_stream_and_native_timings(tmp_path):
    body = (
        'data: {"choices":[{"delta":{"content":"你好"}}]}\n\n'
        'data: {"choices":[],"timings":{"predicted_n":2,"predicted_ms":10}}\n\n'
        "data: [DONE]\n\n"
    ).encode()

    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            for offset in range(0, len(body), 7):
                yield body[offset : offset + 7]

        def close(self):
            self.closed = True

    stream = Stream()
    received = []

    def transport(request):
        received.append(request.content)
        return httpx.Response(200, stream=stream)

    original = httpx.Client
    observer = Observer()
    payload = {
        "model": "fixed",
        "messages": [{"image": "data:image/jpeg;base64,AA=="}],
        "stream": True,
    }
    with (
        observer.capture(),
        httpx.Client(transport=httpx.MockTransport(transport)) as client,
    ):
        with client.stream(
            "POST", "http://127.0.0.1:8088/v1/chat/completions", json=payload
        ) as response:
            assert b"".join(response.iter_bytes()) == body
    assert httpx.Client is original and stream.closed
    assert json.loads(received[0]) == payload
    rows = observer.export(tmp_path, "one")
    assert rows[0]["native_timings"]["predicted_n"] == 2
    assert rows[0]["first_content_received_monotonic"] >= rows[0]["sent_monotonic"]
    assert rows[0]["response_sha256"] == hashlib.sha256(body).hexdigest()
    assert rows[0]["request"]["messages"][0]["image"]["chars"] > 0


def test_frozen_cer_does_not_hide_numeral_or_missing_transcript_errors():
    assert character_errors("你好，Ａ。", "你好a")["edit_distance"] == 0
    assert character_errors("六百四十", "640")["edit_distance"] > 0
    assert character_errors("测试", "")["cer"] == 1
    assert character_errors("", "多余")["cer"] is None
