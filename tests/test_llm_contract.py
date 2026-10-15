"""Provider-contract regressions: reject silent fallback and incomplete streams."""
import json
import httpx
import pytest
from app.llm import LLM


def client_for(monkeypatch, handler):
    original = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: original(transport=httpx.MockTransport(handler), **kw))


def model():
    obj = LLM(model="assistant-vlm", max_tokens=128, context_tokens=2048,
              chat_template_kwargs={"enable_thinking": False})
    obj._loaded = True
    return obj


def test_explicit_model_mismatch_does_not_choose_another_model(monkeypatch):
    client_for(monkeypatch, lambda request: httpx.Response(200, json={"data": [{"id": "wrong"}]}))
    llm = model()
    assert not llm.load()
    assert llm.model == "assistant-vlm" and not llm._loaded
    assert "Expected model" in llm.last_error


def test_final_chunk_content_and_usage_both_survive(monkeypatch):
    payloads = []
    def handler(request):
        body = json.loads(request.content); payloads.append(body)
        if request.url.path.endswith("/input_tokens"):
            return httpx.Response(200, json={"input_tokens": 1904})
        event = {"choices": [{"delta": {"content": "你好"}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 1904, "completion_tokens": 2}}
        return httpx.Response(200, text="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n")
    client_for(monkeypatch, handler)
    events = list(model().generate_stream("看图", images_b64=["YQ=="]))
    assert "".join(text for text, meta in events) == "你好"
    assert events[-1][1]["done"] and events[-1][1]["usage"]["completion_tokens"] == 2
    assert payloads[0] == payloads[1]
    assert payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert payloads[0]["messages"][-1]["content"][1]["type"] == "image_url"


def test_budget_rejection_never_starts_generation(monkeypatch):
    requests = []
    def handler(request):
        requests.append(request.url.path)
        return httpx.Response(200, json={"input_tokens": 1905})
    client_for(monkeypatch, handler)
    with pytest.raises(ValueError, match="CONTEXT_BUDGET_EXCEEDED"):
        list(model().generate_stream("long request"))
    assert requests == ["/v1/chat/completions/input_tokens"]


@pytest.mark.parametrize("stream", [
    "data: [DONE]\n\n", 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
])
def test_incomplete_stream_is_failure(monkeypatch, stream):
    def handler(request):
        if request.url.path.endswith("/input_tokens"):
            return httpx.Response(200, json={"input_tokens": 10})
        return httpx.Response(200, text=stream)
    client_for(monkeypatch, handler)
    with pytest.raises(RuntimeError): list(model().generate_stream("hello"))


def test_provider_error_is_not_an_empty_success(monkeypatch):
    client_for(monkeypatch, lambda request: httpx.Response(503, text="not ready"))
    with pytest.raises(httpx.HTTPStatusError): list(model().generate_stream("hello"))
