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


def test_history_trims_whole_old_pairs_using_native_counts_and_keeps_current_image(monkeypatch):
    counts=[];inference=[]
    def handler(request):
        payload=json.loads(request.content)
        if request.url.path.endswith('/input_tokens'):
            counts.append(payload)
            return httpx.Response(200,json={'input_tokens':2000 if len(payload['messages'])>4 else 100})
        inference.append(payload)
        return httpx.Response(200,text='data: {"choices":[{"delta":{"content":"回答"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
    client_for(monkeypatch,handler)
    history=[{'role':'user','content':'old'},{'role':'assistant','content':'old reply'},
             {'role':'user','content':'recent'},{'role':'assistant','content':'recent reply'}]
    events=list(model().generate_stream('now',system_prompt='system',images_b64=['YQ=='],few_shot=history,trim_history=True))
    assert events[0][1]['history_pairs_dropped']==1 and events[0][1]['history_pairs_kept']==1
    assert len(counts)==2 and counts[-1]==inference[0]
    assert inference[0]['messages'][0]['role']=='system'
    assert inference[0]['messages'][1]['content']=='recent'
    assert inference[0]['messages'][-1]['content'][1]['type']=='image_url'
    assert len(history)==4  # No mutation of the controller's snapshot.


def test_unfit_current_turn_does_not_silently_truncate_user_request(monkeypatch):
    calls=[]
    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200,json={'input_tokens':2000})
    client_for(monkeypatch,handler)
    with pytest.raises(ValueError,match='CONTEXT_BUDGET_EXCEEDED'):
        list(model().generate_stream('too long',few_shot=[{'role':'user','content':'old'},{'role':'assistant','content':'reply'}],trim_history=True))
    assert len(calls)==2 and all(p.endswith('/input_tokens') for p in calls)


def test_history_trimming_rejects_an_incomplete_or_tool_protocol_pair():
    with pytest.raises(ValueError,match='complete text turn pairs'):
        list(model().generate_stream('now',few_shot=[{'role':'tool','content':'unpaired'}],trim_history=True))
