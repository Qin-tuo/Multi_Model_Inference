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


def test_tool_history_is_preserved_and_evicted_as_a_whole_group(monkeypatch):
    history=[{'role':'user','content':'查询笔记'},
             {'role':'assistant','content':'','tool_calls':[{'id':'old-call','type':'function','function':{'name':'list_notes','arguments':'{}'}}]},
             {'role':'tool','tool_call_id':'old-call','content':'{"notes":[]}'},
             {'role':'assistant','content':'没有笔记'}]
    payloads=[]
    def handler(request):
        value=json.loads(request.content);payloads.append(value)
        if request.url.path.endswith('/input_tokens'):
            return httpx.Response(200,json={'input_tokens':2000 if len(value['messages'])>2 else 100})
        return httpx.Response(200,text='data: {"choices":[{"delta":{"content":"你好"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
    client_for(monkeypatch,handler)
    result=list(model().generate_stream('新问题',few_shot=history,trim_history=True,history_group_sizes=[4]))
    assert result[0][1]['history_pairs_dropped']==1
    assert payloads[0]['messages'][:4]==history
    assert payloads[-1]['messages']==[{'role':'user','content':'新问题'}]


@pytest.mark.parametrize('finish',['tool_calls','length'])
def test_streamed_tool_call_is_complete_before_any_side_effect_claim(monkeypatch,finish):
    from app.actions.schema import tools
    chunks=[{'choices':[{'delta':{'content':'已保存。','tool_calls':[{'index':0,'id':'tool-1','type':'function','function':{'name':'create_note','arguments':'{"title":"测'}}]}}]},
            {'choices':[{'delta':{'tool_calls':[{'index':0,'function':{'arguments':'试","body":"内容"}'}}]},'finish_reason':finish}]}]
    def handler(request):
        if request.url.path.endswith('/input_tokens'):return httpx.Response(200,json={'input_tokens':100})
        return httpx.Response(200,text=''.join('data: '+json.dumps(c,ensure_ascii=False)+'\n\n' for c in chunks)+'data: [DONE]\n\n')
    client_for(monkeypatch,handler)
    if finish=='length':
        with pytest.raises(RuntimeError,match='Incomplete tool'):list(model().generate_stream('保存',tools=tools()))
    else:
        events=list(model().generate_stream('保存',tools=tools()))
        assert not any(text for text,_ in events)
        call=events[-1][1]['tool_calls'][0]
        assert json.loads(call['function']['arguments'])=={'title':'测试','body':'内容'}
        assert events[-1][1]['assistant_content']=='已保存。'
