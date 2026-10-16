# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LLM — Ollama or OpenAI-compatible backend (llama.cpp)."""

import httpx
import json
from app.request_io import RequestIO, llama_slots_idle, control_json
from typing import Optional, Iterator, Dict, Any


class LLM:
    def __init__(
        self,
        model: str = "",
        base_url: str = "http://localhost:8080",
        backend: str = "openai",
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: str = "",
        timeout: float = 120.0,
        context_tokens: int = 2048,
        chat_template_kwargs: Optional[dict] = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.backend = (backend or "openai").lower()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.context_tokens = context_tokens
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.last_error = None
        self._loaded = False
        self.requests = RequestIO()

    def load(self) -> bool:
        self._loaded = False
        self.last_error = None
        try:
            self._verify_identity(min(10,self.timeout),allow_select=True)
            self._loaded = True
            return True
        except Exception as e:
            self.last_error = str(e)
            print(f"LLM connection error: {e}")
            return False

    def _verify_identity(self,timeout,allow_select=False):
        path='/v1/models' if self.backend=='openai' else '/api/tags'
        payload=control_json(self.base_url+path,timeout)
        key,field=('data','id') if self.backend=='openai' else ('models','name')
        rows=payload.get(key) if isinstance(payload,dict) else None
        if not isinstance(rows,list) or not rows or any(not isinstance(row,dict) or not isinstance(row.get(field),str) or not row[field] for row in rows):
            raise ValueError('Invalid model identity response')
        names=[row[field] for row in rows]
        if not self.model and allow_select and self.backend=='openai':self.model=names[0]
        matches=self.model in names or (self.backend!='openai' and self.model.split(':')[0] in [n.split(':')[0] for n in names])
        if not matches:raise ValueError(f'Expected model {self.model!r}; available: {names}')

    def _messages(
        self, prompt: str, system_prompt: Optional[str] = None,
        few_shot: Optional[list[dict]] = None,
    ) -> list:
        msgs = []
        sp = system_prompt or self.system_prompt
        if sp:
            msgs.append({"role": "system", "content": sp})
        if few_shot:
            msgs.extend(few_shot)
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _messages_multimodal(
        self, prompt: str, images_b64: list[str],
        system_prompt: Optional[str] = None,
        few_shot: Optional[list[dict]] = None,
    ) -> list:
        msgs = []
        sp = system_prompt or self.system_prompt
        if sp:
            msgs.append({"role": "system", "content": sp})
        if few_shot:
            msgs.extend(few_shot)
        content: list[dict] = [{"type": "text", "text": prompt}]
        for b64 in images_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        msgs.append({"role": "user", "content": content})
        return msgs

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        images_b64: Optional[list[str]] = None,
        few_shot: Optional[list[dict]] = None,
        cancel_event=None,
        trim_history: bool = False,
        tools=None,
        tool_messages=None,
        history_group_sizes=None,
    ) -> Iterator[tuple]:
        """Yields (content, metadata) tuples. Pass images_b64 for multimodal VLM requests."""
        if not self._loaded:
            raise RuntimeError("LLM is not loaded")
        mt = max_tokens if max_tokens is not None else self.max_tokens
        t = temperature if temperature is not None else self.temperature
        if images_b64:
            msgs = self._messages_multimodal(prompt, images_b64, system_prompt, few_shot)
        else:
            msgs = self._messages(prompt, system_prompt, few_shot)
        if tool_messages: msgs.extend(tool_messages)

        if self.backend == "openai":
            history_pairs = 0
            group_sizes = []
            if trim_history:
                prior = few_shot or []
                group_sizes = history_group_sizes if history_group_sizes is not None else [2]*(len(prior)//2)
                if sum(group_sizes)!=len(prior) or any(type(n) is not int or not 2<=n<=8 for n in group_sizes):
                    raise ValueError('Trimmable history must contain complete text turn pairs and closed tool groups')
                offset=0
                for size in group_sizes:
                    group=prior[offset:offset+size];offset+=size;pending=set()
                    if group[0].get('role')!='user' or group[-1].get('role')!='assistant' or not all(isinstance(m.get('content'),str) for m in group):
                        raise ValueError('Trimmable history must contain complete text turn pairs')
                    for message in group[1:-1]:
                        if message.get('role')=='assistant' and message.get('tool_calls') and not pending:
                            ids=[c['id'] for c in message['tool_calls']]
                            if len(set(ids))!=len(ids):raise ValueError('Duplicate tool call ID in history')
                            pending=set(ids)
                        elif message.get('role')=='tool' and message.get('tool_call_id') in pending:pending.remove(message['tool_call_id'])
                        else:raise ValueError('Unpaired tool history')
                    if pending:raise ValueError('Incomplete tool results in history')
                history_pairs = len(group_sizes)
            yield from self._stream_openai(msgs, mt, t, cancel_event, history_pairs, tools, group_sizes)
        else:
            yield from self._stream_ollama(msgs, mt, t)

    def _stream_openai(self, messages, max_tokens, temperature, cancel_event=None, history_pairs=0, tools=None, history_group_sizes=None) -> Iterator[tuple]:
        # Use one payload for native multimodal token counting and inference.
        payload = {
            "model": self.model, "messages": messages,
            "chat_template_kwargs": self.chat_template_kwargs,
            "max_tokens": max_tokens, "temperature": temperature,
            "stream": True, "stream_options": {"include_usage": True},
        }
        if tools is not None:
            payload.update(tools=tools,tool_choice='auto' if tools else 'none',parallel_tool_calls=False)
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        with self.requests.operation(cancel_event,timeout=self.timeout) as operation, httpx.Client(timeout=self.timeout, trust_env=False) as client:
            dropped = 0
            while True:
                operation.check()
                count = client.post(f"{self.base_url}/v1/chat/completions/input_tokens", json=payload, extensions=operation.extensions)
                operation.check()
                count.raise_for_status()
                input_tokens = count.json()["input_tokens"]
                if type(input_tokens) is not int or input_tokens < 0:
                    raise ValueError("Invalid native input token count")
                if input_tokens + max_tokens + 16 <= self.context_tokens: break
                if dropped >= history_pairs: raise ValueError("CONTEXT_BUDGET_EXCEEDED")
                first = 1 if messages[0]['role']=='system' else 0
                size = history_group_sizes[dropped] if history_group_sizes is not None else 2
                del messages[first:first+size]
                dropped += 1
            yield ("", {"input_tokens_preflight": input_tokens, "history_pairs_kept":history_pairs-dropped,
                        "history_pairs_dropped":dropped})
            operation.check()
            final = {}
            finished = False
            calls = {}; buffered = ''
            with client.stream("POST", f"{self.base_url}/v1/chat/completions", json=payload, extensions=operation.extensions) as r:
                operation.check()
                r.raise_for_status()
                for line in r.iter_lines():
                    operation.check()
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        if not finished:
                            raise RuntimeError("Model stream ended without finish_reason")
                        if calls:
                            if final.get('finish_reason')!='tool_calls':raise RuntimeError('Incomplete tool call; nothing executed')
                            ordered=[calls[k] for k in sorted(calls)]
                            if any(not c['id'] or not c['function']['name'] for c in ordered):raise ValueError('Incomplete tool identity')
                            final.update(tool_calls=ordered,assistant_content=buffered)
                        elif tools is not None and buffered:yield(buffered,{})
                        yield ("", {**final, "done": True})
                        return
                    data = json.loads(raw)
                    if data.get("error"):
                        raise RuntimeError(f"Model stream error: {data['error']}")
                    for choice in data.get("choices", []):
                        delta = choice.get('delta') or {}
                        for part in delta.get('tool_calls',[]):
                            if not tools:raise ValueError('Model requested a disabled tool')
                            index=part.get('index')
                            if type(index) is not int or not 0<=index<2:raise ValueError('Too many model tool calls')
                            call=calls.setdefault(index,{'id':'','type':'function','function':{'name':'','arguments':''}})
                            if part.get('type','function')!='function':raise ValueError('Unsupported tool type')
                            if part.get('id'):
                                if call['id'] and call['id']!=part['id']:raise ValueError('Tool identity changed mid-stream')
                                call['id']=part['id']
                            function=part.get('function',{})
                            for field,limit in [('name',64),('arguments',4096)]:
                                value=function.get(field,'')
                                if not isinstance(value,str):raise ValueError('Invalid streamed tool argument')
                                call['function'][field]+=value
                                if len(call['function'][field])>limit:raise ValueError('Tool response budget exceeded')
                            if not isinstance(call['id'],str) or len(call['id'])>128:raise ValueError('Invalid tool ID')
                        content = delta.get("content")
                        if content:
                            if tools is None:yield (content,{})
                            else:
                                buffered+=content
                                if len(buffered)>16384:raise ValueError('Model text response budget exceeded')
                        if choice.get("finish_reason") is not None:
                            finished = True
                            final["finish_reason"] = choice["finish_reason"]
                    if data.get("usage"):
                        final["usage"] = data["usage"]
                        final["eval_count"] = data["usage"].get("completion_tokens")
            raise RuntimeError("Model stream disconnected before [DONE]")

    def _stream_ollama(self, messages, max_tokens, temperature) -> Iterator[tuple]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", f"{self.base_url}/api/chat", json={
                    "model": self.model, "messages": messages, "stream": True,
                    "keep_alive": "1h",
                    "options": {"num_predict": max_tokens, "temperature": temperature},
                }) as r:
                    if r.status_code != 200:
                        yield ("", {})
                        return
                    for line in r.iter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            done = data.get("done", False)
                            meta = {}
                            if done:
                                meta = {"done": True, "eval_count": data.get("eval_count", 0)}
                            if content:
                                yield (content, meta)
                            elif done:
                                yield ("", meta)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"LLM stream error: {e}")
            yield ("", {})

    def health_check(self) -> bool:
        if not self._loaded:
            return False
        try:
            self._verify_identity(min(5,self.timeout))
            self.last_error=None
            return True
        except Exception as exc:
            self.last_error=str(exc)
            return False

    def cancel(self, event):
        self.requests.cancel(event)

    def require_tools(self):
        props=control_json(self.base_url+'/props',min(5,self.timeout))
        caps=props.get('chat_template_caps',{})
        if caps.get('supports_tools') is not True or caps.get('supports_tool_calls') is not True:
            raise RuntimeError('Configured model template does not support local tools')

    def idle(self):
        return self.requests.idle() and llama_slots_idle(self.base_url)

    def unload(self):
        self._loaded = False
