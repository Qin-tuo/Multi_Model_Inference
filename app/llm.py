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
from app.request_io import RequestIO, llama_slots_idle
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
            with httpx.Client(timeout=10.0, trust_env=False) as client:
                if self.backend == "openai":
                    r = client.get(f"{self.base_url}/v1/models")
                    if r.status_code != 200:
                        return False
                    models = [m.get("id", "") for m in r.json().get("data", [])]
                    if not models:
                        return False
                    if self.model and self.model not in models:
                        raise ValueError(f"Expected model {self.model!r}; available: {models}")
                    if not self.model:
                        self.model = models[0]
                else:
                    r = client.get(f"{self.base_url}/api/tags")
                    if r.status_code != 200:
                        return False
                    names = [m.get("name", "") for m in r.json().get("models", [])]
                    base = self.model.split(":")[0]
                    if base not in [n.split(":")[0] for n in names] and self.model not in names:
                        print(f"Model '{self.model}' not found. Available: {', '.join(names)}")
                        return False
            self._loaded = True
            return True
        except Exception as e:
            self.last_error = str(e)
            print(f"LLM connection error: {e}")
            return False

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

        if self.backend == "openai":
            yield from self._stream_openai(msgs, mt, t, cancel_event)
        else:
            yield from self._stream_ollama(msgs, mt, t)

    def _stream_openai(self, messages, max_tokens, temperature, cancel_event=None) -> Iterator[tuple]:
        # Use one payload for native multimodal token counting and inference.
        payload = {
            "model": self.model, "messages": messages,
            "chat_template_kwargs": self.chat_template_kwargs,
            "max_tokens": max_tokens, "temperature": temperature,
            "stream": True, "stream_options": {"include_usage": True},
        }
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        with self.requests.operation(cancel_event) as operation, httpx.Client(timeout=self.timeout, trust_env=False) as client:
            count = client.post(f"{self.base_url}/v1/chat/completions/input_tokens", json=payload, extensions=operation.extensions)
            operation.check()
            count.raise_for_status()
            input_tokens = count.json()["input_tokens"]
            if type(input_tokens) is not int or input_tokens < 0:
                raise ValueError("Invalid native input token count")
            if input_tokens + max_tokens + 16 > self.context_tokens:
                raise ValueError("CONTEXT_BUDGET_EXCEEDED")
            yield ("", {"input_tokens_preflight": input_tokens})
            operation.check()
            final = {}
            finished = False
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
                        yield ("", {**final, "done": True})
                        return
                    data = json.loads(raw)
                    if data.get("error"):
                        raise RuntimeError(f"Model stream error: {data['error']}")
                    for choice in data.get("choices", []):
                        content = (choice.get("delta") or {}).get("content")
                        if content:
                            yield (content, {})
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
            with httpx.Client(timeout=5.0, trust_env=False) as client:
                url = f"{self.base_url}/v1/models" if self.backend == "openai" else f"{self.base_url}/api/tags"
                return client.get(url).status_code == 200
        except Exception:
            return False

    def cancel(self, event):
        self.requests.cancel(event)

    def idle(self):
        return self.requests.idle() and llama_slots_idle(self.base_url)

    def unload(self):
        self._loaded = False
