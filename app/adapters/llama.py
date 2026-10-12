"""Configure Pipecat's maintained OpenAI service for local llama.cpp."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from pipecat.services.openai.llm import OpenAILLMService


@dataclass(frozen=True)
class LlamaConfig:
    """Resolved model identity and limits from the project's locked manifest."""

    model: str
    base_url: str = "http://127.0.0.1:8088/v1"
    max_tokens: int = 128
    temperature: float = 0.2

    def __post_init__(self) -> None:
        endpoint = urlparse(self.base_url)
        if endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("llama.cpp endpoint must be local HTTP")
        if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
            raise ValueError("endpoint cannot contain credentials, a query, or a fragment")
        if endpoint.path.rstrip("/") != "/v1":
            raise ValueError("llama.cpp endpoint must include /v1")
        if not self.model or self.max_tokens <= 0 or not 0 <= self.temperature <= 2:
            raise ValueError("invalid model or generation settings")

    @classmethod
    def from_manifest(
        cls, manifest_path: Path, *, base_url: str = "http://127.0.0.1:8088/v1"
    ) -> "LlamaConfig":
        """Use an explicit server alias, or derive the name from the locked repo."""
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = manifest["models"]["vlm"]
        if entry.get("backend") != "llama.cpp":
            raise ValueError("the primary VLM manifest must select llama.cpp")
        model = entry.get("server_model_id")
        if not model:
            repository = entry.get("repository", "")
            model = repository.rsplit("/", 1)[-1].removesuffix("-GGUF")
        return cls(
            model=model,
            base_url=base_url,
            max_tokens=int(entry["max_output_tokens"]),
        )


def create_llama_service(config: LlamaConfig) -> OpenAILLMService:
    """Use the current upstream Settings API and the local OpenAI-compatible API."""
    return OpenAILLMService(
        api_key="local-no-key",
        base_url=config.base_url,
        settings=OpenAILLMService.Settings(
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        ),
        retry_on_timeout=False,
    )
