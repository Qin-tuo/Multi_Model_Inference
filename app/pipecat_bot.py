"""A local llama.cpp bot using Pipecat 1.12's maintained pipeline APIs.

This file integrates upstream processors. It does not implement an alternative
session manager, turn loop, cancellation state machine, or scheduling framework.
The text/image smoke path is an initial integration check, not full J0/JT approval.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import sys
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Sequence

from pipecat.frames.frames import EndFrame, ErrorFrame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response import LLMFullResponseAggregator
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.workers.runner import WorkerRunner

from app.adapters.llama import LlamaConfig, create_llama_service

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_MESSAGE = (
    "你是运行在用户本机的中文个人助手。回答简洁、准确。"
    "只有工具返回成功后才能说操作已完成；没有记忆证据时说明不知道。"
)


@dataclass
class BotPipeline:
    """The upstream objects available to transport, memory and tool plugins."""

    pipeline: Pipeline
    worker: PipelineWorker
    context: LLMContext
    aggregators: LLMContextAggregatorPair
    llm: FrameProcessor
    responses: LLMFullResponseAggregator


def build_pipeline(
    config: LlamaConfig,
    messages: list[dict[str, Any]],
    *,
    llm: FrameProcessor | None = None,
    before_llm: Sequence[FrameProcessor] = (),
    after_llm: Sequence[FrameProcessor] = (),
    tools: Any = None,
    conversation_id: str | None = None,
) -> BotPipeline:
    """Assemble upstream context/service/worker components and plugin hooks.

    ``llm`` injection is for framework-path tests or another maintained service.
    Memory belongs in context processors; tools use Pipecat's ToolsSchema or
    FunctionSchema handlers. The application does not own turn transitions.
    """
    context = LLMContext(messages=messages) if tools is None else LLMContext(messages=messages, tools=tools)
    aggregators = LLMContextAggregatorPair(context)
    service = llm if llm is not None else create_llama_service(config)
    responses = LLMFullResponseAggregator()
    pipeline = Pipeline(
        [
            aggregators.user(),
            *before_llm,
            service,
            *after_llm,
            responses,
            aggregators.assistant(),
        ]
    )
    worker = PipelineWorker(
        pipeline,
        conversation_id=conversation_id,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=None,
        enable_rtvi=False,
    )
    return BotPipeline(pipeline, worker, context, aggregators, service, responses)


def input_messages(prompt: str, image: Path | None = None) -> list[dict[str, Any]]:
    """Build Pipecat-compatible messages from bounded local text/image inputs."""
    if not prompt.strip() or len(prompt.encode("utf-8")) > 65536:
        raise ValueError("prompt must contain text and fit within 64 KiB")
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_MESSAGE}]
    if image is None:
        messages.append({"role": "user", "content": prompt})
    else:
        mime, _ = mimetypes.guess_type(image.name)
        if mime not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError("image must be JPEG, PNG or WebP")
        if image.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("image exceeds the 16 MiB input bound")
        payload = base64.b64encode(image.read_bytes()).decode("ascii")
        messages.append(
            LLMContext.create_image_url_message(
                url=f"data:{mime};base64,{payload}", text=prompt
            )
        )
    return messages


async def run_smoke(
    config: LlamaConfig,
    messages: list[dict[str, Any]],
    *,
    timeout_seconds: float = 90,
    llm: FrameProcessor | None = None,
    tools: Any = None,
) -> dict[str, Any]:
    """Run one input through a real Pipecat worker; finish via its EndFrame."""
    if timeout_seconds <= 0:
        raise ValueError("timeout must be positive")
    bot = build_pipeline(config, messages, llm=llm, tools=tools)
    completions: list[str] = []
    errors: list[str] = []

    @bot.responses.event_handler("on_completion")
    async def on_completion(_aggregator, completion: str, completed: bool) -> None:
        if completed:
            if completion.strip():
                completions.append(completion)
            # Empty completions can be a tool round. The maintained LLM service
            # owns tool continuation, so don't terminate that intermediate round.
            if completion.strip() or tools is None:
                await bot.worker.queue_frame(EndFrame())

    @bot.worker.event_handler("on_pipeline_error")
    async def on_pipeline_error(_worker, frame: ErrorFrame) -> None:
        errors.append(frame.error)
        await bot.worker.queue_frame(EndFrame())

    runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
    await runner.add_workers(bot.worker)
    await bot.worker.queue_frame(LLMRunFrame())
    started_ns = time.monotonic_ns()
    async with asyncio.timeout(timeout_seconds):
        await runner.run()
    elapsed_ns = time.monotonic_ns() - started_ns
    if errors:
        raise RuntimeError("Pipecat pipeline failed: " + "; ".join(errors))
    if not completions:
        raise RuntimeError("Pipecat finished without a nonempty completed response")
    return {
        "schema_version": 1,
        "framework": "pipecat-ai",
        "framework_version": version("pipecat-ai"),
        "backend": "llama.cpp" if llm is None else "injected_framework_test_processor",
        "model": config.model,
        "base_url": config.base_url,
        "real_model_execution": llm is None,
        "scope": "text_image_pipeline_smoke",
        "elapsed_ns": elapsed_ns,
        "response": completions[-1],
        "context_message_count": len(bot.context.get_messages()),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Pipecat → llama.cpp integration")
    parser.add_argument("--text-smoke", action="store_true", help="Run a single real model request")
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument("--prompt", default=None)
    prompts.add_argument("--prompt-file", type=Path, help="UTF-8 local prompt file")
    parser.add_argument("--image", type=Path, help="Optional local JPEG/PNG/WebP")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "models" / "manifest.json")
    parser.add_argument("--base-url", default="http://127.0.0.1:8088/v1")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--output", type=Path, help="Also write the actual result JSON")
    parser.add_argument("--memory-dir", type=Path, help="Explicit local Mem0 store to retrieve from")
    parser.add_argument("--user", default="owner", help="Manually selected memory profile")
    parser.add_argument("--embedding-path", type=Path,
                        default=Path(os.environ.get("ASSISTANT_CACHE_DIR", Path.home() / ".cache/multi-model-assistant")) / "models/bge-small-zh-v1.5-onnx")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.text_smoke:
        print("Specify --text-smoke to run the initial Pipecat integration check.", file=sys.stderr)
        return 2
    try:
        if args.prompt_file is not None:
            if args.prompt_file.stat().st_size > 65536:
                raise ValueError("prompt file exceeds 64 KiB")
            prompt = args.prompt_file.read_text(encoding="utf-8")
        else:
            prompt = args.prompt or "请用一句话介绍你可以帮助我做什么。"
        config = LlamaConfig.from_manifest(args.manifest, base_url=args.base_url)
        messages = input_messages(prompt, args.image)
        memory_evidence = None
        if args.memory_dir:
            from app.memory import Mem0Adapter
            from app.context_memory import memory_message
            with Mem0Adapter(args.memory_dir, bge_path=args.embedding_path, embedding_backend="onnx",
                             llama_base_url=config.base_url, llama_model=config.model) as memory:
                found = memory.search(args.user, prompt, top_k=5)
                evidence_message, memory_evidence = memory_message(found["items"], config.base_url)
                memory_evidence["retrieval_method"] = found["retrieval_method"]
                memory_evidence["generation"] = found["generation"]
            messages.insert(-1, evidence_message)
        result = asyncio.run(run_smoke(config, messages, timeout_seconds=args.timeout))
        if memory_evidence is not None:
            result["memory_evidence"] = memory_evidence
        output = json.dumps(result, ensure_ascii=False, indent=2)
        print(output)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output + "\n", encoding="utf-8")
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, TimeoutError) as error:
        print(f"Pipecat integration failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
