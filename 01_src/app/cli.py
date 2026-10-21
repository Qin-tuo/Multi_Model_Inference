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

"""当前应用命令行入口：查看配置与状态、管理服务、发送单次文字或图像请求。"""

import base64
import json
import time
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich.console import Console

from app.config import Config
from app.conversation.llm import LLM
from app.monitoring.monitor import get_system_stats, format_stats

console = Console()
app = typer.Typer(
    name="assistant", help="Orin NX headless assistant", add_completion=False
)


def _load_llm(config: Config) -> LLM:
    """用有效配置构建并连接语言模型适配器，后端不可用时抛出明确错误。"""
    llm = LLM(
        model=config.llm.model,
        base_url=config.llm.base_url,
        backend=config.llm.backend,
        max_tokens=config.llm.max_tokens,
        temperature=config.llm.temperature,
        timeout=config.llm.timeout,
        system_prompt=config.llm.system_prompt,
        context_tokens=config.llm.context_tokens,
        chat_template_kwargs=config.llm.chat_template_kwargs,
    )
    if not llm.load():
        raise RuntimeError(llm.last_error or "LLM failed to connect")
    return llm


def _image(path: Path, config: Config) -> str:
    """限制输入图片文件大小，按配置边界缩放后生成 JPEG Base64，拒绝不可解码图像。"""
    import cv2

    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("Image exceeds 10 MiB")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("Image cannot be decoded")
    h, w = image.shape[:2]
    scale = min(1, config.vision.width / w, config.vision.height / h)
    if scale < 1:
        image = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))))
    ok, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, config.vision.jpeg_quality]
    )
    if not ok:
        raise ValueError("Image cannot be encoded")
    return base64.b64encode(encoded).decode("ascii")


def _stream(llm: LLM, question: str, *, images=None, system_prompt=None, display=True):
    """消费模型流并累计文本与元数据，可选实时打印；记录总耗时及首个内容时间。"""
    t0 = time.monotonic()
    first = None
    text = ""
    metadata = {}
    for content, meta in llm.generate_stream(
        question, images_b64=images, system_prompt=system_prompt
    ):
        metadata.update(meta)
        if content:
            if first is None:
                first = time.monotonic()
            if display:
                console.print(content, end="", highlight=False, markup=False)
            text += content
    if not metadata.get("done") or not text.strip():
        raise RuntimeError("Model produced no complete text response")
    return {
        "text": text,
        "metadata": metadata,
        "elapsed_seconds": time.monotonic() - t0,
        "first_content_seconds": first - t0 if first is not None else None,
    }


@app.command()
def ask(
    question: str = typer.Argument(...),
    config_path: Optional[str] = typer.Option(None, "--config", "-c"),
    image: Optional[Path] = typer.Option(None, "--image", exists=True, dir_okay=False),
    json_output: bool = typer.Option(False, "--json-output"),
):
    """执行一次文字或文件图像问答，按选项输出正文或 JSON，结束时卸载本地适配器。"""
    llm = None
    try:
        config = Config.load(config_path)
        images = [_image(image, config)] if image else None
        llm = _load_llm(config)
        result = _stream(
            llm,
            question,
            images=images,
            display=not json_output,
            system_prompt=config.vision.system_prompt if images else None,
        )
        result["mode"] = "file_image_text" if image else "text"
        result["model"] = llm.model
        if json_output:
            typer.echo(json.dumps(result, ensure_ascii=False))
        else:
            console.print()
    except (ValueError, OSError, RuntimeError, httpx.HTTPError) as exc:
        typer.echo(f"Request failed: {exc}", err=True)
        raise typer.Exit(1)
    finally:
        if llm is not None:
            llm.unload()


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def services(
    ctx: typer.Context,
    action: str = typer.Argument(..., help="check, install, up, down, status or logs"),
):
    """把服务操作及剩余参数交给 systemd 管理入口，复用同一套启动与归属校验。"""
    from deploy.services import main as service_main

    raise typer.Exit(service_main([action, *ctx.args]))


@app.command()
def config(config_path: Optional[str] = typer.Option(None, "--config", "-c")):
    """加载并输出最终有效配置，便于检查参数覆盖结果。"""
    typer.echo(
        json.dumps(Config.load(config_path).effective(), ensure_ascii=False, indent=2)
    )


@app.command()
def info():
    """展示主机资源快照，并检查配置中的语言模型是否就绪。"""
    console.print(format_stats(get_system_stats()))
    llm = _load_llm(Config.load())
    console.print(f"Ready: {llm.model}")
    llm.unload()


def main():
    """启动 Typer 命令分发器，交由各命令解析参数和处理退出码。"""
    app()


if __name__ == "__main__":
    main()
