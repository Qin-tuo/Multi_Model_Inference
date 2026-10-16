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

"""Upstream CLI adapted for local text/image requests; no RAG initialization."""

import base64
import json
import time
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich.console import Console

from app.config import Config
from app.llm import LLM
from app.monitor import get_system_stats, format_stats

console = Console()
app = typer.Typer(name="assistant", help="Orin NX headless assistant", add_completion=False)


def _load_llm(config: Config) -> LLM:
    llm = LLM(
        model=config.llm.model, base_url=config.llm.base_url,
        backend=config.llm.backend, max_tokens=config.llm.max_tokens,
        temperature=config.llm.temperature, timeout=config.llm.timeout,
        system_prompt=config.llm.system_prompt,
        context_tokens=config.llm.context_tokens,
        chat_template_kwargs=config.llm.chat_template_kwargs,
    )
    if not llm.load():
        raise RuntimeError(llm.last_error or "LLM failed to connect")
    return llm


def _image(path: Path, config: Config) -> str:
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
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, config.vision.jpeg_quality])
    if not ok:
        raise ValueError("Image cannot be encoded")
    return base64.b64encode(encoded).decode("ascii")


def _stream(llm: LLM, question: str, *, images=None, system_prompt=None, display=True):
    t0 = time.monotonic()
    first = None
    text = ""
    metadata = {}
    for content, meta in llm.generate_stream(question, images_b64=images, system_prompt=system_prompt):
        metadata.update(meta)
        if content:
            if first is None:
                first = time.monotonic()
            if display:
                console.print(content, end="", highlight=False, markup=False)
            text += content
    if not metadata.get("done") or not text.strip():
        raise RuntimeError("Model produced no complete text response")
    return {"text": text, "metadata": metadata,
            "elapsed_seconds": time.monotonic() - t0,
            "first_content_seconds": first - t0 if first is not None else None}


@app.command()
def ask(
    question: str = typer.Argument(...),
    config_path: Optional[str] = typer.Option(None, "--config", "-c"),
    image: Optional[Path] = typer.Option(None, "--image", exists=True, dir_okay=False),
    json_output: bool = typer.Option(False, "--json-output"),
):
    """Ask one text/image question. Physical audio is not exercised by this command."""
    llm = None
    try:
        config = Config.load(config_path)
        images = [_image(image, config)] if image else None
        llm = _load_llm(config)
        result = _stream(llm, question, images=images, display=not json_output,
                         system_prompt=config.vision.system_prompt if images else None)
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


@app.command(context_settings={'allow_extra_args':True,'ignore_unknown_options':True})
def services(ctx: typer.Context, action: str = typer.Argument(...,help='check, install, up, down, status or logs')):
    """Manage native backends and the Web app through user systemd."""
    from deploy.services import main as service_main
    raise typer.Exit(service_main([action,*ctx.args]))


@app.command()
def config(config_path: Optional[str] = typer.Option(None, "--config", "-c")):
    """Print the effective typed configuration."""
    typer.echo(json.dumps(Config.load(config_path).effective(), ensure_ascii=False, indent=2))


@app.command()
def info():
    """Show host statistics and configured model readiness."""
    console.print(format_stats(get_system_stats()))
    llm = _load_llm(Config.load())
    console.print(f"Ready: {llm.model}")
    llm.unload()


def main():
    app()


if __name__ == "__main__":
    main()
