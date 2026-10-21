#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read the actual current camera and diagnostics pages in headless Firefox."""

import base64
import json
import time
from urllib.request import Request, urlopen


def probe(base, webdriver, output):
    session = None

    def call(method, path, payload=None):
        request = Request(
            webdriver + path,
            method=method,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=30) as response:
            value = json.load(response)["value"]
        if isinstance(value, dict) and value.get("error"):
            raise RuntimeError(str(value))
        return value

    def js(script):
        return call(
            "POST", f"/session/{session}/execute/sync", {"script": script, "args": []}
        )

    def wait(script):
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            if js(script):
                return
            time.sleep(0.2)
        raise TimeoutError("Browser condition: " + script)

    try:
        session = call(
            "POST",
            "/session",
            {
                "capabilities": {
                    "alwaysMatch": {
                        "browserName": "firefox",
                        "moz:firefoxOptions": {"args": ["-headless"]},
                    }
                }
            },
        )["sessionId"]
        call("POST", f"/session/{session}/url", {"url": base})
        wait(
            "return !document.querySelector('#visionTools').hidden && !document.querySelector('#sendText').disabled"
        )
        wait("return document.querySelector('#camImg').naturalWidth > 0")
        value = js("""return {scene:document.querySelector('#sceneInfo').textContent,
            camera_width:document.querySelector('#camImg').naturalWidth,
            camera_height:document.querySelector('#camImg').naturalHeight,
            targets:document.querySelector('#visionTarget').options.length,
            faces:document.querySelector('#visionFace').options.length,
            enroll_disabled:document.querySelector('#visionEnroll').disabled};""")
        js("document.querySelector('#visionList').click()")
        wait(
            "return document.querySelector('#visionIdentities').textContent.length > 0"
        )
        (output / "browser-camera.png").write_bytes(
            base64.b64decode(call("GET", f"/session/{session}/screenshot"))
        )
        call("POST", f"/session/{session}/url", {"url": base + "/debug"})
        wait(
            "return document.body.textContent.includes('检测分割') && document.body.textContent.includes('物体特征')"
        )
        (output / "browser-debug.png").write_bytes(
            base64.b64decode(call("GET", f"/session/{session}/screenshot"))
        )
        value["debug_text"] = js("return document.body.textContent")
        value["model_panel"] = js("""return {
            rows: document.querySelector('#debugModels').children.length,
            panel_hidden: document.querySelector('#debugModelPanel').hidden,
            counter_label: document.querySelector('#debugModelPanel').textContent.includes('拒绝/失败')
        };""")
        model_panel = value["model_panel"]
        if model_panel != {
            "rows": 10,
            "panel_hidden": False,
            "counter_label": True,
        }:
            raise RuntimeError("Model diagnostics panel does not match the deployed models")
        value["status"] = "BROWSER_RENDER_PASS"
        (output / "browser.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        )
        return value
    finally:
        if session:
            call("DELETE", f"/session/{session}")



def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8092")
    parser.add_argument("--webdriver", default="http://127.0.0.1:4444")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    probe(args.base_url, args.webdriver, args.output)


if __name__ == "__main__":
    main()
