#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read the actual B1 camera and diagnostics pages in headless Firefox."""

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
            legacy_hidden: document.querySelector('#debugLegacyStages').hidden,
            b1_hidden: document.querySelector('#debugB1Models').hidden,
            counter_label: document.querySelector('#debugB1Models').textContent.includes('拒绝/失败')
        };""")
        model_panel = value["model_panel"]
        if model_panel != {
            "rows": 10,
            "legacy_hidden": True,
            "b1_hidden": False,
            "counter_label": True,
        }:
            raise RuntimeError("B1 diagnostics panel does not match the deployed models")
        value["status"] = "B1_BROWSER_RENDER_PASS"
        (output / "browser.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        )
        return value
    finally:
        if session:
            call("DELETE", f"/session/{session}")
