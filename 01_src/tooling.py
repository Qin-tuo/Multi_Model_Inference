# SPDX-License-Identifier: Apache-2.0
"""Task selection only; implementations own their arguments and resource leases."""

import argparse
from importlib import import_module
import sys


TASKS = {
    "build": {
        "artifacts": "deploy.vision.artifacts",
        "engines": "deploy.vision.build",
        "export-standard": "deploy.vision.export_standard",
        "export-segmentation": "deploy.vision.export_segmentation",
        "export-attributes": "deploy.vision.export_attributes",
        "vad": "deploy.speech.prepare_vad",
        "speech": "deploy.speech.prepare",
    },
    "validate": {
        "pipeline": "validation.pipeline",
        "recovery": "validation.recovery",
        "services": "validation.services",
        "browser": "validation.browser",
        "restore": "validation.restore",
        "offline": "validation.offline",
        "actions": "validation.actions",
        "action-cancel": "validation.action_cancel",
        "session": "validation.session",
        "turn-cancel": "validation.turn_cancel",
        "cancellation": "validation.cancellation",
        "coexistence": "validation.coexistence",
        "web": "validation.web",
        "service-recovery": "validation.service_recovery",
        "speech-session": "validation.speech_session",
        "speech": "validation.speech",
        "voice-interruptions": "validation.voice_interruptions",
        "web-transport": "validation.web_transport",
        "gestures": "validation.gestures",
        "gesture-replay": "validation.gesture_replay",
        "pose-replay": "validation.pose_replay",
        "rgbd": "validation.rgbd",
        "browser-session": "validation.browser_session",
        "browser-cancel": "validation.browser_cancel",
        "attributes": "validation.attribute_reference",
        "segmentation": "validation.segmentation_reference",
        "face": "validation.face_reference",
        "expression": "validation.expression_reference",
        "standard-models": "validation.standard_reference",
    },
    "benchmark": {
        "freeze": "benchmarks.freeze",
        "freeze-models": "benchmarks.model_protocol",
        "fixtures": "benchmarks.fixtures",
        "vision": "benchmarks.vision_modules",
        "stream": "benchmarks.vision_stream",
        "stream-summary": "benchmarks.vision_summary",
        "models": "benchmarks.model_modules",
        "live": "benchmarks.observe_live_service",
        "models-summary": "benchmarks.summarize_models",
        "models-plot": "benchmarks.plot_models",
        "motion": "benchmarks.motion",
        "gaze-coverage": "benchmarks.gaze_coverage",
    },
    "runtime": {
        "bundle": "deploy.bundle",
        "package": "deploy.package",
    },
}


def dispatch(group):
    choices = TASKS[group]
    parser = argparse.ArgumentParser(
        description=f"{group} tools. Use <task> --help for that task's arguments."
    )
    parser.add_argument("task", choices=choices)
    selected = parser.parse_args(sys.argv[1:2])
    sys.argv = [f"{sys.argv[0]} {selected.task}", *sys.argv[2:]]
    return import_module(choices[selected.task]).main()
