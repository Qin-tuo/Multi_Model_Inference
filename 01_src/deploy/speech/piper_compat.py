"""Use maintained Piper's eSpeak phonemizer for the legacy VoxEdge import."""

from __future__ import annotations

import importlib
import sys
import threading
import types


def install_piper_alias():
    try:
        return importlib.import_module("piper_phonemize")
    except ModuleNotFoundError:
        from piper.phonemize_espeak import EspeakPhonemizer

        phonemizer = EspeakPhonemizer()
        lock = threading.Lock()
        module = types.ModuleType("piper_phonemize")

        def phonemize_espeak(text, voice):
            with lock:
                return phonemizer.phonemize(voice, text)

        module.phonemize_espeak = phonemize_espeak
        sys.modules["piper_phonemize"] = module
        return module
