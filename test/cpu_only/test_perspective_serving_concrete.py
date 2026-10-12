from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
import runpy
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PERSPECTIVE_SERVING = ROOT / "dataflow" / "serving" / "google_api_serving.py"


class _LLMServingABC(ABC):
    @abstractmethod
    def generate_from_input(self, user_inputs, system_prompt):
        pass

    @abstractmethod
    def start_serving(self):
        pass

    @abstractmethod
    def cleanup(self):
        pass


class _Discovery:
    @staticmethod
    def build(*args, **kwargs):
        return object()


def _module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _load(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "googleapiclient", _module("googleapiclient", discovery=_Discovery))
    monkeypatch.setitem(sys.modules, "dataflow", _module("dataflow"))
    monkeypatch.setitem(sys.modules, "dataflow.core", _module("dataflow.core", LLMServingABC=_LLMServingABC))
    return runpy.run_path(str(PERSPECTIVE_SERVING), run_name="perspective_serving_test")


@pytest.mark.cpu
def test_perspective_serving_is_concrete_and_supports_lifecycle_start(monkeypatch):
    namespace = _load(monkeypatch)
    serving_cls = namespace["PerspectiveAPIServing"]

    assert not inspect.isabstract(serving_cls)
    serving = serving_cls(max_workers=1)
    assert serving.start_serving() is None
