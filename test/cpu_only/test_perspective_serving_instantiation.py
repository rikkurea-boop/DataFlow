from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import runpy
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[2]
SERVING = ROOT / "dataflow" / "serving" / "google_api_serving.py"
FILTER = ROOT / "dataflow" / "operators" / "general_text" / "filter" / "perspective_filter.py"


class _LLMServingABC(ABC):
    @abstractmethod
    def generate_from_input(self, user_inputs: list[str], system_prompt: str):
        pass

    @abstractmethod
    def start_serving(self):
        pass

    @abstractmethod
    def cleanup(self):
        pass


def _load_serving(monkeypatch):
    dataflow = types.ModuleType("dataflow")
    core = types.ModuleType("dataflow.core")
    core.LLMServingABC = _LLMServingABC
    dataflow.core = core

    discovery = types.ModuleType("googleapiclient.discovery")
    discovery.build = lambda *args, **kwargs: object()
    googleapiclient = types.ModuleType("googleapiclient")
    googleapiclient.discovery = discovery

    monkeypatch.setitem(sys.modules, "dataflow", dataflow)
    monkeypatch.setitem(sys.modules, "dataflow.core", core)
    monkeypatch.setitem(sys.modules, "googleapiclient", googleapiclient)
    monkeypatch.setitem(sys.modules, "googleapiclient.discovery", discovery)
    monkeypatch.setenv("GOOGLE_API_KEY", "controlled-test-key")

    return runpy.run_path(str(SERVING), run_name="perspective_serving_test")


@pytest.mark.cpu
def test_perspective_serving_is_concrete_and_has_noop_start(monkeypatch):
    namespace = _load_serving(monkeypatch)
    serving_cls = namespace["PerspectiveAPIServing"]

    serving = serving_cls(max_workers=2)

    assert serving.max_workers == 2
    assert serving.start_serving() is None
    assert serving.cleanup() is None


@pytest.mark.cpu
def test_builtin_filter_constructs_perspective_serving_directly():
    source = FILTER.read_text(encoding="utf-8")
    assert "PerspectiveAPIServing(max_workers=10)" in source
