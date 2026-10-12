from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
PROMPTED_GENERATOR = ROOT / "dataflow" / "operators" / "core_text" / "generate" / "prompted_generator.py"


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class _OperatorABC:
    pass


class _LLMServingABC:
    pass


class _Registry:
    def register(self):
        return lambda cls: cls


class _Storage:
    def __init__(self):
        self.df = pd.DataFrame([{"raw_content": "hello"}])
        self.writes = []

    def read(self, output_type):
        assert output_type == "dataframe"
        return self.df.copy()

    def write(self, dataframe):
        self.writes.append(dataframe.copy())
        return "cache.jsonl"


class _FailingServing:
    def generate_from_input(self, **kwargs):
        raise RuntimeError("controlled generation failure")


class _SuccessfulServing:
    def generate_from_input(self, **kwargs):
        return ["generated"]


def _load(monkeypatch):
    dataflow = types.ModuleType("dataflow")
    dataflow.get_logger = lambda: _Logger()
    core = types.ModuleType("dataflow.core")
    core.OperatorABC = _OperatorABC
    core.LLMServingABC = _LLMServingABC
    registry = types.ModuleType("dataflow.utils.registry")
    registry.OPERATOR_REGISTRY = _Registry()
    storage = types.ModuleType("dataflow.utils.storage")
    storage.DataFlowStorage = object

    monkeypatch.setitem(sys.modules, "dataflow", dataflow)
    monkeypatch.setitem(sys.modules, "dataflow.core", core)
    monkeypatch.setitem(sys.modules, "dataflow.utils", types.ModuleType("dataflow.utils"))
    monkeypatch.setitem(sys.modules, "dataflow.utils.registry", registry)
    monkeypatch.setitem(sys.modules, "dataflow.utils.storage", storage)

    return runpy.run_path(str(PROMPTED_GENERATOR), run_name="prompted_generator_failure_test")


@pytest.mark.cpu
def test_generation_failure_is_propagated_and_does_not_write(monkeypatch):
    namespace = _load(monkeypatch)
    generator = namespace["PromptedGenerator"](llm_serving=_FailingServing())
    storage = _Storage()

    with pytest.raises(RuntimeError, match="controlled generation failure"):
        generator.run(storage=storage)

    assert storage.writes == []


@pytest.mark.cpu
def test_successful_generation_still_writes_and_returns_output_key(monkeypatch):
    namespace = _load(monkeypatch)
    generator = namespace["PromptedGenerator"](llm_serving=_SuccessfulServing())
    storage = _Storage()

    result = generator.run(storage=storage)

    assert result == "generated_content"
    assert len(storage.writes) == 1
    assert storage.writes[0]["generated_content"].tolist() == ["generated"]
