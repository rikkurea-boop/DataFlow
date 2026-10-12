from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from pathlib import Path
import runpy
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[2]
SERVING = ROOT / "dataflow" / "serving" / "light_rag_serving.py"


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


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class _QueryParam:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _EmbeddingFunc:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _Rag:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def initialize_storages(self):
        return None

    async def aquery(self, question, system_prompt=None, param=None):
        return f"answer:{question}:{param.kwargs.get('mode')}"


async def _initialize_pipeline_status():
    return None


async def _openai_complete_if_cache(*args, **kwargs):
    return "ok"


async def _ollama_embed(*args, **kwargs):
    return []


class _TqdmAsyncio:
    @staticmethod
    async def gather(*tasks):
        return await asyncio.gather(*tasks)


def _load(monkeypatch):
    dataflow = types.ModuleType("dataflow")
    core = types.ModuleType("dataflow.core")
    core.LLMServingABC = _LLMServingABC
    logger_mod = types.ModuleType("dataflow.logger")
    logger_mod.get_logger = lambda: _Logger()

    lightrag = types.ModuleType("lightrag")
    lightrag.LightRAG = _Rag
    lightrag.QueryParam = _QueryParam
    llm_pkg = types.ModuleType("lightrag.llm")
    openai_mod = types.ModuleType("lightrag.llm.openai")
    openai_mod.openai_complete_if_cache = _openai_complete_if_cache
    ollama_mod = types.ModuleType("lightrag.llm.ollama")
    ollama_mod.ollama_embed = _ollama_embed
    utils_mod = types.ModuleType("lightrag.utils")
    utils_mod.EmbeddingFunc = _EmbeddingFunc
    kg_pkg = types.ModuleType("lightrag.kg")
    shared_mod = types.ModuleType("lightrag.kg.shared_storage")
    shared_mod.initialize_pipeline_status = _initialize_pipeline_status

    tqdm_pkg = types.ModuleType("tqdm")
    tqdm_asyncio_mod = types.ModuleType("tqdm.asyncio")
    tqdm_asyncio_mod.tqdm_asyncio = _TqdmAsyncio

    for name, module in {
        "dataflow": dataflow,
        "dataflow.core": core,
        "dataflow.logger": logger_mod,
        "lightrag": lightrag,
        "lightrag.llm": llm_pkg,
        "lightrag.llm.openai": openai_mod,
        "lightrag.llm.ollama": ollama_mod,
        "lightrag.utils": utils_mod,
        "lightrag.kg": kg_pkg,
        "lightrag.kg.shared_storage": shared_mod,
        "tqdm": tqdm_pkg,
        "tqdm.asyncio": tqdm_asyncio_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setenv("DF_API_KEY", "controlled-test-key")
    return runpy.run_path(str(SERVING), run_name="lightrag_scope_test")


@pytest.mark.cpu
def test_lightrag_dependencies_are_visible_to_module_helpers(monkeypatch):
    namespace = _load(monkeypatch)
    serving_cls = namespace["LightRAGServing"]
    serving_cls(document_list=[])

    rag = asyncio.run(
        namespace["initialize_rag"](
            "model", "https://example.invalid/v1", "key", "embed", "http://embed", 8, 32
        )
    )

    assert isinstance(rag, _Rag)


@pytest.mark.cpu
def test_query_param_is_visible_to_generate_from_input(monkeypatch):
    namespace = _load(monkeypatch)
    serving = namespace["LightRAGServing"](document_list=[])
    serving.rag = _Rag()

    result = asyncio.run(serving.generate_from_input(["q1", "q2"], "system"))

    assert result == ["answer:q1:hybrid", "answer:q2:hybrid"]
