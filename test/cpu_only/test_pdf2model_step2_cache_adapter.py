from __future__ import annotations

import ast
import runpy
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
KBC_TEMPLATE = ROOT / "dataflow" / "statics" / "pipelines" / "gpu_pipelines" / "kbcleaning" / "kbcleaning_pipeline_vllm.py"
VQA_TEMPLATE = ROOT / "dataflow" / "statics" / "pipelines" / "api_pipelines" / "pdf_vqa_extract_pipeline.py"


class _Capture:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def run(self, *args, **kwargs):
        return None


class _FileStorage(_Capture):
    def step(self):
        return self


class _PipelineABC:
    def __init__(self, *args, **kwargs):
        pass

    def compile(self):
        return None


class _QAExtractPrompt:
    def build_prompt(self):
        return "prompt"


def _module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _load_template(monkeypatch, path: Path):
    modules = {
        "dataflow": _module("dataflow"),
        "dataflow.operators": _module("dataflow.operators"),
        "dataflow.operators.knowledge_cleaning": _module(
            "dataflow.operators.knowledge_cleaning",
            KBCChunkGenerator=_Capture,
            FileOrURLToMarkdownConverterFlash=_Capture,
            FileOrURLToMarkdownConverterAPI=_Capture,
            KBCTextCleaner=_Capture,
            QAExtractor=_Capture,
        ),
        "dataflow.operators.core_text": _module(
            "dataflow.operators.core_text",
            Text2MultiHopQAGenerator=_Capture,
            ChunkedPromptedGenerator=_Capture,
        ),
        "dataflow.utils": _module("dataflow.utils"),
        "dataflow.utils.storage": _module("dataflow.utils.storage", FileStorage=_FileStorage),
        "dataflow.serving": _module(
            "dataflow.serving",
            LocalModelLLMServing_vllm=_Capture,
            APILLMServing_request=_Capture,
        ),
        "dataflow.operators.pdf2vqa": _module(
            "dataflow.operators.pdf2vqa",
            MinerU2LLMInputOperator=_Capture,
            LLMOutputParser=_Capture,
            QA_Merger=_Capture,
            PDF_Merger=_Capture,
            VQAFormatter=_Capture,
        ),
        "dataflow.pipeline": _module("dataflow.pipeline", PipelineABC=_PipelineABC),
        "dataflow.prompts": _module("dataflow.prompts"),
        "dataflow.prompts.pdf2vqa": _module("dataflow.prompts.pdf2vqa", QAExtractPrompt=_QAExtractPrompt),
        "pypdf": _module("pypdf", PdfWriter=_Capture),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return runpy.run_path(str(path), run_name="pdf2model_template_test")


def _assert_under(path_value: str, root: Path):
    path = Path(path_value)
    assert path.is_absolute()
    assert root.resolve() in path.parents or path == root.resolve()


@pytest.mark.cpu
def test_kbc_custom_cache_consumes_detector_manifest_and_writes_selected_cache(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    namespace = _load_template(monkeypatch, KBC_TEMPLATE)
    pipeline_cls = namespace["KBCleaning_PDFvllm_GPUPipeline"]
    selected = tmp_path / "selected-cache"

    pipeline = pipeline_cls(cache_base=str(selected))

    assert pipeline.storage.kwargs["first_entry_file_name"] == str(selected / ".cache" / "pdf_list.jsonl")
    assert pipeline.storage.kwargs["cache_path"] == str(selected / ".cache" / "gpu")
    assert pipeline.extract_format_qa_step5.kwargs["output_json_file"] == str(selected / ".cache" / "data" / "qa.json")
    _assert_under(pipeline.knowledge_cleaning_step1.kwargs["intermediate_dir"], selected)


@pytest.mark.cpu
def test_vqa_custom_cache_consumes_detector_manifest_and_writes_selected_cache(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    namespace = _load_template(monkeypatch, VQA_TEMPLATE)
    pipeline_cls = namespace["PDF_VQA_extract_optimized_pipeline"]
    selected = tmp_path / "selected-cache"

    pipeline = pipeline_cls(cache_base=str(selected))

    assert pipeline.storage.kwargs["first_entry_file_name"] == str(selected / ".cache" / "pdf_list.jsonl")
    assert pipeline.storage.kwargs["cache_path"] == str(selected / ".cache" / "gpu")
    assert pipeline.vqa_format_converter.kwargs["output_json_file"] == str(selected / ".cache" / "data" / "qa.json")

    for component, key in [
        (pipeline.pdf_merger, "output_dir"),
        (pipeline.mineru_executor, "intermediate_dir"),
        (pipeline.llm_output_parser, "output_dir"),
        (pipeline.llm_output_parser, "intermediate_dir"),
        (pipeline.qa_merger, "output_dir"),
    ]:
        _assert_under(component.kwargs[key], selected)


@pytest.mark.cpu
@pytest.mark.parametrize(
    (template, class_name),
    [
        (KBC_TEMPLATE, "KBCleaning_PDFvllm_GPUPipeline"),
        (VQA_TEMPLATE, "PDF_VQA_extract_optimized_pipeline"),
    ],
)
def test_relative_custom_cache_resolves_from_step2_working_directory(monkeypatch, tmp_path, template, class_name):
    monkeypatch.chdir(tmp_path)
    namespace = _load_template(monkeypatch, template)
    pipeline = namespace[class_name](cache_base="relative-cache")
    expected = (tmp_path / "relative-cache").resolve()
    assert pipeline.storage.kwargs["first_entry_file_name"] == str(expected / ".cache" / "pdf_list.jsonl")
    assert pipeline.storage.kwargs["cache_path"] == str(expected / ".cache" / "gpu")


@pytest.mark.cpu
def test_kbc_standalone_without_cache_preserves_legacy_example_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    namespace = _load_template(monkeypatch, KBC_TEMPLATE)
    pipeline = namespace["KBCleaning_PDFvllm_GPUPipeline"]()

    assert pipeline.storage.kwargs["first_entry_file_name"] == "../../example_data/KBCleaningPipeline/kbc_test.jsonl"
    assert pipeline.storage.kwargs["cache_path"] == "./.cache/gpu"
    assert pipeline.knowledge_cleaning_step1.kwargs["intermediate_dir"] == "../example_data/KBCleaningPipeline/flash/"
    assert pipeline.extract_format_qa_step5.kwargs["output_json_file"] == "./.cache/data/qa.json"


@pytest.mark.cpu
def test_vqa_standalone_without_cache_preserves_legacy_example_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    namespace = _load_template(monkeypatch, VQA_TEMPLATE)
    pipeline = namespace["PDF_VQA_extract_optimized_pipeline"]()

    assert pipeline.storage.kwargs["first_entry_file_name"] == "./example_data/PDF2VQAPipeline/vqa_extract_test.jsonl"
    assert pipeline.storage.kwargs["cache_path"] == "./cache"
    assert pipeline.pdf_merger.kwargs["output_dir"] == "./cache"
    assert pipeline.mineru_executor.kwargs["intermediate_dir"] == "../example_data/PDF2VQAPipeline/flash/"
    assert pipeline.llm_output_parser.kwargs["output_dir"] == "./cache"
    assert pipeline.qa_merger.kwargs["output_dir"] == "./cache"
    assert pipeline.vqa_format_converter.kwargs["output_json_file"] == "./.cache/data/qa.json"


def _has_cache_cli_binding(path: Path, constructor_name: str) -> bool:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    cache_arg = False
    constructor_binding = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            if any(isinstance(arg, ast.Constant) and arg.value == "--cache" for arg in node.args):
                cache_arg = True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == constructor_name:
            for keyword in node.keywords:
                if keyword.arg == "cache_base" and isinstance(keyword.value, ast.Attribute):
                    if isinstance(keyword.value.value, ast.Name) and keyword.value.value.id == "args" and keyword.value.attr == "cache":
                        constructor_binding = True

    return cache_arg and constructor_binding


@pytest.mark.cpu
@pytest.mark.parametrize(
    (template, constructor_name),
    [
        (KBC_TEMPLATE, "KBCleaning_PDFvllm_GPUPipeline"),
        (VQA_TEMPLATE, "PDF_VQA_extract_optimized_pipeline"),
    ],
)
def test_step2_templates_bind_cli_cache_to_pipeline_constructor(template, constructor_name):
    assert _has_cache_cli_binding(template, constructor_name)
