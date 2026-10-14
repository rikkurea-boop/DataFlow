"""Filesystem-boundary regressions for current-run Text/PDF QA artifacts.

The expensive generation backends are replaced, not the selected-cache file
system operations or the real conversion script.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from dataflow.cli_funcs import cli_text, cli_pdf
from dataflow.operators.knowledge_cleaning.generate.qa_extract import QAExtractor
from dataflow.operators.pdf2vqa.generate.vqa_formatter import VQAFormatter


pytestmark = pytest.mark.cpu
ROOT = Path(__file__).resolve().parents[2]
CONVERTER = ROOT / "dataflow/cli_funcs/text2model_pipeline/merge_filter_qa_pairs.py"
ALPACA = [{"instruction": "Explain", "input": "New question", "output": "New answer"}]
VQA = [{"messages": [{"role": "user", "content": "New question"},
                     {"role": "assistant", "content": "New answer"}], "images": []}]


class _Storage:
    def __init__(self, df):
        self.df = df
        self.writes = []

    def read(self, output_type="dataframe"):
        return self.df

    def write(self, value):
        self.writes.append(value)


def _prepare(cache, fmt="alpaca"):
    data = cache / ".cache" / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "qa.json").write_text(json.dumps(VQA if fmt == "sharegpt" else ALPACA), encoding="utf-8")
    (data / "dataset_info.json").write_text('{"previous":{"file_name":"qa.json"}}', encoding="utf-8")
    return data


@pytest.mark.parametrize("kind", ["text", "pdf"])
def test_no_valid_samples_cannot_leave_stale_producer_output(tmp_path, monkeypatch, kind):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "qa.json"
    output.write_text(json.dumps(ALPACA), encoding="utf-8")
    if kind == "text":
        op = QAExtractor(output_json_file=str(output))
        storage = _Storage(pd.DataFrame({"QA_pairs": [[]]}))
        result = op.run(storage)
        assert result == ["instruction", "input", "output"]
    else:
        op = VQAFormatter(output_json_file=str(output))
        storage = _Storage(pd.DataFrame({"vqa_pair": [{"question": "", "answer": "x"}]}))
        result = op.run(storage, input_qa_item_key="vqa_pair")
        assert result == ["messages", "images"]
    assert not output.exists()
    assert storage.writes == []


@pytest.mark.parametrize("kind", ["text", "pdf"])
def test_valid_producer_overwrites_stale_json(tmp_path, monkeypatch, kind):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "qa.json"
    output.write_text('["old"]', encoding="utf-8")
    if kind == "text":
        op = QAExtractor(output_json_file=str(output))
        storage = _Storage(pd.DataFrame({"QA_pairs": [[{"question": "New question", "answer": "New answer"}]]}))
        op.run(storage)
        result = json.loads(output.read_text(encoding="utf-8"))
        assert result[0]["input"] == "New question"
    else:
        op = VQAFormatter(output_json_file=str(output))
        storage = _Storage(pd.DataFrame({"vqa_pair": [{"question": "New question", "answer": "New answer"}]}))
        op.run(storage, input_qa_item_key="vqa_pair")
        result = json.loads(output.read_text(encoding="utf-8"))
        assert result == VQA
    assert len(storage.writes) == 1


@pytest.mark.parametrize("failure", ["empty", "missing_enhanced", "invalid_main"])
def test_real_text_converter_failed_run_exits_nonzero_and_invalidates_outputs(tmp_path, failure):
    cache = tmp_path / "selected"
    data = _prepare(cache)
    gpu = cache / ".cache" / "gpu"
    gpu.mkdir(parents=True)
    incoming = gpu / "text2qa_step_step3.json"
    if failure == "empty":
        incoming.write_text("[]", encoding="utf-8")
    elif failure == "missing_enhanced":
        incoming.write_text('[{"enhanced_chunk_path":"absent.json"}]', encoding="utf-8")
    else:
        incoming.write_text('{"not":"a list"}', encoding="utf-8")
    result = subprocess.run([sys.executable, str(CONVERTER), "--cache", str(cache)],
                            capture_output=True, text=True, timeout=40)
    assert result.returncode != 0, result.stdout + result.stderr
    assert not (data / "qa.json").exists()
    assert not (data / "dataset_info.json").exists()


def test_real_text_converter_success_overwrites_both_training_artifacts(tmp_path):
    cache = tmp_path / "selected"
    data = _prepare(cache)
    gpu = cache / ".cache" / "gpu"
    gpu.mkdir(parents=True)
    enhanced = gpu / "enhanced.json"
    enhanced.write_text(json.dumps({"qa_pairs": [{"question": "New question", "answer": "New answer"}]}), encoding="utf-8")
    (gpu / "text2qa_step_step3.json").write_text(json.dumps([{"enhanced_chunk_path": str(enhanced)}]), encoding="utf-8")
    result = subprocess.run([sys.executable, str(CONVERTER), "--cache", str(cache)],
                            capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((data / "qa.json").read_text())[0]["input"] == "New question"
    assert json.loads((data / "dataset_info.json").read_text())["kb_qa"]["file_name"] == "qa.json"


@pytest.mark.parametrize("selected", [False, True])
def test_text_cli_noop_generation_never_consumes_stale_stage_or_training_data(tmp_path, monkeypatch, selected):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cache = tmp_path / "selected" if selected else workspace
    data = _prepare(cache)
    other = _prepare(workspace) if selected else None
    gpu = cache / ".cache" / "gpu"
    gpu.mkdir(parents=True)
    (gpu / "text2qa_step_step3.json").write_text('["old"]', encoding="utf-8")
    (gpu / "text_input.jsonl").write_text('{"text":"current"}\n', encoding="utf-8")
    cfg = cache / ".cache" / "train_config.yaml"
    cfg.write_text("dataset: kb_qa\n", encoding="utf-8")
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(cli_text, "verify_environment", lambda: True)
    called = []

    def run(script, description, args=None, cwd=None):
        called.append(description)
        return True  # generator exits normally but does not produce any QA

    monkeypatch.setattr(cli_text, "run_script_with_args", run)
    assert not cli_text.cli_text2model_train(input_dir=str(workspace), cache_path=str(cache))
    assert "Model training" not in called
    assert not (data / "qa.json").exists()
    assert not (data / "dataset_info.json").exists()
    assert not (gpu / "text2qa_step_step3.json").exists()
    if other is not None:
        assert (other / "qa.json").exists()
        assert (other / "dataset_info.json").exists()


@pytest.mark.parametrize("selected", [False, True])
def test_pdf_cli_noop_generation_never_trains_on_stale_qa(tmp_path, monkeypatch, selected):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cache = tmp_path / "selected" if selected else workspace
    data = _prepare(cache)
    other = _prepare(workspace) if selected else None
    (cache / ".cache" / "train_config.yaml").write_text("dataset: pdf_kbc_dataset\n", encoding="utf-8")
    (cache / ".cache" / "pdf2model_state.json").write_text('{"qa":"kbc"}', encoding="utf-8")
    cfg = workspace / ".cache" / "train_config.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("dataset: pdf_kbc_dataset\n", encoding="utf-8")
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(cli_pdf, "verify_environment", lambda: True)
    monkeypatch.setattr(cli_pdf, "check_required_files", lambda: True)
    monkeypatch.setattr(cli_pdf, "get_dataflow_script_path", lambda name: Path(name))
    called = []

    def run(script, description, args=None, cwd=None):
        called.append(description)
        return True  # no-op data processor

    monkeypatch.setattr(cli_pdf, "run_script_with_args", run)
    assert not cli_pdf.cli_pdf2model_train(lf_yaml=str(cfg), cache_path=str(cache))
    assert not any("Training" in x for x in called)
    assert not (data / "qa.json").exists()
    if other is not None:
        assert (other / "qa.json").exists()


@pytest.mark.parametrize("selected", [False, True])
def test_pdf_cli_valid_refresh_allows_reusing_selected_cache(tmp_path, monkeypatch, selected):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cache = tmp_path / "selected" if selected else workspace
    data = _prepare(cache)
    (cache / ".cache" / "train_config.yaml").write_text("dataset: pdf_kbc_dataset\n", encoding="utf-8")
    (cache / ".cache" / "pdf2model_state.json").write_text('{"qa":"kbc"}', encoding="utf-8")
    config = workspace / ".cache" / "train_config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("dataset: pdf_kbc_dataset\n", encoding="utf-8")
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(cli_pdf, "verify_environment", lambda: True)
    monkeypatch.setattr(cli_pdf, "check_required_files", lambda: True)
    monkeypatch.setattr(cli_pdf, "get_dataflow_script_path", lambda name: Path(name))
    trained = []

    def run(script, description, args=None, cwd=None):
        if description == "Step 2: Data Processing":
            (data / "qa.json").write_text(json.dumps([dict(ALPACA[0], input="Current question")]), encoding="utf-8")
        if "Training" in description:
            trained.append(json.loads((data / "qa.json").read_text())[0]["input"])
        return True

    monkeypatch.setattr(cli_pdf, "run_script_with_args", run)
    assert cli_pdf.cli_pdf2model_train(lf_yaml=str(config), cache_path=str(cache))
    assert trained == ["Current question"]
    assert json.loads((data / "dataset_info.json").read_text())["pdf_kbc_dataset"]["file_name"] == "qa.json"
