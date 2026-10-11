from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from typer.testing import CliRunner

from dataflow.cli import app
from dataflow.cli_funcs import cli_text
from dataflow.cli_funcs.text2model_pipeline.llama_factory_trainer import LlamaFactoryTrainer


runner = CliRunner()


def _write_config(cache: Path) -> Path:
    config = cache / ".cache" / "train_config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("model_name_or_path: stub\n", encoding="utf-8")
    return config


@pytest.mark.cpu
def test_public_text2model_train_passes_input_and_cache(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    input_dir = workspace / "input"
    cache = tmp_path / "cache"
    workspace.mkdir()
    input_dir.mkdir()
    cache.mkdir()
    monkeypatch.chdir(workspace)

    captured = {}

    def fake_train(**kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(cli_text, "cli_text2model_train", fake_train)

    result = runner.invoke(
        app,
        ["text2model", "train", str(input_dir), "--cache", str(cache)],
    )

    assert result.exit_code == 0, result.output
    assert captured["input_dir"] == str(input_dir)
    assert captured["cache_path"] == str(cache)
    assert captured["lf_yaml"] == f"{cache}/.cache/train_config.yaml"


@pytest.mark.cpu
def test_public_text2model_train_preserves_explicit_lf_yaml(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    input_dir = workspace / "input"
    cache = tmp_path / "cache"
    custom_yaml = tmp_path / "configs" / "custom.yaml"
    workspace.mkdir()
    input_dir.mkdir()
    cache.mkdir()
    custom_yaml.parent.mkdir()
    custom_yaml.write_text("model_name_or_path: stub\n", encoding="utf-8")
    monkeypatch.chdir(workspace)

    captured = {}

    def fake_train(**kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(cli_text, "cli_text2model_train", fake_train)

    result = runner.invoke(
        app,
        [
            "text2model",
            "train",
            str(input_dir),
            "--cache",
            str(cache),
            "--lf-yaml",
            str(custom_yaml),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["lf_yaml"] == str(custom_yaml)
    assert captured["cache_path"] == str(cache)


@pytest.mark.cpu
def test_text2model_helper_uses_custom_input_and_cache_without_moving_workspace_script(
    monkeypatch, tmp_path
):
    workspace = tmp_path / "workspace"
    input_dir = workspace / "input"
    cache = tmp_path / "cache"
    workspace.mkdir()
    input_dir.mkdir()
    cache.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("PWD", str(workspace))

    config = _write_config(cache)
    customizable_script = workspace / "text_to_qa_pipeline.py"
    customizable_script.write_text("# controlled\n", encoding="utf-8")

    monkeypatch.setattr(cli_text, "verify_environment", lambda: True)
    monkeypatch.setattr(
        cli_text,
        "get_dataflow_script_path",
        lambda name: Path("/controlled") / name,
    )

    calls = []

    def fake_run_script(script_path, description, args=None, cwd=None):
        calls.append((Path(script_path), description, list(args or []), cwd))
        if description == "JSON/JSONL merging":
            out = cache / ".cache" / "gpu" / "text_input.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text('{"text":"current"}\n', encoding="utf-8")
            return True
        if description == "Text2QA generation":
            out = cache / ".cache" / "gpu" / "text2qa_step_step3.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("[]", encoding="utf-8")
            return True
        if description == "QA format conversion":
            data_dir = cache / ".cache" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            (data_dir / "qa.json").write_text(
                json.dumps([{"instruction": "i", "input": "q", "output": "a"}]),
                encoding="utf-8",
            )
            (data_dir / "dataset_info.json").write_text("{}", encoding="utf-8")
            return True
        if description == "Model training":
            return False
        raise AssertionError(description)

    monkeypatch.setattr(cli_text, "run_script_with_args", fake_run_script)

    ok = cli_text.cli_text2model_train(
        input_keys="text",
        lf_yaml=str(config),
        input_dir=str(input_dir),
        cache_path=str(cache),
    )

    assert ok is False
    assert [call[1] for call in calls] == [
        "JSON/JSONL merging",
        "Text2QA generation",
        "QA format conversion",
        "Model training",
    ]
    assert calls[0][2] == [str(input_dir), "--cache", str(cache)]
    assert calls[1][0] == customizable_script
    assert calls[1][2] == ["--cache", str(cache)]
    assert calls[2][2] == ["--cache", str(cache)]
    assert calls[3][2] == ["--config", str(config), "--cache", str(cache)]
    assert all(call[3] == str(workspace) for call in calls)


@pytest.mark.cpu
def test_text2model_helper_default_paths_and_old_positional_call_remain_compatible(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PWD", str(tmp_path))
    config = tmp_path / "train.yaml"
    config.write_text("model_name_or_path: stub\n", encoding="utf-8")

    monkeypatch.setattr(cli_text, "verify_environment", lambda: True)
    monkeypatch.setattr(
        cli_text,
        "get_dataflow_script_path",
        lambda name: Path("/controlled") / name,
    )
    calls = []

    def stop_at_merge(script_path, description, args=None, cwd=None):
        calls.append((description, list(args or []), cwd))
        return False

    monkeypatch.setattr(cli_text, "run_script_with_args", stop_at_merge)

    # Preserve compatibility with the original two positional helper arguments.
    ok = cli_text.cli_text2model_train(None, "train.yaml")

    assert ok is False
    assert calls[0][1] == [str(tmp_path), "--cache", str(tmp_path)]
    assert calls[0][2] == str(tmp_path)


@pytest.mark.cpu
def test_text2model_helper_missing_custom_input_reaches_path_validation(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    cache = tmp_path / "cache"
    missing = workspace / "missing"
    workspace.mkdir()
    cache.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("PWD", str(workspace))
    config = _write_config(cache)

    monkeypatch.setattr(cli_text, "verify_environment", lambda: True)

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("pipeline must not run for a missing input directory")

    monkeypatch.setattr(cli_text, "run_script_with_args", should_not_run)

    ok = cli_text.cli_text2model_train(
        lf_yaml=str(config),
        input_dir=str(missing),
        cache_path=str(cache),
    )

    assert ok is False


@pytest.mark.cpu
def test_trainer_preflight_reads_selected_cache_not_workspace(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    cache = tmp_path / "cache"
    workspace.mkdir()
    data_dir = cache / ".cache" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "qa.json").write_text(
        json.dumps([{"instruction": "i", "input": "q", "output": "a"}]),
        encoding="utf-8",
    )
    (data_dir / "dataset_info.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("PWD", str(workspace))
    monkeypatch.setitem(sys.modules, "llamafactory", ModuleType("llamafactory"))

    trainer = LlamaFactoryTrainer(
        config_path=str(cache / ".cache" / "train_config.yaml"),
        cache_base=str(cache),
    )

    assert trainer.check_environment() is True


@pytest.mark.cpu
def test_text2qa_template_routes_qaextractor_output_through_selected_cache():
    source_path = (
        Path(__file__).resolve().parents[2]
        / "dataflow"
        / "cli_funcs"
        / "text2model_pipeline"
        / "text_to_qa_pipeline.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    output_expr = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = getattr(node.func, "id", None)
        if func_name != "QAExtractor":
            continue
        for keyword in node.keywords:
            if keyword.arg == "output_json_file":
                output_expr = keyword.value
                break

    assert output_expr is not None
    assert any(isinstance(node, ast.Name) and node.id == "cache_path" for node in ast.walk(output_expr))
