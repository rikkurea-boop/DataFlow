from pathlib import Path

import pytest
from typer.testing import CliRunner

from dataflow.cli import app
from dataflow.cli_funcs import cli_text


runner = CliRunner()


def _write_config(tmp_path: Path) -> Path:
    config = tmp_path / "train.yaml"
    config.write_text("model_name_or_path: stub\n", encoding="utf-8")
    return config


@pytest.mark.cpu
def test_public_text2model_train_crosses_helper_boundary(monkeypatch, tmp_path):
    """The public CLI must enter the helper instead of failing at kwarg binding."""
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    def entered_helper():
        raise RuntimeError("P08_HELPER_ENTERED")

    monkeypatch.setattr(cli_text, "verify_environment", entered_helper)

    result = runner.invoke(app, ["text2model", "train", str(input_dir)])

    assert result.exit_code == 1
    assert "P08_HELPER_ENTERED" in result.output
    assert "unexpected keyword argument 'input_dir'" not in result.output


@pytest.mark.cpu
def test_text2model_train_uses_custom_input_dir(monkeypatch, tmp_path, capsys):
    """A custom public input directory must reach the helper's merge boundary."""
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "custom-input"
    input_dir.mkdir()
    _write_config(tmp_path)

    monkeypatch.setattr(cli_text, "verify_environment", lambda: True)
    monkeypatch.setattr(
        cli_text,
        "get_dataflow_script_path",
        lambda name: Path("/controlled") / name,
    )

    calls = []

    def fake_run_script(script_path, description, args=None, cwd=None):
        calls.append((script_path, description, args, cwd))
        # Stop at the controlled boundary before the independently tracked
        # missing merge_json_jsonl.py candidate can affect this regression.
        return False

    monkeypatch.setattr(cli_text, "run_script_with_args", fake_run_script)

    ok = cli_text.cli_text2model_train(
        input_keys="text",
        lf_yaml="train.yaml",
        input_dir=str(input_dir),
    )

    assert ok is False
    assert calls
    assert calls[0][2] == [str(input_dir), "--cache", str(tmp_path)]
    assert calls[0][3] == str(tmp_path)
    assert "Processing fields: text" in capsys.readouterr().out


@pytest.mark.cpu
def test_text2model_train_nonexistent_input_reaches_validation(monkeypatch, tmp_path):
    """A missing custom directory should return False at path validation."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path)
    missing = tmp_path / "does-not-exist"

    monkeypatch.setattr(cli_text, "verify_environment", lambda: True)

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("pipeline boundary must not run for a missing input directory")

    monkeypatch.setattr(cli_text, "run_script_with_args", should_not_run)

    ok = cli_text.cli_text2model_train(
        lf_yaml="train.yaml",
        input_dir=str(missing),
    )

    assert ok is False


@pytest.mark.cpu
def test_text2model_train_default_input_and_positional_compatibility(monkeypatch, tmp_path):
    """The new optional parameter must preserve the old two-argument helper call."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path)

    monkeypatch.setattr(cli_text, "verify_environment", lambda: True)
    monkeypatch.setattr(
        cli_text,
        "get_dataflow_script_path",
        lambda name: Path("/controlled") / name,
    )

    calls = []

    def fake_run_script(script_path, description, args=None, cwd=None):
        calls.append((script_path, description, args, cwd))
        return False

    monkeypatch.setattr(cli_text, "run_script_with_args", fake_run_script)

    # Preserve compatibility with existing callers that pass only the original
    # two positional parameters: input_keys and lf_yaml.
    ok = cli_text.cli_text2model_train(None, "train.yaml")

    assert ok is False
    assert calls
    assert calls[0][2] == [str(tmp_path), "--cache", str(tmp_path)]
    assert calls[0][3] == str(tmp_path)
