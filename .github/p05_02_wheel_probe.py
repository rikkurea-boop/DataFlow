from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def _venv_python(venv_root: Path) -> Path:
    if os.name == "nt":
        return venv_root / "Scripts" / "python.exe"
    return venv_root / "bin" / "python"


def _venv_cli(venv_root: Path) -> Path:
    if os.name == "nt":
        return venv_root / "Scripts" / "dataflow.exe"
    return venv_root / "bin" / "dataflow"


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, **kwargs)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: p05_02_wheel_probe.py ARTIFACT_DIR")

    artifact_dir = Path(sys.argv[1]).resolve()
    wheels = sorted(artifact_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise AssertionError(f"expected exactly one wheel, got: {wheels}")
    wheel = wheels[0]

    python_manifest = artifact_dir / "source_python_files.json"
    scaffold_manifest = artifact_dir / "source_scaffold_files.json"
    if not python_manifest.is_file() or not scaffold_manifest.is_file():
        raise AssertionError("source manifests are missing from build artifact")

    uv = shutil.which("uv")
    if uv is None:
        raise AssertionError("uv executable not found")

    with tempfile.TemporaryDirectory(prefix="p05-02-venv-") as tmp:
        venv_root = Path(tmp) / "venv"
        _run([uv, "venv", "--python", sys.executable, str(venv_root)])
        python = _venv_python(venv_root)
        _run([uv, "pip", "install", "--python", str(python), str(wheel)])

        probe = r'''
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

venv_root = Path(os.environ["P05_VENV_ROOT"]).resolve()
python_manifest = Path(os.environ["P05_PYTHON_MANIFEST"])
scaffold_manifest = Path(os.environ["P05_SCAFFOLD_MANIFEST"])

# The distribution must be the installed wheel, never a source checkout.
dist = importlib.metadata.distribution("open-dataflow")
files = {str(path).replace("\\\\", "/") for path in (dist.files or [])}
expected_python = set(json.loads(python_manifest.read_text(encoding="utf-8")))
expected_scaffold = set(json.loads(scaffold_manifest.read_text(encoding="utf-8")))
missing_python = sorted(expected_python - files)
missing_scaffold = sorted(expected_scaffold - files)
assert not missing_python, f"wheel omitted Python files: {missing_python[:50]}"
assert not missing_scaffold, f"wheel omitted scaffold files: {missing_scaffold[:50]}"

modules = [
    "dataflow",
    "dataflow.cli",
    "dataflow.cli_funcs",
    "dataflow.cli_funcs.cli_init",
    "dataflow.cli_funcs.paths",
    "dataflow.prompts.reasoning",
    "dataflow.prompts.reasoning.general",
    "dataflow.prompts.model_evaluation.general",
    "dataflow.operators.reasoning",
]
for name in modules:
    module = importlib.import_module(name)
    module_file = getattr(module, "__file__", None)
    if module_file:
        resolved = Path(module_file).resolve()
        assert resolved == venv_root or venv_root in resolved.parents, (
            f"{name} resolved outside clean wheel environment: {resolved}"
        )

from dataflow.cli_funcs.paths import DataFlowPath
scaffold = DataFlowPath.get_dataflow_scaffold_dir().resolve()
assert scaffold.is_dir(), f"installed scaffold directory missing: {scaffold}"
assert venv_root in scaffold.parents, f"scaffold resolved outside installed wheel: {scaffold}"
for rel in [
    "cookiecutter.json",
    "hooks",
    "{{cookiecutter.repo_name}}/README.md",
    "{{cookiecutter.repo_name}}/pyproject.toml",
    "{{cookiecutter.repo_name}}/{{cookiecutter.package_name}}/__init__.py",
    "{{cookiecutter.repo_name}}/{{cookiecutter.package_name}}/operators/core/my_prompted_generator.py",
    "{{cookiecutter.repo_name}}/{{cookiecutter.package_name}}/prompts/core.py",
    "{{cookiecutter.repo_name}}/{{cookiecutter.package_name}}/pipelines/my_pipeline.py",
    "{{cookiecutter.repo_name}}/{{cookiecutter.package_name}}/data/example_data.json",
]:
    path = scaffold / rel
    assert path.exists(), f"installed scaffold resource missing: {path}"

print(f"installed_distribution={dist.version}")
print(f"installed_scaffold={scaffold}")
print(f"python_files_checked={len(expected_python)}")
print(f"scaffold_files_checked={len(expected_scaffold)}")
'''
        env = os.environ.copy()
        env["P05_VENV_ROOT"] = str(venv_root)
        env["P05_PYTHON_MANIFEST"] = str(python_manifest)
        env["P05_SCAFFOLD_MANIFEST"] = str(scaffold_manifest)
        env["PYTHONUTF8"] = "1"

        with tempfile.TemporaryDirectory(prefix="p05-02-installed-cwd-") as clean_cwd:
            _run([str(python), "-c", probe], cwd=clean_cwd, env=env)

            cli = _venv_cli(venv_root)
            if not cli.is_file():
                raise AssertionError(f"installed dataflow CLI missing: {cli}")
            result = _run(
                [str(cli), "init", "repo", "--no-input"],
                cwd=clean_cwd,
                env=env,
                capture_output=True,
            )
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)

            generated = Path(clean_cwd) / "my-dataflow-extension"
            expected_generated = [
                "README.md",
                "pyproject.toml",
                "my_dataflow_extension/__init__.py",
                "my_dataflow_extension/operators/core/my_prompted_generator.py",
                "my_dataflow_extension/prompts/core.py",
                "my_dataflow_extension/pipelines/my_pipeline.py",
                "my_dataflow_extension/data/example_data.json",
            ]
            missing_generated = [
                rel for rel in expected_generated if not (generated / rel).exists()
            ]
            assert not missing_generated, (
                f"real scaffold command failed to materialize resources: {missing_generated}"
            )
            print(f"generated_scaffold={generated}")


if __name__ == "__main__":
    main()
