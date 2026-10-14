"""Independent manager RED regressions for P11.04/P11.05 candidate.

These tests are intentionally added only to an isolated validation branch.
"""
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.cpu

SETUP = r"""
import importlib
import sys
import tempfile
from pathlib import Path

import dataflow
r = importlib.import_module("dataflow.utils.registry")
tmp = tempfile.TemporaryDirectory()
root = Path(tmp.name) / "dataflow"
dataflow.__path__.append(str(root))

def make_registry(name, paths):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "__init__.py").write_text("", encoding="utf-8")
    for filename, code in paths.items():
        target = directory / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code, encoding="utf-8")
    original_listdir = r.os.listdir
    def listdir(path):
        if Path(path).name == name:
            return original_listdir(directory)
        return original_listdir(path)
    r.os.listdir = listdir
    try:
        instance = r.Registry(name)
    finally:
        r.os.listdir = original_listdir
    r._manager_review_registry = instance
    return instance
"""

def fresh(script):
    proc = subprocess.run(
        [sys.executable, "-c", SETUP + "\n" + textwrap.dedent(script)],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, (
        f"Manager RED regression failed with exit {proc.returncode}\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

def test_prior_module_name_only_in_comment_must_not_break_valid_registered_lookup():
    fresh(r"""
        reg = make_registry("p11_manager_falsepositive", {
            "a.py": "# Shared is mentioned in documentation only\nraise RuntimeError('unrelated import exploded')\n",
            "z.py": (
                "from dataflow.utils import registry as r\n"
                "@r._manager_review_registry.register()\n"
                "class Shared: pass\n"
            ),
        })
        importlib.import_module("dataflow.p11_manager_falsepositive.z")
        value = reg.get("Shared")
        assert value.__module__ == "dataflow.p11_manager_falsepositive.z"
        assert "dataflow.p11_manager_falsepositive.a" not in sys.modules
    """)

def test_duplicate_resolution_after_top_level_loader_initialization():
    fresh(r"""
        source = (
            "from dataflow.utils import registry as r\n"
            "@r._manager_review_registry.register()\n"
            "class Shared: pass\n"
        )
        reg = make_registry("p11_manager_loaded", {
            "a/__init__.py": "",
            "a/provider.py": source,
            "z/__init__.py": "",
            "z/provider.py": source,
        })
        reg._init_loaders()
        assert None not in reg.loader_map.values()
        assert "dataflow.p11_manager_loaded.a.provider" not in sys.modules
        importlib.import_module("dataflow.p11_manager_loaded.z.provider")
        value = reg.get("Shared")
        assert value.__module__ == "dataflow.p11_manager_loaded.a.provider", value.__module__
    """)

def test_unregistered_class_and_string_mentions_do_not_trigger_import():
    fresh(r"""
        reg = make_registry("p11_manager_notregistered", {
            "a.py": (
                'Documentation: Shared is an example.\n'
                "class Shared: pass\n"
                "raise RuntimeError('unrelated code executed')\n"
            ),
            "z.py": (
                "from dataflow.utils import registry as r\n"
                "@r._manager_review_registry.register()\n"
                "class Shared: pass\n"
            ),
        })
        importlib.import_module("dataflow.p11_manager_notregistered.z")
        assert reg.get("Shared").__module__ == "dataflow.p11_manager_notregistered.z"
        assert "dataflow.p11_manager_notregistered.a" not in sys.modules
    """)


def test_explicit_register_call_is_a_candidate_without_eager_unrelated_import():
    fresh(r"""
        reg = make_registry("p11_manager_direct_register", {
            "a.py": (
                "from dataflow.utils import registry as r\n"
                "class Shared: pass\n"
                "r._manager_review_registry.register(Shared)\n"
            ),
            "b.py": "raise RuntimeError('not a candidate')\n",
            "z.py": (
                "from dataflow.utils import registry as r\n"
                "@r._manager_review_registry.register()\n"
                "class Shared: pass\n"
            ),
        })
        importlib.import_module("dataflow.p11_manager_direct_register.z")
        assert reg.get("Shared").__module__ == "dataflow.p11_manager_direct_register.a"
        assert "dataflow.p11_manager_direct_register.b" not in sys.modules
    """)
