from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "dataflow" / "serving" / "lite_llm_serving.py"


@pytest.mark.cpu
def test_litellm_docs_match_explicit_switch_model_mutability():
    source = SOURCE.read_text(encoding="utf-8")
    assert "Configuration parameters are immutable after initialization" not in source
    assert "All configuration parameters are immutable after initialization" not in source
    assert "switch_model" in source
    assert "instance-local" in source
