from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "dataflow" / "serving" / "local_model_vlm_serving.py"


def _source():
    return SOURCE.read_text(encoding="utf-8")


@pytest.mark.cpu
def test_generic_local_vlm_entrypoint_forwards_json_schema():
    source = _source()
    start = source.index("def generate_from_input(self,")
    block = source[start:]
    assert "json_schema=json_schema" in block


@pytest.mark.cpu
def test_single_image_generation_applies_guided_json_without_mutating_default_params():
    source = _source()
    start = source.index("def generate_from_input_one_image(")
    end = source.index("def generate_from_input_multi_images(", start)
    block = source[start:end]
    assert "GuidedDecodingParams" in block
    assert "json=json_schema" in block
    assert "sampling_params = self.sampling_params" in block
    assert "self.llm.generate(inputs, sampling_params=sampling_params)" in block
