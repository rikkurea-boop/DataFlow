import sys
import types

import pandas as pd
import pytest

from dataflow.operators.reasoning.generate.reasoning_pretrain_format_convert_generator import (
    ReasoningPretrainFormatConvertGenerator,
)
from dataflow.utils.storage import BatchedFileStorage


pytestmark = pytest.mark.cpu


def test_zero_row_remote_dataset_can_cross_real_batched_write_boundary(
    monkeypatch, tmp_path
):
    """A valid zero-row dataset should remain representable through a public operator."""

    class _Dataset:
        def to_pandas(self):
            return pd.DataFrame(columns=["question", "answer"])

    calls = []

    def load_dataset(dataset_name, config=None, *, split=None):
        calls.append((dataset_name, config, split))
        return _Dataset()

    datasets_module = types.ModuleType("datasets")
    datasets_module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)

    storage = BatchedFileStorage(
        "hf:example/empty:main:train",
        cache_path=str(tmp_path),
        file_name_prefix="p07_04",
        cache_type="jsonl",
    ).step()
    storage.batch_size = 8

    # Existing public storage semantics explicitly admit a zero-row remote dataset.
    assert storage.read().empty
    assert storage.record_count == 0

    operator = ReasoningPretrainFormatConvertGenerator()
    operator.run(
        storage=storage,
        input_read_key_question="question",
        input_read_key_answer="answer",
        output_key="text",
    )

    output_path = tmp_path / "p07_04_step1.jsonl"
    assert output_path.is_file()
    assert output_path.read_text(encoding="utf-8") == ""
    assert calls == [("example/empty", "main", "train")]
