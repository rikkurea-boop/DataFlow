import pandas as pd
import pytest

from dataflow.core import OperatorABC
from dataflow.pipeline import StreamBatchedPipelineABC
from dataflow.utils.storage import StreamBatchedFileStorage


def _write_source(tmp_path, cache_type):
    source = tmp_path / f"source.{cache_type}"
    dataframe = pd.DataFrame({"record_id": [0, 1, 2, 3, 4]})
    if cache_type == "jsonl":
        dataframe.to_json(source, orient="records", lines=True, force_ascii=False)
    else:
        dataframe.to_csv(source, index=False)
    return source


def _read_records(path, cache_type):
    if cache_type == "jsonl":
        return pd.read_json(path, lines=True)["record_id"].tolist()
    return pd.read_csv(path)["record_id"].tolist()


class _InterruptAfterWriteOnce(OperatorABC):
    """Deterministically interrupt after a real storage write for batch [2, 3]."""

    def __init__(self):
        super().__init__()
        self.interrupted = False
        self.delivered_batches = []

    def run(self, storage):
        dataframe = storage.read("dataframe").copy()
        record_ids = dataframe["record_id"].tolist()
        self.delivered_batches.append(record_ids)
        storage.write(dataframe)
        if record_ids == [2, 3] and not self.interrupted:
            self.interrupted = True
            raise RuntimeError("intentional interruption after persisted write")


class _InterruptBeforeWriteOnce(OperatorABC):
    """Control: interruption before write should replay without duplicating output."""

    def __init__(self):
        super().__init__()
        self.interrupted = False
        self.delivered_batches = []

    def run(self, storage):
        dataframe = storage.read("dataframe").copy()
        record_ids = dataframe["record_id"].tolist()
        self.delivered_batches.append(record_ids)
        if record_ids == [2, 3] and not self.interrupted:
            self.interrupted = True
            raise RuntimeError("intentional interruption before persisted write")
        storage.write(dataframe)


class _StreamRetryPipeline(StreamBatchedPipelineABC):
    def __init__(self, source, cache_path, cache_type, operator, prefix):
        super().__init__()
        self.storage = StreamBatchedFileStorage(
            first_entry_file_name=str(source),
            cache_path=str(cache_path),
            file_name_prefix=prefix,
            cache_type=cache_type,
        )
        self.operator = operator

    def forward(self):
        self.operator.run(self.storage.step())


@pytest.mark.cpu
@pytest.mark.parametrize("cache_type", ["jsonl", "csv"])
def test_stream_resume_after_prewrite_interruption_replays_batch_once(tmp_path, cache_type):
    source = _write_source(tmp_path, cache_type)
    cache_path = tmp_path / "cache-prewrite"
    operator = _InterruptBeforeWriteOnce()
    pipeline = _StreamRetryPipeline(
        source, cache_path, cache_type, operator, "p07_01_prewrite"
    )

    pipeline.compile()
    with pytest.raises(RuntimeError, match="interruption before persisted write"):
        pipeline.forward(batch_size=2, resume_from_last=True)

    output_path = cache_path / f"p07_01_prewrite_step1.{cache_type}"
    checkpoint_path = cache_path / "p07_01_prewrite_last_success_step.txt"
    assert _read_records(output_path, cache_type) == [0, 1]
    assert checkpoint_path.read_text().strip() == "0,1"

    pipeline.forward(batch_size=2, resume_from_last=True)

    assert operator.delivered_batches == [[0, 1], [2, 3], [2, 3], [4]]
    assert _read_records(output_path, cache_type) == [0, 1, 2, 3, 4]


@pytest.mark.cpu
@pytest.mark.parametrize("cache_type", ["jsonl", "csv"])
def test_stream_resume_does_not_duplicate_batch_persisted_before_interruption(
    tmp_path, cache_type
):
    source = _write_source(tmp_path, cache_type)
    cache_path = tmp_path / "cache-postwrite"
    operator = _InterruptAfterWriteOnce()
    pipeline = _StreamRetryPipeline(
        source, cache_path, cache_type, operator, "p07_01_postwrite"
    )

    pipeline.compile()
    with pytest.raises(RuntimeError, match="interruption after persisted write"):
        pipeline.forward(batch_size=2, resume_from_last=True)

    output_path = cache_path / f"p07_01_postwrite_step1.{cache_type}"
    checkpoint_path = cache_path / "p07_01_postwrite_last_success_step.txt"

    # The second batch is already on disk, but the public pipeline checkpoint still
    # points at that batch because the operator did not return successfully.
    assert _read_records(output_path, cache_type) == [0, 1, 2, 3]
    assert checkpoint_path.read_text().strip() == "0,1"

    pipeline.forward(batch_size=2, resume_from_last=True)

    # Retry correctly re-delivers the uncheckpointed batch [2, 3]. Storage/pipeline
    # recovery must nevertheless produce each logical record exactly once.
    assert operator.delivered_batches == [[0, 1], [2, 3], [2, 3], [4]]
    assert _read_records(output_path, cache_type) == [0, 1, 2, 3, 4]
