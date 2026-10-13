import pandas as pd
import pytest

from dataflow.core import OperatorABC
from dataflow.pipeline import BatchedPipelineABC
from dataflow.utils.storage import BatchedFileStorage


class _VersionedProducer(OperatorABC):
    def __init__(self, version: str):
        super().__init__()
        self.version = version

    def run(self, storage):
        dataframe = storage.read("dataframe").copy()
        dataframe["version"] = self.version
        storage.write(dataframe)


class _ObservingConsumer(OperatorABC):
    def __init__(self):
        super().__init__()
        self.seen_versions = []

    def run(self, storage):
        dataframe = storage.read("dataframe").copy()
        self.seen_versions.append(dataframe["version"].tolist())
        storage.write(dataframe)


class _RerunnableBatchedPipeline(BatchedPipelineABC):
    def __init__(self, source, cache_path, producer, consumer):
        super().__init__()
        self.storage = BatchedFileStorage(
            first_entry_file_name=str(source),
            cache_path=str(cache_path),
            file_name_prefix="p07_02",
            cache_type="jsonl",
        )
        self.producer = producer
        self.consumer = consumer

    def forward(self):
        self.producer.run(self.storage.step())
        self.consumer.run(self.storage.step())


def _source(tmp_path):
    source = tmp_path / "input.jsonl"
    pd.DataFrame([{"seed": "row"}]).to_json(
        source, orient="records", lines=True, force_ascii=False
    )
    return source


@pytest.mark.cpu
def test_public_step_rewrite_invalidates_prior_read_view(tmp_path):
    """A public rewrite of step 1 must be visible to an existing step-1 reader."""
    source = _source(tmp_path)
    root = BatchedFileStorage(
        first_entry_file_name=str(source),
        cache_path=str(tmp_path / "cache-public"),
        file_name_prefix="public_transition",
        cache_type="jsonl",
    )

    producer_storage = root.step()  # step 0 writes step 1
    consumer_storage = root.step()  # step 1 reads step 1

    producer_storage.write([{"value": "old"}])
    assert consumer_storage.read("dict") == [{"value": "old"}]

    # Re-running the upstream public step overwrites the persisted step-1 artifact.
    producer_storage.write([{"value": "new"}])
    persisted = pd.read_json(
        tmp_path / "cache-public" / "public_transition_step1.jsonl", lines=True
    )
    assert persisted.to_dict("records") == [{"value": "new"}]

    # The already-created downstream storage view must follow the current artifact.
    assert consumer_storage.read("dict") == [{"value": "new"}]


@pytest.mark.cpu
def test_batched_pipeline_rerun_without_resume_consumes_current_upstream_output(tmp_path):
    """resume_from_last=False is an overwrite/rerun contract, not a stale replay."""
    source = _source(tmp_path)
    cache_path = tmp_path / "cache-pipeline"
    producer = _VersionedProducer("first")
    consumer = _ObservingConsumer()
    pipeline = _RerunnableBatchedPipeline(
        source=source,
        cache_path=cache_path,
        producer=producer,
        consumer=consumer,
    )

    pipeline.compile()
    pipeline.forward(batch_size=1, resume_from_last=False)
    assert consumer.seen_versions[-1] == ["first"]

    # Same compiled pipeline, explicitly rerun from the beginning. The public
    # producer configuration changes, so step 1 is overwritten with new data.
    producer.version = "second"
    pipeline.forward(batch_size=1, resume_from_last=False)

    persisted_step1 = pd.read_json(cache_path / "p07_02_step1.jsonl", lines=True)
    assert persisted_step1["version"].tolist() == ["second"]

    # Downstream must observe the just-written current run, not its prior read cache.
    assert consumer.seen_versions[-1] == ["second"]
