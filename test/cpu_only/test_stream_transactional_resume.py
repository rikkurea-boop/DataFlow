"""P07.01 — production-boundary tests for opt-in stream recovery."""
import os
import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from dataflow.core import OperatorABC
from dataflow.pipeline import StreamBatchedPipelineABC
from dataflow.utils.storage import StreamBatchedFileStorage
from dataflow.utils.stream_transaction import TransactionError


class _Pass(OperatorABC):
    transaction_safe = True

    def __init__(self, fail=None, output_empty=False, fail_commit=None):
        super().__init__()
        self.fail = fail
        self.output_empty = output_empty
        self.fail_commit = fail_commit
        self.failed = False

    def transaction_signature(self):
        return {"empty": self.output_empty}

    def run(self, storage):
        data = storage.read("dataframe")
        records = data["id"].tolist() if "id" in data else []
        if self.fail == "before" and records == [2, 3] and not self.failed:
            self.failed = True
            raise RuntimeError("before write")
        storage.write([] if self.output_empty else data)
        if self.fail == "after" and records == [2, 3] and not self.failed:
            self.failed = True
            raise RuntimeError("after write")


class _Double(OperatorABC):
    transaction_safe = True

    def transaction_signature(self):
        return {}

    def run(self, storage):
        data = storage.read("dataframe")
        storage.write(data)
        try:
            storage.write(data)
        except TransactionError:
            pass  # Suppressed write errors must still abort the transaction.


class _Pipeline(StreamBatchedPipelineABC):
    def __init__(self, source, cache, fmt, ops):
        super().__init__()
        self.storage = StreamBatchedFileStorage(str(source), str(cache), "p07_01", fmt)
        for i, operator in enumerate(ops):
            setattr(self, "op" + str(i), operator)

    def forward(self):
        for i in range(len([x for x in vars(self) if x.startswith("op") and x[2:].isdigit()])):
            getattr(self, "op" + str(i)).run(self.storage.step())


def _source(tmp_path, fmt, rows=5):
    source = tmp_path / ("source." + fmt)
    data = pd.DataFrame({"id": list(range(rows))})
    if fmt == "csv":
        data.to_csv(source, index=False)
    else:
        data.to_json(source, orient="records", lines=True)
    return source


def _output(path, fmt):
    if not path.exists() or path.stat().st_size == 0:
        return []
    if fmt == "csv":
        return pd.read_csv(path)["id"].tolist()
    return pd.read_json(path, lines=True)["id"].tolist()


def _run(source, cache, fmt, ops):
    p = _Pipeline(source, cache, fmt, ops)
    p.compile()
    p.forward(batch_size=2, transactional_resume=True)


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
@pytest.mark.parametrize("fail", ["before", "after"])
def test_transaction_replay_on_same_instance(tmp_path, fmt, fail):
    source = _source(tmp_path, fmt)
    cache = tmp_path / "cache"
    op = _Pass(fail=fail)
    pipeline = _Pipeline(source, cache, fmt, [op])
    pipeline.compile()
    with pytest.raises(RuntimeError, match=fail + " write"):
        pipeline.forward(batch_size=2, transactional_resume=True)
    pipeline.forward(batch_size=2, transactional_resume=True)
    assert _output(cache / ("p07_01_step1." + fmt), fmt) == [0, 1, 2, 3, 4]


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_transaction_replay_after_new_pipeline(tmp_path, fmt):
    source = _source(tmp_path, fmt)
    cache = tmp_path / "cache"
    with pytest.raises(RuntimeError):
        _run(source, cache, fmt, [_Pass(fail="after")])
    _run(source, cache, fmt, [_Pass()])
    assert _output(cache / ("p07_01_step1." + fmt), fmt) == [0, 1, 2, 3, 4]


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_two_stages_and_stage_resume(tmp_path, fmt):
    source = _source(tmp_path, fmt)
    cache = tmp_path / "cache"
    with pytest.raises(RuntimeError):
        _run(source, cache, fmt, [_Pass(), _Pass(fail="after")])
    _run(source, cache, fmt, [_Pass(), _Pass()])
    for n in (1, 2):
        assert _output(cache / (f"p07_01_step{n}." + fmt), fmt) == [0, 1, 2, 3, 4]


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_logical_empty_stage(tmp_path, fmt):
    source = _source(tmp_path, fmt)
    cache = tmp_path / "cache"
    _run(source, cache, fmt, [_Pass(output_empty=True), _Pass()])
    assert _output(cache / ("p07_01_step1." + fmt), fmt) == []
    assert not (cache / ("p07_01_step2." + fmt)).exists()


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_multiple_writes_even_if_caught_fails(tmp_path, fmt):
    source = _source(tmp_path, fmt)
    with pytest.raises(TransactionError, match="one successful write"):
        _run(source, tmp_path / "cache", fmt, [_Double()])


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_unowned_outputs_fail_closed(tmp_path, fmt):
    source = _source(tmp_path, fmt)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / ("p07_01_step1." + fmt)).write_bytes(b"stale")
    with pytest.raises(TransactionError, match="unowned"):
        _run(source, cache, fmt, [_Pass()])


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_missing_committed_output_fails_closed(tmp_path, fmt):
    source = _source(tmp_path, fmt)
    cache = tmp_path / "cache"
    _run(source, cache, fmt, [_Pass()])
    (cache / ("p07_01_step1." + fmt)).unlink()
    with pytest.raises(TransactionError, match="missing"):
        _run(source, cache, fmt, [_Pass()])


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_transaction_option_off_keeps_legacy(tmp_path, fmt):
    source = _source(tmp_path, fmt)
    cache = tmp_path / "cache"
    p = _Pipeline(source, cache, fmt, [_Pass()])
    p.compile()
    p.forward(batch_size=2, resume_from_last=True)
    assert (cache / "p07_01_last_success_step.txt").exists()
    assert not (cache / "p07_01_stream_transaction.json").exists()


@pytest.mark.cpu
@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_force_killed_process_rolls_back_persisted_batch(tmp_path, fmt):
    source = _source(tmp_path, fmt)
    cache = tmp_path / "cache"
    # Exercise real production pipeline/storage; os._exit cannot run normal
    # exception handling, so recovery occurs in a fresh parent-side instance.
    script = textwrap.dedent("""
        import importlib.util
        import os
        import sys

        spec = importlib.util.spec_from_file_location("test_stream_transactional_resume", sys.argv[4])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        operator = mod._Pass()
        def crash_after_write(storage):
            data = storage.read("dataframe")
            storage.write(data)
            if data["id"].tolist() == [2, 3]:
                os._exit(79)
        operator.run = crash_after_write
        pipeline = mod._Pipeline(sys.argv[1], sys.argv[2], sys.argv[3], [operator])
        pipeline.compile()
        pipeline.forward(batch_size=2, transactional_resume=True)
    """)
    result = subprocess.run([sys.executable, "-c", script, str(source), str(cache), fmt, __file__],
                            capture_output=True, text=True)
    assert result.returncode == 79, result.stderr
    _run(source, cache, fmt, [_Pass()])
    assert _output(cache / ("p07_01_step1." + fmt), fmt) == [0, 1, 2, 3, 4]
