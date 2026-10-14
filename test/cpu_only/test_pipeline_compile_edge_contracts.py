"""Additional runtime stress regressions for recoverable Pipeline.compile()."""

from collections import Counter

import pandas as pd
import pytest

from dataflow.core import OperatorABC
from dataflow.pipeline.Pipeline import (
    BatchedPipelineABC,
    PipelineABC,
    StreamBatchedPipelineABC,
)
from dataflow.utils.storage import DataFlowStorage

pytestmark = pytest.mark.cpu

PIPELINES = [PipelineABC, BatchedPipelineABC, StreamBatchedPipelineABC]


class _Store(DataFlowStorage):
    def __init__(self, *, reject_keys=False):
        self.operator_step = 0
        self.reject_keys = reject_keys
        self.dataframe = pd.DataFrame({"value": [2, 3, 4]})

    def step(self):
        self.operator_step += 1
        return self

    def get_keys_from_dataframe(self):
        if self.reject_keys:
            raise LookupError("get-keys failure during graph construction")
        return list(self.dataframe.columns)

    def read(self, output_type="dataframe"):
        assert output_type == "dataframe"
        return self.dataframe.copy()

    def write(self, dataframe):
        self.dataframe = dataframe.copy()


class _Increment(OperatorABC):
    def __init__(self):
        super().__init__()
        self.runs = 0

    def run(self, storage, input_key, output_key):
        self.runs += 1
        df = storage.read()
        df[output_key] = df[input_key] + 1
        storage.write(df)


class _Holder:
    def __init__(self, storage):
        self.storage = storage


def _execute(pipeline, base):
    if base is PipelineABC:
        pipeline.forward()
    else:
        pipeline.forward(resume_from_last=False)


@pytest.mark.parametrize("base", PIPELINES)
def test_failed_forward_restores_two_distinct_nested_storage_cursors(base):
    first, second = _Store(), _Store()

    class Example(base):
        def __init__(self):
            super().__init__()
            self.configuration = {
                "first": [{"item": _Holder(first)}],
                "second": (_Holder(second),),
            }
            self.alpha, self.beta = _Increment(), _Increment()
            self.fail = True

        def forward(self):
            a = self.configuration["first"][0]["item"].storage
            b = self.configuration["second"][0].storage
            self.alpha.run(storage=a.step(), input_key="value", output_key="alpha")
            self.beta.run(storage=b.step(), input_key="value", output_key="beta")
            if self.fail:
                raise RuntimeError("failed after both cursors advanced")

    p = Example()
    old_first, old_second = p.alpha, p.beta
    with pytest.raises(RuntimeError, match="both cursors"):
        p.compile()
    assert (first.operator_step, second.operator_step) == (0, 0)
    assert p.alpha is old_first and p.beta is old_second
    assert not p.compiled and p.op_runtimes == []
    assert p.op_nodes_list == [] and p.llm_serving_counter == Counter()
    p.fail = False
    p.compile()
    p.compile()
    assert (first.operator_step, second.operator_step) == (1, 1)
    assert len(p.op_runtimes) == 2
    assert old_first.runs == old_second.runs == 0
    _execute(p, base)
    assert (old_first.runs, old_second.runs) == (1, 1)
    assert first.dataframe["alpha"].tolist() == [3, 4, 5]
    assert second.dataframe["beta"].tolist() == [3, 4, 5]


@pytest.mark.parametrize("base", PIPELINES)
def test_cyclic_alias_container_does_not_skip_storage_rollback(base):
    store = _Store()
    cycle = []
    cycle.extend([cycle, {"again": [store, store]}])

    class Example(base):
        def __init__(self):
            super().__init__()
            self.settings = {"nested": [cycle]}
            self.worker = _Increment()
            self.fail = True

        def forward(self):
            self.worker.run(
                storage=self.settings["nested"][0][1]["again"][0].step(),
                input_key="value", output_key="result",
            )
            if self.fail:
                raise RuntimeError("cycle failure")

    p = Example()
    with pytest.raises(RuntimeError, match="cycle failure"):
        p.compile()
    assert store.operator_step == 0
    assert not p.compiled
    p.fail = False
    p.compile()
    assert store.operator_step == 1
    assert len(p.op_runtimes) == 1
    _execute(p, base)
    assert store.dataframe["result"].tolist() == [3, 4, 5]


@pytest.mark.parametrize("base", PIPELINES)
def test_storage_cursor_restored_when_graph_introspection_raises(base):
    store = _Store(reject_keys=True)

    class Example(base):
        def __init__(self):
            super().__init__()
            self.nested = {"config": _Holder(store)}
            self.worker = _Increment()

        def forward(self):
            self.worker.run(
                storage=self.nested["config"].storage.step(),
                input_key="value", output_key="result",
            )

    p = Example()
    worker = p.worker
    with pytest.raises(LookupError, match="get-keys failure"):
        p.compile()
    assert store.operator_step == 0
    assert p.worker is worker
    assert p.op_runtimes == [] and p.op_nodes_list == []
    assert not p.compiled and not hasattr(p, "input_dataset_node")
    store.reject_keys = False
    p.compile()
    assert len(p.op_nodes_list) == 3
    assert store.operator_step == 1
    _execute(p, base)
    assert worker.runs == 1


@pytest.mark.parametrize("base", PIPELINES)
def test_repeat_compile_after_real_execution_does_not_execute_again(base):
    store = _Store()

    class Example(base):
        def __init__(self):
            super().__init__()
            self.nested_storage = (_Holder(store),)
            self.worker = _Increment()

        def forward(self):
            self.worker.run(
                storage=self.nested_storage[0].storage.step(),
                input_key="value", output_key="result",
            )

    p = Example()
    p.compile()
    worker = p.worker._operator
    graph_ids = tuple(map(id, p.op_nodes_list))
    original_runtime_ids = tuple(map(id, p.op_runtimes))
    _execute(p, base)
    assert worker.runs == 1
    p.compile()
    p.compile()
    assert worker.runs == 1
    assert tuple(map(id, p.op_nodes_list)) == graph_ids
    assert tuple(map(id, p.op_runtimes)) == original_runtime_ids
    assert store.operator_step == 1
    _execute(p, base)
    assert worker.runs == 2
    assert store.dataframe["result"].tolist() == [3, 4, 5]


@pytest.mark.parametrize("base", PIPELINES)
def test_keyboard_interrupt_rolls_back_compile_without_swallowing_signal(base):
    store = _Store()

    class Example(base):
        def __init__(self):
            super().__init__()
            self.resources = {"store": store}
            self.worker = _Increment()
            self.fail = True

        def forward(self):
            self.worker.run(
                storage=self.resources["store"].step(),
                input_key="value", output_key="result",
            )
            if self.fail:
                raise KeyboardInterrupt("compile cancelled")

    p = Example()
    with pytest.raises(KeyboardInterrupt, match="compile cancelled"):
        p.compile()
    assert store.operator_step == 0
    assert not p.compiled and p.op_nodes_list == []
    p.fail = False
    p.compile()
    assert p.compiled


@pytest.mark.parametrize("base", PIPELINES)
def test_failed_compile_removes_transient_attributes_then_retries(base):
    store = _Store()

    class Example(base):
        def __init__(self):
            super().__init__()
            self.config = {"storage": store}
            self.worker = _Increment()
            self.fail = True

        def forward(self):
            self._transient_during_compile = "should not survive failure"
            self.worker.run(
                storage=self.config["storage"].step(),
                input_key="value", output_key="result",
            )
            if self.fail:
                raise RuntimeError("late compile failure")

    p = Example()
    with pytest.raises(RuntimeError, match="late compile"):
        p.compile()
    assert not hasattr(p, "_transient_during_compile")
    assert store.operator_step == 0 and not p.compiled
    p.fail = False
    p.compile()
    assert p.compiled and store.operator_step == 1
