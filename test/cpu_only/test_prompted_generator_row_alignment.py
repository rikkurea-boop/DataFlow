from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "dataflow" / "operators" / "core_text" / "generate" / "prompted_generator.py"

class _Logger:
    def info(self,*a,**k): pass
    def error(self,*a,**k): pass
class _Registry:
    def register(self): return lambda cls: cls
class _Serving:
    def __init__(self): self.inputs=None
    def generate_from_input(self, user_inputs, **kwargs):
        self.inputs=list(user_inputs)
        return [f"out:{x}" for x in user_inputs]
class _Storage:
    def __init__(self, values): self.df=pd.DataFrame({"raw_content":values}); self.written=None
    def read(self, output_type): return self.df.copy()
    def write(self, df): self.written=df.copy(); return "cache"

def _load(monkeypatch):
    dataflow=types.ModuleType("dataflow"); dataflow.get_logger=lambda:_Logger()
    core=types.ModuleType("dataflow.core"); core.OperatorABC=object; core.LLMServingABC=object
    reg=types.ModuleType("dataflow.utils.registry"); reg.OPERATOR_REGISTRY=_Registry()
    st=types.ModuleType("dataflow.utils.storage"); st.DataFlowStorage=object
    monkeypatch.setitem(sys.modules,"dataflow",dataflow)
    monkeypatch.setitem(sys.modules,"dataflow.core",core)
    monkeypatch.setitem(sys.modules,"dataflow.utils",types.ModuleType("dataflow.utils"))
    monkeypatch.setitem(sys.modules,"dataflow.utils.registry",reg)
    monkeypatch.setitem(sys.modules,"dataflow.utils.storage",st)
    return runpy.run_path(str(SOURCE),run_name="prompted_alignment_test")

@pytest.mark.cpu
def test_falsey_rows_preserve_one_input_and_output_per_dataframe_row(monkeypatch):
    ns=_load(monkeypatch); serving=_Serving(); storage=_Storage(["A","",0,False,None,"B"])
    op=ns["PromptedGenerator"](llm_serving=serving,user_prompt="P:")
    assert op.run(storage)=="generated_content"
    assert serving.inputs == ["P:A","P:","P:0","P:False","P:None","P:B"]
    assert storage.written["generated_content"].tolist() == [f"out:{x}" for x in serving.inputs]
    assert len(storage.written)==6
