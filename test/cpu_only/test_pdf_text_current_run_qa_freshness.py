from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
QA_EXTRACTOR = ROOT / "dataflow" / "operators" / "knowledge_cleaning" / "generate" / "qa_extract.py"
VQA_FORMATTER = ROOT / "dataflow" / "operators" / "pdf2vqa" / "generate" / "vqa_formatter.py"
TEXT_CONVERTER = ROOT / "dataflow" / "cli_funcs" / "text2model_pipeline" / "merge_filter_qa_pairs.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.cpu
def test_qa_extractor_removes_stale_output_when_current_run_has_no_qa():
    source = _source(QA_EXTRACTOR)
    no_qa = source.index("if not all_qas:")
    next_section = source.index("# 保存JSON", no_qa)
    block = source[no_qa:next_section]
    assert "self.output_json_file" in block
    assert "Path(self.output_json_file)" in block
    assert ".unlink()" in block


@pytest.mark.cpu
def test_vqa_formatter_removes_stale_output_when_current_run_has_no_samples():
    source = _source(VQA_FORMATTER)
    no_results = source.index("if not results:")
    next_section = source.index("if self.output_json_file:", no_results)
    block = source[no_results:next_section]
    assert "self.output_json_file" in block
    assert "Path(self.output_json_file)" in block
    assert ".unlink()" in block


@pytest.mark.cpu
def test_text_conversion_failure_exits_nonzero():
    source = _source(TEXT_CONVERTER)
    failure = source.index('print("Data conversion failed")')
    tail = source[failure:failure + 160]
    assert "raise SystemExit(1)" in tail


@pytest.mark.cpu
def test_text_dataset_info_failure_exits_nonzero():
    source = _source(TEXT_CONVERTER)
    failure = source.index('print("Configuration file creation failed")')
    tail = source[failure:failure + 180]
    assert "raise SystemExit(1)" in tail


@pytest.mark.cpu
def test_qa_extractor_preserves_reusable_no_result_return_contract():
    source = _source(QA_EXTRACTOR)
    no_qa = source.index("if not all_qas:")
    next_section = source.index("# 保存JSON", no_qa)
    block = source[no_qa:next_section]
    assert "return [output_instruction_key, output_question_key, output_answer_key]" in block


@pytest.mark.cpu
def test_vqa_formatter_preserves_reusable_no_result_return_contract():
    source = _source(VQA_FORMATTER)
    no_results = source.index("if not results:")
    next_section = source.index("if self.output_json_file:", no_results)
    block = source[no_results:next_section]
    assert "return [output_messages_key, output_images_key]" in block


@pytest.mark.cpu
def test_successful_pdf_formatters_still_overwrite_output():
    qa_source = _source(QA_EXTRACTOR)
    vqa_source = _source(VQA_FORMATTER)
    assert "with open(output_path, 'w', encoding='utf-8') as f:" in qa_source
    assert 'with open(output_path, "w", encoding="utf-8") as f:' in vqa_source


@pytest.mark.cpu
def test_successful_text_conversion_still_creates_both_training_files():
    source = _source(TEXT_CONVERTER)
    assert 'qa_file = output_dir / "qa.json"' in source
    assert 'config_file = output_dir / "dataset_info.json"' in source
    assert "Ready for LlamaFactory training!" in source
