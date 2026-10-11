from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PDF_CLI = ROOT / "dataflow" / "cli_funcs" / "cli_pdf.py"
TEXT_CLI = ROOT / "dataflow" / "cli_funcs" / "cli_text.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _before(source: str, left: str, right: str) -> bool:
    left_pos = source.find(left)
    right_pos = source.find(right)
    return left_pos >= 0 and right_pos >= 0 and left_pos < right_pos


@pytest.mark.cpu
def test_pdf_selected_qa_path_is_established_before_step2():
    source = _source(PDF_CLI)
    assert _before(
        source,
        'qa_json_path = cache_path_obj / ".cache" / "data" / "qa.json"',
        '# Step 2: Data Processing',
    )


@pytest.mark.cpu
def test_pdf_stale_qa_is_invalidated_before_step2():
    source = _source(PDF_CLI)
    assert _before(source, "qa_json_path.unlink()", '# Step 2: Data Processing')


@pytest.mark.cpu
def test_text_selected_training_artifacts_are_established_before_step3():
    source = _source(TEXT_CLI)
    assert _before(
        source,
        'qa_file = cache_path_obj / ".cache" / "data" / "qa.json"',
        '# Step 3: Convert to training format',
    )
    assert _before(
        source,
        'dataset_info_file = cache_path_obj / ".cache" / "data" / "dataset_info.json"',
        '# Step 3: Convert to training format',
    )


@pytest.mark.cpu
def test_text_stale_qa_is_invalidated_before_step3():
    source = _source(TEXT_CLI)
    step3 = source.index('# Step 3: Convert to training format')
    conversion = source.index('run_script_with_args(script3_path, "QA format conversion"', step3)
    qa_unlink = source.find("qa_file.unlink()", step3, conversion)
    assert qa_unlink >= 0


@pytest.mark.cpu
def test_text_stale_dataset_info_is_invalidated_before_step3():
    source = _source(TEXT_CLI)
    step3 = source.index('# Step 3: Convert to training format')
    conversion = source.index('run_script_with_args(script3_path, "QA format conversion"', step3)
    info_unlink = source.find("dataset_info_file.unlink()", step3, conversion)
    assert info_unlink >= 0


@pytest.mark.cpu
def test_pdf_postcondition_still_requires_qa_after_step2():
    source = _source(PDF_CLI)
    step2 = source.index('# Step 2: Data Processing')
    check = source.index("if qa_json_path.exists():", step2)
    assert check > step2


@pytest.mark.cpu
def test_text_postcondition_still_requires_both_training_files_after_step3():
    source = _source(TEXT_CLI)
    step3 = source.index('# Step 3: Convert to training format')
    check = source.index("if not qa_file.exists() or not dataset_info_file.exists():", step3)
    assert check > step3


@pytest.mark.cpu
def test_current_stage_invocations_remain_present():
    pdf_source = _source(PDF_CLI)
    text_source = _source(TEXT_CLI)
    assert 'run_script_with_args(script2, "Step 2: Data Processing"' in pdf_source
    assert 'run_script_with_args(script3_path, "QA format conversion"' in text_source
