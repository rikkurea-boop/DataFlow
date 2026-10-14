"""Current-run training artifact checks shared by Text2Model and PDF2Model.

Only cache-local, specifically named generated outputs are invalidated. Other
files, other cache roots and source inputs are deliberately left untouched.
"""
import json
from pathlib import Path


def invalidate_generated_outputs(cache_base, *, text_intermediate=False):
    """Remove only the outputs this invocation is about to regenerate.

    Never invalidate a cache merely to *read* it: existing QA may be
    intentionally reused by workflows that do not invoke a producer.
    """
    cache = Path(cache_base)
    output = cache / ".cache" / "data"
    for name in ("qa.json", "dataset_info.json"):
        (output / name).unlink(missing_ok=True)
    if text_intermediate:
        (cache / ".cache" / "gpu" / "text2qa_step_step3.json").unlink(missing_ok=True)


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def validate_training_qa(cache_base, *, qa_type="kbc"):
    """Validate every QA record from the current producer, not file existence."""
    qa_file = Path(cache_base) / ".cache" / "data" / "qa.json"
    try:
        with qa_file.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
    except (OSError, ValueError):
        return False
    if not isinstance(rows, list) or not rows:
        return False
    if qa_type == "vqa":
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("images"), list):
                return False
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                return False
            if not any(isinstance(msg, dict) and msg.get("role") == "user"
                       and _nonempty(msg.get("content")) for msg in messages):
                return False
            if not any(isinstance(msg, dict) and msg.get("role") == "assistant"
                       and _nonempty(msg.get("content")) for msg in messages):
                return False
    else:
        for row in rows:
            if not isinstance(row, dict) or not all(
                _nonempty(row.get(k)) for k in ("instruction", "input", "output")
            ):
                return False
    return True


def validate_dataset_info(cache_base, dataset_name):
    """Ensure the generated metadata points to the selected cache's QA file."""
    path = Path(cache_base) / ".cache" / "data" / "dataset_info.json"
    try:
        with path.open("r", encoding="utf-8") as handle:
            info = json.load(handle)
        return (isinstance(info, dict) and isinstance(info.get(dataset_name), dict)
                and info[dataset_name].get("file_name") == "qa.json")
    except (OSError, ValueError, TypeError):
        return False
