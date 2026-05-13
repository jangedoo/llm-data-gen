"""Verify the HF dataset card renders the Generation Stats section from an aggregate."""
from pathlib import Path

import pytest

from datagen.core.gen_config import GenerationPipelineConfig


def _write_minimal_config(tmp_path: Path) -> Path:
    output_dir = tmp_path / "out"
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f"""
dataset_name = "Card test"
description = "A test dataset"
authors = ["Tester"]
generation_output_dir = "{output_dir.name}"

[sources.src]
path = "dummy/source"

[models.dummy]
backend = "dummy"
[models.dummy.params]
response = "x"

[generator]
generator = "templated"

[generator.params]
default_model = "dummy"
default_system_prompt = "Sys"
default_prompt_template = "{{{{ input.text }}}}"
output_template = '{{"x": "{{{{ llm_output }}}}"}}'
source_datasets = [{{ name = "src", max_records = 1 }}]

[curator.params]
upload_to_hf = true
upload_repo_id = "user/repo"
incremental_upload = false
train_test_split = false
language = ["en"]
license = "mit"
""".strip()
    )
    return cfg


def test_card_includes_stats_section_when_aggregate_provided(tmp_path):
    pytest.importorskip("huggingface_hub")
    cfg = GenerationPipelineConfig.from_path(_write_minimal_config(tmp_path))

    aggregate = {
        "total_sessions": 3,
        "total_runs": 2,
        "first_session_at": "2026-05-13T09:00:00+00:00",
        "last_session_at": "2026-05-14T10:00:00+00:00",
        "totals": {
            "rows": 1234,
            "valid": 1200,
            "invalid": 34,
            "total_tokens": 2_345_678,
        },
    }

    card = cfg.create_hf_dataset_card(aggregate=aggregate)
    text = str(card)

    assert "## Generation Stats" in text
    assert "**Total rows:** 1,234" in text
    assert "**Valid / Invalid:** 1,200 / 34" in text
    assert "**Total tokens:** 2,345,678" in text
    assert "Generated across 3 session(s) over 2 run(s)" in text
    assert "**First session:** 2026-05-13" in text
    assert "**Last session:** 2026-05-14" in text


def test_card_omits_stats_section_when_no_aggregate(tmp_path):
    pytest.importorskip("huggingface_hub")
    cfg = GenerationPipelineConfig.from_path(_write_minimal_config(tmp_path))

    card = cfg.create_hf_dataset_card()
    text = str(card)
    assert "## Generation Stats" not in text
