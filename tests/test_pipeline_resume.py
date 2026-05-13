"""End-to-end test: pipeline resume + sharded output + summary.

Uses DummyLLM and an in-memory HF Dataset so no network calls are made.
"""
import json
from pathlib import Path

import datasets
import pytest

from datagen.core.gen_config import GenerationPipelineConfig
from datagen.core.pipeline import GenerationPipeline
from datagen.core.run_state import RunState


def _write_config(
    tmp_path: Path,
    *,
    upload_every_n_rows: int = 50,
    max_records: int | None = 200,
    incremental_upload: bool = False,
) -> Path:
    output_dir = tmp_path / "out"
    cfg = tmp_path / "config.toml"
    max_records_line = (
        f"max_records = {max_records}, " if max_records is not None else ""
    )
    cfg.write_text(
        f"""
dataset_name = "Resume test"
description = "Resume test"
authors = ["Tester"]
generation_output_dir = "{output_dir.name}"
generation_logging_steps = 25

[sources.src]
path = "dummy/source"
split = "train"

[models.dummy]
backend = "dummy"
[models.dummy.params]
response = "translated"

[generator]
generator = "templated"

[generator.params]
default_model = "dummy"
default_system_prompt = "Sys"
default_prompt_template = "Translate: {{{{ input.text }}}}"
output_template = '{{"text": "{{{{ input.text }}}}", "out": "{{{{ llm_output }}}}"}}'
source_datasets = [
    {{ name = "src", {max_records_line}max_failures = 1000 }},
]

[curator.params]
upload_to_hf = false
incremental_upload = {str(incremental_upload).lower()}
upload_every_n_rows = {upload_every_n_rows}
train_test_split = false
""".strip()
    )
    return cfg


@pytest.fixture
def fake_source(monkeypatch):
    """Patch HFDataSourceConfig.create_dataset to return an in-memory dataset of 200 rows."""
    rows = [{"text": f"row-{i:04d}"} for i in range(200)]
    ds = datasets.Dataset.from_list(rows)

    from datagen.core import gen_config as gc

    def fake_create(self):
        return ds

    monkeypatch.setattr(gc.HFDataSourceConfig, "create_dataset", fake_create)
    return rows


def _read_all_shards(out_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for shard in sorted((out_dir / "valid").glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            rows.append(json.loads(line))
    return rows


def test_full_run_writes_sharded_output_and_summary(tmp_path, fake_source):
    cfg = _write_config(tmp_path, upload_every_n_rows=50, max_records=200)

    pipeline = GenerationPipeline(GenerationPipelineConfig.from_path(cfg))
    pipeline.start()

    out = tmp_path / "out"
    # Expect 4 closed shards (50 rows each) + 1 empty rotated shard
    valid_shards = sorted((out / "valid").glob("shard_*.jsonl"))
    assert len(valid_shards) >= 4

    rows = _read_all_shards(out)
    assert len(rows) == 200
    texts = [r["text"] for r in rows]
    assert texts == [f"row-{i:04d}" for i in range(200)]

    state = RunState.load_or_init(out)
    assert state.status == "completed"
    assert state.datasets["src"].processed == 200
    assert state.datasets["src"].completed is True

    summary = json.loads((out / "summary.json").read_text())
    assert summary["totals"]["rows"] == 200
    assert summary["totals"]["valid"] == 200
    assert summary["total_sessions"] == 1
    assert summary["total_runs"] == 1

    session_files = sorted((out / "runs").glob("session_*.json"))
    assert len(session_files) == 1
    session = json.loads(session_files[0].read_text())
    assert session["status"] == "completed"
    assert session["is_resume"] is False
    assert session["totals"]["rows"] == 200


def test_resume_after_interruption_reaches_full_count(
    tmp_path, fake_source, monkeypatch
):
    cfg = _write_config(tmp_path, upload_every_n_rows=50, max_records=200)
    out = tmp_path / "out"

    # First run: simulate a crash on the 121st row by injecting a failing
    # state.save() call. The pipeline saves after every row, so this raises
    # cleanly out of the main loop (after the row was already written and
    # state updated for that row).
    from datagen.core import run_state as rs

    real_save = rs.RunState.save
    counter = {"n": 0}

    def crashing_save(self):
        counter["n"] += 1
        # The pipeline calls save() many times before any rows (init + first
        # ds_state init etc.). We want to crash AFTER row 120 has been
        # written and state-updated. Easier: count post-row saves only.
        real_save(self)
        if self.total_processed == 120:
            raise RuntimeError("simulated crash")

    monkeypatch.setattr(rs.RunState, "save", crashing_save)

    pipeline = GenerationPipeline(GenerationPipelineConfig.from_path(cfg))
    with pytest.raises(RuntimeError, match="simulated crash"):
        pipeline.start()

    # Restore save behavior
    monkeypatch.setattr(rs.RunState, "save", real_save)

    state_after_crash = RunState.load_or_init(out)
    assert state_after_crash.datasets["src"].processed == 120
    assert state_after_crash.datasets["src"].next_input_index == 120

    # Sanity: shards on disk should contain 120 rows so far
    rows_so_far = _read_all_shards(out)
    assert len(rows_so_far) == 120

    # Resume — same command, state file picks up where we left off
    pipeline2 = GenerationPipeline(GenerationPipelineConfig.from_path(cfg))
    pipeline2.start()  # no fresh flag

    state = RunState.load_or_init(out)
    assert state.status == "completed"
    assert state.datasets["src"].processed == 200

    rows = _read_all_shards(out)
    texts = [r["text"] for r in rows]
    # 200 unique rows, in original input order, no duplicates
    assert texts == [f"row-{i:04d}" for i in range(200)]

    # Two session files, both for the same logical run, second marked as resume.
    session_files = sorted((out / "runs").glob("session_*.json"))
    assert len(session_files) == 2
    sessions = [json.loads(p.read_text()) for p in session_files]
    assert {s["run_id"] for s in sessions} == {state.run_id}
    statuses = {s["status"] for s in sessions}
    assert "failed" in statuses
    assert "completed" in statuses
    second = next(s for s in sessions if s["status"] == "completed")
    assert second["is_resume"] is True

    # summary.json aggregates both session deltas to the full 200 rows.
    summary = json.loads((out / "summary.json").read_text())
    assert summary["total_sessions"] == 2
    assert summary["total_runs"] == 1
    assert summary["totals"]["rows"] == 200
    assert summary["totals"]["valid"] == 200


def test_fresh_flag_starts_over(tmp_path, fake_source):
    cfg = _write_config(tmp_path, upload_every_n_rows=50, max_records=50)
    out = tmp_path / "out"

    pipeline = GenerationPipeline(GenerationPipelineConfig.from_path(cfg))
    pipeline.start()

    state1 = RunState.load_or_init(out)
    first_run_id = state1.run_id

    pipeline2 = GenerationPipeline(GenerationPipelineConfig.from_path(cfg))
    pipeline2.start(fresh=True)

    state2 = RunState.load_or_init(out)
    assert state2.run_id != first_run_id


def test_per_row_stats_attached_to_meta(tmp_path, fake_source):
    cfg = _write_config(tmp_path, upload_every_n_rows=50, max_records=10)

    pipeline = GenerationPipeline(GenerationPipelineConfig.from_path(cfg))
    pipeline.start()

    rows = _read_all_shards(tmp_path / "out")
    assert len(rows) == 10
    for row in rows:
        assert "__meta" in row
        assert "stats" in row["__meta"]
        stats = row["__meta"]["stats"]
        assert stats is not None
        assert stats["latency_ms"] >= 0
        # DummyLLM has no token info — base impl returns zeros
        assert stats["prompt_tokens"] == 0


def test_incremental_upload_with_split_is_rejected(tmp_path):
    """Config validation: incremental_upload=true + train_test_split=true is invalid."""
    output_dir = tmp_path / "out"
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f"""
dataset_name = "x"
description = "x"
authors = ["t"]
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
upload_repo_id = "u/r"
incremental_upload = true
train_test_split = true
""".strip()
    )

    with pytest.raises(ValueError, match="train_test_split"):
        GenerationPipelineConfig.from_path(cfg, create_output_dir=False)


def test_incremental_upload_without_upload_to_hf_is_rejected(tmp_path):
    output_dir = tmp_path / "out"
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f"""
dataset_name = "x"
description = "x"
authors = ["t"]
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
upload_to_hf = false
incremental_upload = true
train_test_split = false
""".strip()
    )

    with pytest.raises(ValueError, match="upload_to_hf"):
        GenerationPipelineConfig.from_path(cfg, create_output_dir=False)
