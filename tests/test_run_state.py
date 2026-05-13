import json
from pathlib import Path

from datagen.core.run_state import (
    DatasetState,
    RunState,
    ShardState,
    UploadState,
    STATE_FILENAME,
)


def test_load_or_init_returns_fresh_state_when_no_file(tmp_path: Path):
    state = RunState.load_or_init(tmp_path)

    assert state.status == "running"
    assert state.datasets == {}
    assert state.shards == ShardState()
    assert state.upload == UploadState()
    assert state.run_id.startswith("run_")


def test_save_and_reload_round_trips_all_fields(tmp_path: Path):
    state = RunState.load_or_init(tmp_path)
    ds = state.get_or_create_dataset("ds1")
    ds.next_input_index = 42
    ds.processed = 42
    ds.valid = 40
    ds.invalid = 2
    state.shards.current_valid_shard = 3
    state.shards.rows_in_current_valid_shard = 250
    state.upload.uploaded_through_shard = 2
    state.save()

    reloaded = RunState.load_or_init(tmp_path)

    assert reloaded.run_id == state.run_id
    assert reloaded.datasets["ds1"].next_input_index == 42
    assert reloaded.datasets["ds1"].valid == 40
    assert reloaded.shards.current_valid_shard == 3
    assert reloaded.shards.rows_in_current_valid_shard == 250
    assert reloaded.upload.uploaded_through_shard == 2


def test_fresh_flag_ignores_existing_state(tmp_path: Path):
    state = RunState.load_or_init(tmp_path)
    state.get_or_create_dataset("ds1").processed = 99
    state.save()

    fresh = RunState.load_or_init(tmp_path, fresh=True)

    assert fresh.run_id != state.run_id
    assert fresh.datasets == {}


def test_save_is_atomic_no_tmp_file_left(tmp_path: Path):
    state = RunState.load_or_init(tmp_path)
    state.save()

    assert (tmp_path / STATE_FILENAME).exists()
    leftover = list(tmp_path.glob("*.tmp"))
    assert leftover == []


def test_corrupt_state_falls_back_to_fresh(tmp_path: Path):
    (tmp_path / STATE_FILENAME).write_text("not json {{{")

    state = RunState.load_or_init(tmp_path)

    assert state.datasets == {}
    assert state.run_id.startswith("run_")


def test_totals_aggregate_across_datasets(tmp_path: Path):
    state = RunState.load_or_init(tmp_path)
    state.get_or_create_dataset("a").processed = 10
    state.get_or_create_dataset("a").valid = 8
    state.get_or_create_dataset("a").invalid = 2
    state.get_or_create_dataset("b").processed = 5
    state.get_or_create_dataset("b").valid = 5

    assert state.total_processed == 15
    assert state.total_valid == 13
    assert state.total_invalid == 2
