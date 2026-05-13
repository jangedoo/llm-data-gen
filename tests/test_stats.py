import json
from pathlib import Path

from datagen.core.stats import (
    CallStats,
    StatsAggregator,
    compute_aggregate,
    write_aggregate,
)


def _stats(latency_ms=10.0, prompt=5, completion=15):
    return CallStats(
        latency_ms=latency_ms,
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


def test_aggregator_totals_match_recorded(tmp_path: Path):
    agg = StatsAggregator(tmp_path)
    for _ in range(8):
        agg.record("ds1", valid=True, stats=_stats())
    for _ in range(2):
        agg.record("ds1", valid=False, stats=_stats(latency_ms=50))

    snap = agg.snapshot()

    assert snap["totals"]["rows"] == 10
    assert snap["totals"]["valid"] == 8
    assert snap["totals"]["invalid"] == 2
    assert snap["totals"]["prompt_tokens"] == 50
    assert snap["totals"]["completion_tokens"] == 150
    assert snap["totals"]["total_tokens"] == 200


def test_per_dataset_breakdown(tmp_path: Path):
    agg = StatsAggregator(tmp_path)
    agg.record("a", valid=True, stats=_stats(prompt=1, completion=1))
    agg.record("b", valid=True, stats=_stats(prompt=10, completion=20))
    agg.record("b", valid=False, stats=None)

    snap = agg.snapshot()

    assert snap["per_dataset"]["a"]["rows"] == 1
    assert snap["per_dataset"]["a"]["total_tokens"] == 2
    assert snap["per_dataset"]["b"]["rows"] == 2
    assert snap["per_dataset"]["b"]["valid"] == 1
    assert snap["per_dataset"]["b"]["invalid"] == 1
    assert snap["per_dataset"]["b"]["total_tokens"] == 30


def test_percentiles_are_in_range(tmp_path: Path):
    agg = StatsAggregator(tmp_path)
    for ms in range(1, 101):  # 1..100 ms
        agg.record("ds", valid=True, stats=_stats(latency_ms=float(ms)))

    snap = agg.snapshot()
    p50 = snap["totals"]["latency_ms_p50"]
    p95 = snap["totals"]["latency_ms_p95"]

    assert 40 <= p50 <= 60
    assert 90 <= p95 <= 100


def test_write_session_persists_session_file(tmp_path: Path):
    agg = StatsAggregator(tmp_path, run_id="run_test", is_resume=False)
    agg.record("ds", valid=True, stats=_stats())
    path = agg.write_session(status="completed")

    assert path.exists()
    assert path.parent.name == "runs"
    data = json.loads(path.read_text())
    assert data["totals"]["rows"] == 1
    assert data["session_id"] == agg.session_id
    assert data["run_id"] == "run_test"
    assert data["status"] == "completed"
    assert data["is_resume"] is False
    assert "started_at" in data
    assert "ended_at" in data
    assert "wall_time_seconds" in data


def test_session_snapshot_carries_latency_count(tmp_path: Path):
    agg = StatsAggregator(tmp_path)
    agg.record("ds", valid=True, stats=_stats(latency_ms=100))
    agg.record("ds", valid=True, stats=_stats(latency_ms=200))
    agg.record("ds", valid=False, stats=None)  # no call_stats — shouldn't bump latency_count

    snap = agg.snapshot_session(status="completed")
    assert snap["totals"]["latency_count"] == 2
    assert snap["per_dataset"]["ds"]["latency_count"] == 2


def _make_session_file(
    runs_dir: Path,
    *,
    session_id: str,
    run_id: str,
    started_at: str,
    ended_at: str,
    wall_time_seconds: float,
    totals: dict,
    per_dataset: dict,
    status: str = "completed",
) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{session_id}.json"
    path.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "run_id": run_id,
                "status": status,
                "is_resume": False,
                "started_at": started_at,
                "ended_at": ended_at,
                "wall_time_seconds": wall_time_seconds,
                "totals": totals,
                "per_dataset": per_dataset,
            }
        )
    )
    return path


def _bucket(rows, valid, invalid, prompt, completion, latency_mean, latency_count):
    return {
        "rows": rows,
        "valid": valid,
        "invalid": invalid,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "latency_ms_mean": latency_mean,
        "latency_ms_p50": latency_mean,
        "latency_ms_p95": latency_mean,
        "tokens_per_row_mean": (prompt + completion) / rows if rows else 0.0,
        "latency_count": latency_count,
    }


def test_compute_aggregate_sums_across_sessions(tmp_path: Path):
    runs = tmp_path / "runs"
    _make_session_file(
        runs,
        session_id="session_a",
        run_id="run_1",
        started_at="2026-05-13T09:00:00+00:00",
        ended_at="2026-05-13T09:10:00+00:00",
        wall_time_seconds=600.0,
        totals=_bucket(10, 9, 1, 100, 200, 100.0, 9),
        per_dataset={"src": _bucket(10, 9, 1, 100, 200, 100.0, 9)},
    )
    _make_session_file(
        runs,
        session_id="session_b",
        run_id="run_1",  # same run_id — interrupted + resumed
        started_at="2026-05-13T09:11:00+00:00",
        ended_at="2026-05-13T09:20:00+00:00",
        wall_time_seconds=540.0,
        totals=_bucket(5, 5, 0, 50, 100, 200.0, 5),
        per_dataset={"src": _bucket(5, 5, 0, 50, 100, 200.0, 5)},
    )
    _make_session_file(
        runs,
        session_id="session_c",
        run_id="run_2",  # distinct run
        started_at="2026-05-14T10:00:00+00:00",
        ended_at="2026-05-14T10:05:00+00:00",
        wall_time_seconds=300.0,
        totals=_bucket(2, 2, 0, 20, 40, 50.0, 2),
        per_dataset={"src": _bucket(2, 2, 0, 20, 40, 50.0, 2)},
    )

    agg = compute_aggregate(tmp_path)

    assert agg["total_sessions"] == 3
    assert agg["total_runs"] == 2  # run_1 and run_2
    assert agg["first_session_at"] == "2026-05-13T09:00:00+00:00"
    assert agg["last_session_at"] == "2026-05-14T10:05:00+00:00"
    assert agg["total_wall_time_seconds"] == 1440.0

    t = agg["totals"]
    assert t["rows"] == 17
    assert t["valid"] == 16
    assert t["invalid"] == 1
    assert t["total_tokens"] == 510
    # Call-weighted mean: (100*9 + 200*5 + 50*2) / (9+5+2) = 2000/16 = 125.0
    assert t["latency_ms_mean"] == 125.0


def test_compute_aggregate_drops_percentiles(tmp_path: Path):
    runs = tmp_path / "runs"
    _make_session_file(
        runs,
        session_id="session_x",
        run_id="run_x",
        started_at="2026-05-13T09:00:00+00:00",
        ended_at="2026-05-13T09:01:00+00:00",
        wall_time_seconds=60.0,
        totals=_bucket(1, 1, 0, 5, 10, 20.0, 1),
        per_dataset={"src": _bucket(1, 1, 0, 5, 10, 20.0, 1)},
    )

    agg = compute_aggregate(tmp_path)
    assert "latency_ms_p50" not in agg["totals"]
    assert "latency_ms_p95" not in agg["totals"]
    assert "latency_ms_p50" not in agg["per_dataset"]["src"]


def test_compute_aggregate_with_no_sessions(tmp_path: Path):
    agg = compute_aggregate(tmp_path)
    assert agg["total_sessions"] == 0
    assert agg["total_runs"] == 0
    assert agg["first_session_at"] is None
    assert agg["last_session_at"] is None
    assert agg["totals"]["rows"] == 0
    assert agg["per_dataset"] == {}


def test_write_aggregate_produces_summary_json(tmp_path: Path):
    runs = tmp_path / "runs"
    _make_session_file(
        runs,
        session_id="session_y",
        run_id="run_y",
        started_at="2026-05-13T09:00:00+00:00",
        ended_at="2026-05-13T09:01:00+00:00",
        wall_time_seconds=60.0,
        totals=_bucket(3, 3, 0, 10, 20, 50.0, 3),
        per_dataset={"src": _bucket(3, 3, 0, 10, 20, 50.0, 3)},
    )

    path = write_aggregate(tmp_path)
    assert path == tmp_path / "summary.json"
    data = json.loads(path.read_text())
    assert data["total_sessions"] == 1
    assert data["totals"]["rows"] == 3


def test_record_without_stats_still_increments_rows(tmp_path: Path):
    agg = StatsAggregator(tmp_path)
    agg.record("ds", valid=True, stats=None)
    agg.record("ds", valid=False, stats=None)

    snap = agg.snapshot()

    assert snap["totals"]["rows"] == 2
    assert snap["totals"]["total_tokens"] == 0
    assert snap["totals"]["latency_ms_mean"] == 0.0
