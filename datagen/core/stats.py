"""Per-call LLM stats, per-session aggregator, and lifetime aggregation.

A *session* is one invocation of the pipeline (one call to ``GenerationPipeline.start()``).
Each session writes its own delta snapshot to ``<output_dir>/runs/session_<id>.json``.
``<output_dir>/summary.json`` is the rolling aggregation across every session file
in ``runs/`` and is rewritten on every save.

``StatsAggregator`` keeps streaming totals + a small reservoir sample of latencies
for percentile estimates. Percentiles are kept *per session* (where they are
meaningful) and dropped from the lifetime aggregate (cannot be recomposed from
per-session percentiles).
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SUMMARY_FILENAME = "summary.json"
RUNS_DIRNAME = "runs"
SESSION_FILE_PREFIX = "session_"

# Cap on how many latency samples we keep per dataset for percentile estimates.
# Reservoir sampling keeps the distribution unbiased while bounding memory.
LATENCY_RESERVOIR_SIZE = 2000


@dataclass
class CallStats:
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class _Bucket:
    rows: int = 0
    valid: int = 0
    invalid: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_sum_ms: float = 0.0
    latency_count: int = 0
    # reservoir sample of latencies (ms) for percentile estimates
    latency_samples: list[float] = field(default_factory=list)
    _samples_seen: int = 0

    def add_row(self, *, valid: bool, stats: Optional[CallStats]) -> None:
        self.rows += 1
        if valid:
            self.valid += 1
        else:
            self.invalid += 1
        if stats is None:
            return
        self.prompt_tokens += stats.prompt_tokens
        self.completion_tokens += stats.completion_tokens
        self.total_tokens += stats.total_tokens
        self.latency_sum_ms += stats.latency_ms
        self.latency_count += 1
        self._reservoir_add(stats.latency_ms)

    def _reservoir_add(self, value: float) -> None:
        self._samples_seen += 1
        if len(self.latency_samples) < LATENCY_RESERVOIR_SIZE:
            self.latency_samples.append(value)
        else:
            j = random.randint(0, self._samples_seen - 1)
            if j < LATENCY_RESERVOIR_SIZE:
                self.latency_samples[j] = value

    def snapshot(self) -> dict:
        latency_mean = (
            self.latency_sum_ms / self.latency_count if self.latency_count else 0.0
        )
        p50 = _percentile(self.latency_samples, 50)
        p95 = _percentile(self.latency_samples, 95)
        tokens_per_row_mean = (
            self.total_tokens / self.rows if self.rows else 0.0
        )
        return {
            "rows": self.rows,
            "valid": self.valid,
            "invalid": self.invalid,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms_mean": round(latency_mean, 2),
            "latency_ms_p50": round(p50, 2),
            "latency_ms_p95": round(p95, 2),
            "tokens_per_row_mean": round(tokens_per_row_mean, 2),
            "latency_count": self.latency_count,
        }


def _percentile(samples: list[float], pct: float) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _new_session_id() -> str:
    return SESSION_FILE_PREFIX + datetime.datetime.now(datetime.UTC).strftime(
        "%Y%m%d_%H%M%S_%f"
    )


class StatsAggregator:
    """Tracks the *current session*. One instance per ``pipeline.start()`` call."""

    def __init__(
        self,
        output_dir: Path,
        *,
        run_id: str = "",
        is_resume: bool = False,
        started_at: Optional[float] = None,
        session_id: Optional[str] = None,
    ):
        self.output_dir = output_dir
        self.run_id = run_id
        self.is_resume = is_resume
        self._started = started_at or time.time()
        self._started_iso = datetime.datetime.now(datetime.UTC).isoformat()
        self._session_id = session_id or _new_session_id()
        self._totals = _Bucket()
        self._per_dataset: dict[str, _Bucket] = defaultdict(_Bucket)

    @property
    def session_id(self) -> str:
        return self._session_id

    def record(self, dataset_name: str, valid: bool, stats: Optional[CallStats]) -> None:
        self._totals.add_row(valid=valid, stats=stats)
        self._per_dataset[dataset_name].add_row(valid=valid, stats=stats)

    def snapshot(self) -> dict:
        """Lightweight session snapshot — used by ``format_summary`` and tests."""
        return {
            "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "wall_time_seconds": round(time.time() - self._started, 2),
            "totals": self._totals.snapshot(),
            "per_dataset": {
                name: bucket.snapshot()
                for name, bucket in self._per_dataset.items()
            },
        }

    def snapshot_session(self, status: str) -> dict:
        snap = self.snapshot()
        return {
            "session_id": self._session_id,
            "run_id": self.run_id,
            "status": status,
            "is_resume": self.is_resume,
            "started_at": self._started_iso,
            "ended_at": snap["generated_at"],
            "wall_time_seconds": snap["wall_time_seconds"],
            "totals": snap["totals"],
            "per_dataset": snap["per_dataset"],
        }

    def write_session(self, status: str) -> Path:
        runs_dir = self.output_dir / RUNS_DIRNAME
        runs_dir.mkdir(parents=True, exist_ok=True)
        path = runs_dir / f"{self._session_id}.json"
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.snapshot_session(status), indent=2))
        os.replace(tmp, path)
        return path

    def format_summary(self) -> str:
        snap = self.snapshot()
        t = snap["totals"]
        lines = [
            f"--- Session summary ({snap['wall_time_seconds']}s) ---",
            f"  rows={t['rows']:,}  valid={t['valid']:,}  invalid={t['invalid']:,}",
            f"  tokens: prompt={t['prompt_tokens']:,}  completion={t['completion_tokens']:,}  total={t['total_tokens']:,}",
            f"  latency_ms: mean={t['latency_ms_mean']}  p50={t['latency_ms_p50']}  p95={t['latency_ms_p95']}",
        ]
        for name, ds in snap["per_dataset"].items():
            lines.append(
                f"  [{name}] rows={ds['rows']:,} valid={ds['valid']:,} "
                f"tokens={ds['total_tokens']:,} mean_latency_ms={ds['latency_ms_mean']}"
            )
        return "\n".join(lines)


def _empty_agg_bucket() -> dict:
    return {
        "rows": 0,
        "valid": 0,
        "invalid": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "latency_ms_sum": 0.0,
        "latency_count": 0,
    }


def _add_into(target: dict, src: dict) -> None:
    target["rows"] += int(src.get("rows", 0))
    target["valid"] += int(src.get("valid", 0))
    target["invalid"] += int(src.get("invalid", 0))
    target["prompt_tokens"] += int(src.get("prompt_tokens", 0))
    target["completion_tokens"] += int(src.get("completion_tokens", 0))
    target["total_tokens"] += int(src.get("total_tokens", 0))
    lc = int(src.get("latency_count", 0))
    target["latency_count"] += lc
    target["latency_ms_sum"] += float(src.get("latency_ms_mean", 0.0)) * lc


def _finalize_agg(b: dict) -> dict:
    latency_mean = (
        b["latency_ms_sum"] / b["latency_count"] if b["latency_count"] else 0.0
    )
    tokens_per_row = b["total_tokens"] / b["rows"] if b["rows"] else 0.0
    return {
        "rows": b["rows"],
        "valid": b["valid"],
        "invalid": b["invalid"],
        "prompt_tokens": b["prompt_tokens"],
        "completion_tokens": b["completion_tokens"],
        "total_tokens": b["total_tokens"],
        "latency_ms_mean": round(latency_mean, 2),
        "tokens_per_row_mean": round(tokens_per_row, 2),
    }


def compute_aggregate(output_dir: Path) -> dict:
    """Read every ``runs/session_*.json`` and merge into a lifetime aggregate."""
    runs_dir = Path(output_dir) / RUNS_DIRNAME
    sessions: list[dict] = []
    if runs_dir.exists():
        for path in sorted(runs_dir.glob(f"{SESSION_FILE_PREFIX}*.json")):
            try:
                sessions.append(json.loads(path.read_text()))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not parse session file at {path} ({e}); skipping.")

    totals = _empty_agg_bucket()
    per_dataset: dict[str, dict] = {}
    run_ids: set[str] = set()
    starts: list[str] = []
    ends: list[str] = []
    wall_time_sum = 0.0

    for s in sessions:
        rid = s.get("run_id")
        if rid:
            run_ids.add(rid)
        if s.get("started_at"):
            starts.append(s["started_at"])
        if s.get("ended_at"):
            ends.append(s["ended_at"])
        wall_time_sum += float(s.get("wall_time_seconds", 0.0))

        _add_into(totals, s.get("totals", {}))
        for name, ds_snap in s.get("per_dataset", {}).items():
            pd = per_dataset.setdefault(name, _empty_agg_bucket())
            _add_into(pd, ds_snap)

    return {
        "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "total_sessions": len(sessions),
        "total_runs": len(run_ids),
        "first_session_at": min(starts) if starts else None,
        "last_session_at": max(ends) if ends else None,
        "total_wall_time_seconds": round(wall_time_sum, 2),
        "totals": _finalize_agg(totals),
        "per_dataset": {n: _finalize_agg(b) for n, b in per_dataset.items()},
    }


def write_aggregate(output_dir: Path) -> Path:
    path = Path(output_dir) / SUMMARY_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(compute_aggregate(output_dir), indent=2))
    os.replace(tmp, path)
    return path


def format_aggregate_summary(aggregate: dict) -> str:
    t = aggregate.get("totals", {})
    return (
        f"--- Lifetime aggregate ({aggregate.get('total_sessions', 0)} session(s), "
        f"{aggregate.get('total_runs', 0)} run(s)) ---\n"
        f"  rows={t.get('rows', 0):,}  valid={t.get('valid', 0):,}  invalid={t.get('invalid', 0):,}\n"
        f"  tokens: prompt={t.get('prompt_tokens', 0):,}  "
        f"completion={t.get('completion_tokens', 0):,}  "
        f"total={t.get('total_tokens', 0):,}"
    )
