"""Persistent run-level state for resume support.

Stored at ``<output_dir>/run_state.json``. One run per output dir. Updated
in-place after every row (atomic via tmp + os.replace) so a crash leaves a
consistent snapshot.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

STATE_FILENAME = "run_state.json"


@dataclass
class DatasetState:
    next_input_index: int = 0
    processed: int = 0
    valid: int = 0
    invalid: int = 0
    completed: bool = False


@dataclass
class ShardState:
    """Global shard rotation state (shared across all datasets in the run)."""
    current_valid_shard: int = 1
    rows_in_current_valid_shard: int = 0
    current_invalid_shard: int = 1


@dataclass
class UploadState:
    uploaded_through_shard: int = 0
    last_upload_at: Optional[str] = None
    last_commit_message: Optional[str] = None


@dataclass
class RunState:
    run_id: str
    started_at: str
    updated_at: str
    status: str = "running"  # running | completed | failed
    datasets: dict[str, DatasetState] = field(default_factory=dict)
    shards: ShardState = field(default_factory=ShardState)
    upload: UploadState = field(default_factory=UploadState)

    _path: Optional[Path] = field(default=None, repr=False, compare=False)

    @classmethod
    def _new(cls, output_dir: Path) -> "RunState":
        now = datetime.datetime.now(datetime.UTC).isoformat()
        run_id = f"run_{datetime.datetime.now(datetime.UTC).strftime('%Y%m%d_%H%M%S_%f')}"
        return cls(
            run_id=run_id,
            started_at=now,
            updated_at=now,
            status="running",
            datasets={},
            upload=UploadState(),
            _path=output_dir / STATE_FILENAME,
        )

    @classmethod
    def load_or_init(cls, output_dir: Path, fresh: bool = False) -> "RunState":
        """Load an existing state file from ``output_dir`` or initialize a new one.

        If ``fresh=True``, any existing state file is ignored (and will be
        overwritten on the next save).
        """
        path = output_dir / STATE_FILENAME
        if not fresh and path.exists():
            try:
                data = json.loads(path.read_text())
                state = cls._from_dict(data)
                state._path = path
                logger.info(
                    f"Resuming from existing run_state at {path} "
                    f"(run_id={state.run_id}, status={state.status})"
                )
                return state
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(
                    f"Could not parse existing run_state at {path} ({e}); starting fresh."
                )
        if fresh and path.exists():
            logger.info(f"--fresh requested; existing run_state at {path} will be overwritten.")
        return cls._new(output_dir)

    @classmethod
    def _from_dict(cls, data: dict) -> "RunState":
        datasets = {
            name: DatasetState(**ds) for name, ds in data.get("datasets", {}).items()
        }
        upload = UploadState(**data.get("upload", {}))
        shards = ShardState(**data.get("shards", {}))
        return cls(
            run_id=data["run_id"],
            started_at=data["started_at"],
            updated_at=data["updated_at"],
            status=data.get("status", "running"),
            datasets=datasets,
            shards=shards,
            upload=upload,
        )

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "datasets": {name: asdict(ds) for name, ds in self.datasets.items()},
            "shards": asdict(self.shards),
            "upload": asdict(self.upload),
        }

    def get_or_create_dataset(self, name: str) -> DatasetState:
        ds = self.datasets.get(name)
        if ds is None:
            ds = DatasetState()
            self.datasets[name] = ds
        return ds

    def save(self) -> None:
        if self._path is None:
            raise RuntimeError("RunState has no _path set; cannot save")
        self.updated_at = datetime.datetime.now(datetime.UTC).isoformat()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        os.replace(tmp, self._path)

    def mark_completed(self) -> None:
        self.status = "completed"
        self.save()

    def mark_failed(self) -> None:
        self.status = "failed"
        self.save()

    @property
    def total_processed(self) -> int:
        return sum(ds.processed for ds in self.datasets.values())

    @property
    def total_valid(self) -> int:
        return sum(ds.valid for ds in self.datasets.values())

    @property
    def total_invalid(self) -> int:
        return sum(ds.invalid for ds in self.datasets.values())
