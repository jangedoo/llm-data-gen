import datetime
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import datasets

from datagen.core import stats as stats_module
from datagen.core.gen_config import GenerationPipelineConfig
from datagen.core.run_state import RunState
from datagen.core.stats import CallStats, StatsAggregator

logger = logging.getLogger(__name__)


def _shard_filename(index: int) -> str:
    return f"shard_{index:05d}.jsonl"


class _ShardWriter:
    """Append rows to ``<dir>/shard_NNNNN.jsonl``; rotate on demand."""

    def __init__(self, output_dir: Path, start_index: int = 1):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._index = start_index
        self._handle = None
        self._open()

    def _open(self) -> None:
        self._handle = open(self.output_dir / _shard_filename(self._index), "a")

    @property
    def index(self) -> int:
        return self._index

    @property
    def current_path(self) -> Path:
        return self.output_dir / _shard_filename(self._index)

    def write(self, row: dict) -> None:
        assert self._handle is not None
        self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._handle.flush()

    def rotate(self) -> int:
        """Close current shard, open the next one, return the new index."""
        self.close()
        self._index += 1
        self._open()
        return self._index

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class GenerationPipeline:
    def __init__(self, config: GenerationPipelineConfig):
        self.config = config
        self.generation_stats_per_dataset = defaultdict(lambda: defaultdict(int))
        self._state: Optional[RunState] = None
        self._stats: Optional[StatsAggregator] = None

    @property
    def num_generated(self):
        return sum(
            stat["num_rows"] for stat in self.generation_stats_per_dataset.values()
        )

    @property
    def num_invalid_generated(self):
        return sum(
            stat["num_invalid_rows"]
            for stat in self.generation_stats_per_dataset.values()
        )

    @property
    def num_valid_generated(self):
        return sum(
            stat["num_valid_rows"]
            for stat in self.generation_stats_per_dataset.values()
        )

    def start(self, fresh: bool = False) -> tuple[Path, Path]:
        out = self.config.generation_output_dir
        valid_dir = out / "valid"
        invalid_dir = out / "invalid"
        valid_dir.mkdir(parents=True, exist_ok=True)
        invalid_dir.mkdir(parents=True, exist_ok=True)

        # Resume state (or fresh)
        self._state = RunState.load_or_init(out, fresh=fresh)
        # Capture whether this session is resuming prior work *before* we touch
        # the state, so the session file records it accurately.
        is_resume = self._state.total_processed > 0
        self._state.save()  # touch on disk early so we always have a snapshot

        self._stats = StatsAggregator(
            out, run_id=self._state.run_id, is_resume=is_resume
        )

        upload_every = self.config.curator_config.upload_every_n_rows
        incremental = (
            self.config.curator_config.incremental_upload
            and self.config.curator_config.upload_to_hf
        )

        # Resume offsets per dataset
        start_offsets: dict[str, int] = {}
        completed: set[str] = set()
        for src_ds_config in self.config.generator_config.source_datasets_config:
            name = src_ds_config.source_name
            ds_state = self._state.get_or_create_dataset(name)
            start_offsets[name] = ds_state.next_input_index
            if ds_state.completed:
                completed.add(name)
        if completed:
            logger.info(f"Skipping already-completed datasets: {sorted(completed)}")

        # Note: we no longer replay prior counts into the aggregator. Each
        # session tracks only what it generated; the lifetime totals come from
        # merging every runs/session_*.json into summary.json.

        # Open shard writers, resuming from the last partial shard
        valid_writer = _ShardWriter(
            valid_dir, start_index=self._state.shards.current_valid_shard
        )
        invalid_writer = _ShardWriter(
            invalid_dir, start_index=self._state.shards.current_invalid_shard
        )

        def _mark_dataset_complete(name: str) -> None:
            if self._state is None:
                return
            ds = self._state.get_or_create_dataset(name)
            ds.completed = True
            self._state.save()
            logger.info(f"Marked dataset '{name}' as completed.")

        def _persist_session(status: str) -> None:
            """Write only the per-session file. Cheap; safe to call per-row."""
            assert self._stats is not None
            self._stats.write_session(status=status)

        def _persist_stats(status: str) -> None:
            """Write the session file and recompute summary.json from all sessions."""
            _persist_session(status)
            stats_module.write_aggregate(out)

        generator = self.config.generator_config.create_generator()
        rows = generator.generate(
            start_offsets=start_offsets,
            completed_datasets=completed,
            on_dataset_complete=_mark_dataset_complete,
        )

        last_logged_at = 0
        try:
            for row in rows:
                meta = row["__meta"]
                is_valid = meta["is_valid"]
                ds_name = meta["dataset_name"]
                input_index = meta.get("input_index")
                stats_dict = meta.get("stats")
                call_stats = (
                    CallStats(**stats_dict) if isinstance(stats_dict, dict) else None
                )

                # Write row to current shard
                if is_valid:
                    valid_writer.write(row)
                else:
                    invalid_writer.write(row)

                # Update state + aggregator
                ds_state = self._state.get_or_create_dataset(ds_name)
                if input_index is not None:
                    ds_state.next_input_index = input_index + 1
                else:
                    ds_state.next_input_index += 1
                ds_state.processed += 1
                if is_valid:
                    ds_state.valid += 1
                    self._state.shards.rows_in_current_valid_shard += 1
                else:
                    ds_state.invalid += 1
                self._stats.record(ds_name, valid=is_valid, stats=call_stats)

                # Legacy per-dataset counters (kept for back-compat)
                self.generation_stats_per_dataset[ds_name]["num_rows"] += 1
                self.generation_stats_per_dataset[ds_name]["num_valid_rows"] += (
                    1 if is_valid else 0
                )
                self.generation_stats_per_dataset[ds_name]["num_invalid_rows"] += (
                    0 if is_valid else 1
                )

                # Periodic logging
                total = self._state.total_processed
                if total - last_logged_at >= self.config.generation_logging_steps:
                    last_logged_at = total
                    logger.info(
                        f"Wrote {total:,} rows so far "
                        f"(valid={self._state.total_valid:,}, "
                        f"invalid={self._state.total_invalid:,})"
                    )

                # Persist state + session-stats snapshot (both cheap, atomic).
                # We write the session file every row so a hard crash preserves
                # an accurate per-session row/token count for summary.json.
                self._state.save()
                _persist_session(status="running")

                # Rotate shard + (optionally) upload every N valid rows
                if self._state.shards.rows_in_current_valid_shard >= upload_every:
                    closed_index = valid_writer.index
                    valid_writer.rotate()
                    self._state.shards.current_valid_shard = valid_writer.index
                    self._state.shards.rows_in_current_valid_shard = 0
                    _persist_stats(status="running")
                    self._state.save()
                    logger.info(
                        f"Rotated valid shard at {upload_every:,} rows "
                        f"(closed shard {closed_index})."
                    )
                    if incremental:
                        try:
                            self._push_incremental(
                                through_shard=closed_index, final=False
                            )
                        except Exception:
                            logger.exception(
                                "Incremental upload failed; continuing generation. "
                                "State preserves uploaded_through_shard."
                            )

            # End of stream — per-dataset completion was marked via the
            # on_dataset_complete callback inside generator.generate().
            self._state.save()

        except KeyboardInterrupt:
            logger.warning(
                "Interrupted by user. State and shards preserved at "
                f"{out}; rerun the same command to resume."
            )
            self._state.save()
            _persist_stats(status="interrupted")
            valid_writer.close()
            invalid_writer.close()
            raise
        except Exception:
            logger.exception("Pipeline failed; preserving state for resume.")
            # Persist stats first — mark_failed calls state.save(), which may
            # itself be the failing operation. We don't want to lose the
            # session record on top of the underlying error.
            _persist_stats(status="failed")
            try:
                self._state.mark_failed()
            except Exception:
                logger.exception("Could not persist failed status to run_state.")
            valid_writer.close()
            invalid_writer.close()
            raise

        valid_writer.close()
        invalid_writer.close()

        # Final summary
        _persist_stats(status="completed")
        logger.info(self._stats.format_summary())
        logger.info(
            stats_module.format_aggregate_summary(stats_module.compute_aggregate(out))
        )

        # Final upload
        if self.config.curator_config.upload_to_hf:
            try:
                if incremental:
                    self._push_incremental(
                        through_shard=valid_writer.index, final=True
                    )
                else:
                    self.push_to_hub()
            except Exception:
                logger.exception("Final upload failed; output is preserved locally.")
                raise

        self._state.mark_completed()

        # Return paths to the most recent valid/invalid shards for back-compat
        return valid_writer.current_path, invalid_writer.current_path

    def _push_incremental(self, through_shard: int, final: bool) -> None:
        """Push the growing dataset (all valid shards) as a single 'train' split.

        Note (scalability): this rebuilds the dataset from all local shards on
        every push. Fine to ~100k–500k rows; for true millions-scale runs
        replace with direct parquet shard upload via huggingface_hub.upload_file.
        """
        if self._state is None:
            raise RuntimeError("RunState not initialized")
        if (
            not final
            and through_shard <= self._state.upload.uploaded_through_shard
        ):
            return  # nothing new

        repo_id = self.config.curator_config.upload_repo_id
        if repo_id is None:
            raise RuntimeError("upload_repo_id is not set")

        valid_glob = str(self.config.generation_output_dir / "valid" / "shard_*.jsonl")
        ds: datasets.Dataset = datasets.Dataset.from_json(valid_glob)  # type: ignore
        if len(ds) == 0:
            logger.warning("dataset has no rows. Skipping upload.")
            return
        ds = ds.remove_columns(["__meta"])

        suffix = "(final)" if final else f"through shard {through_shard}"
        commit_msg = f"incremental upload {suffix} ({len(ds):,} rows)"
        ds.push_to_hub(repo_id=repo_id, commit_message=commit_msg)

        self._state.upload.uploaded_through_shard = through_shard
        self._state.upload.last_upload_at = datetime.datetime.now(
            datetime.UTC
        ).isoformat()
        self._state.upload.last_commit_message = commit_msg
        self._state.save()
        logger.info(
            f"Pushed dataset to HF Hub: https://huggingface.co/datasets/{repo_id} "
            f"({commit_msg})"
        )
        if final and self.config.curator_config.update_card:
            aggregate = stats_module.compute_aggregate(
                self.config.generation_output_dir
            )
            self.config.create_hf_dataset_card(aggregate=aggregate).push_to_hub(repo_id)
            logger.info(
                f"Pushed dataset card to HF Hub: https://huggingface.co/datasets/{repo_id}"
            )

    def push_to_hub(
        self,
        path: Path | str | None = None,
        repo_id: str | None = None,
        commit_message: str | None = None,
        update_card: bool | None = None,
        train_test_split: bool | None = None,
    ):
        if repo_id is None and not self.config.curator_config.upload_to_hf:
            raise Exception("upload_to_hub is not set to true in config")
        # Default to all valid shards (new layout) plus the legacy timestamped
        # filename pattern, so manual `push-to-hub` works on either layout.
        if path is None:
            valid_dir = self.config.generation_output_dir / "valid"
            shard_glob = valid_dir / "shard_*.jsonl"
            legacy_glob = valid_dir / "*jsonl"
            path = shard_glob if any(valid_dir.glob("shard_*.jsonl")) else legacy_glob
        path = Path(path) if not isinstance(path, Path) else path
        repo_id = repo_id or self.config.curator_config.upload_repo_id
        if repo_id is None:
            raise Exception("either pass repo_id or set `upload_repo_id` in config")

        update_card = (
            update_card
            if update_card is not None
            else self.config.curator_config.update_card
        )

        ds: datasets.Dataset = datasets.Dataset.from_json(str(path))  # type: ignore
        if len(ds) == 0:
            logger.warning("dataset has no rows. Not uploading to HuggingFace Hub")
            return
        ds = ds.remove_columns(["__meta"])

        train_test_split = (
            train_test_split
            if train_test_split is not None
            else self.config.curator_config.train_test_split
        )

        if train_test_split:
            logger.info("Splitting dataset into train, test, and valid")
            train_remaining_ds = ds.train_test_split(test_size=0.2)
            test_valid_ds = train_remaining_ds["test"].train_test_split(test_size=0.5)
            logger.info(
                f"Split dataset into train: {len(train_remaining_ds['train'])}, test: {len(test_valid_ds['test'])}, valid: {len(test_valid_ds['train'])}"
            )
            final_ds = datasets.DatasetDict(
                {
                    "train": train_remaining_ds["train"],
                    "test": test_valid_ds["test"],
                    "valid": test_valid_ds["train"],
                }
            )
        else:
            final_ds = ds

        final_ds.push_to_hub(
            repo_id=repo_id, commit_message=commit_message or "upload data"
        )
        logger.info(
            f"Pushed dataset to HuggingFace Hub: https://huggingface.co/datasets/{repo_id}"
        )
        if update_card:
            aggregate = stats_module.compute_aggregate(
                self.config.generation_output_dir
            )
            self.config.create_hf_dataset_card(aggregate=aggregate).push_to_hub(repo_id)
            logger.info(
                f"Pushed dataset card to HuggingFace Hub: https://huggingface.co/datasets/{repo_id}"
            )
