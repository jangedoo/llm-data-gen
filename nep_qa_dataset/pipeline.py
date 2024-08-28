import datetime
import json
import logging
from collections import defaultdict
from pathlib import Path

import datasets

from nep_qa_dataset.gen_config import GenerationPipelineConfig

logger = logging.getLogger(__name__)


class GenerationPipeline:
    def __init__(self, config: GenerationPipelineConfig):
        self.config = config
        self.generation_stats_per_dataset = defaultdict(lambda: defaultdict(int))

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

    def start(self) -> tuple[Path, Path]:
        file_name = f"qa_dataset_{datetime.datetime.now(datetime.UTC).strftime('%Y%m%d_%H%M%S')}.jsonl"
        valid_file_path = self.config.generation_output_dir / "valid" / file_name
        invalid_file_path = self.config.generation_output_dir / "invalid" / file_name
        valid_file_path.parent.mkdir(exist_ok=True, parents=True)
        invalid_file_path.parent.mkdir(exist_ok=True, parents=True)
        generator = self.config.generator_config.create_generator()
        rows = generator.generate()

        with open(valid_file_path, "a") as valid_f, open(
            invalid_file_path, "a"
        ) as invalid_f:
            for row_num, row in enumerate(rows, start=1):
                meta = row["__meta"]
                is_valid = meta["is_valid"]
                ds_name = meta["dataset_name"]

                f = valid_f if is_valid else invalid_f
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

                self.generation_stats_per_dataset[ds_name]["num_rows"] += 1
                self.generation_stats_per_dataset[ds_name]["num_valid_rows"] += (
                    1 if is_valid else 0
                )
                self.generation_stats_per_dataset[ds_name]["num_invalid_rows"] += (
                    0 if is_valid else 1
                )

                if row_num % self.config.generation_logging_steps == 0:
                    logger.info(f"Wrote {row_num:,} rows so far")
                    logger.info(
                        f"{self.num_valid_generated:,} valid rows written to {valid_file_path} and {self.num_invalid_generated:,} invalid ones written to {invalid_file_path} so far"
                    )

        logger.info(f"Wrote {row_num:,} rows")
        logger.info(
            f"{self.num_valid_generated:,} valid rows written to {valid_file_path} and {self.num_invalid_generated:,} invalid ones written to {invalid_file_path}"
        )

        if self.config.curator_config.upload_to_hf:
            self.push_to_hub()
        return valid_file_path, invalid_file_path

    def push_to_hub(
        self,
        path: Path | str | None = None,
        repo_id: str | None = None,
        commit_message: str | None = None,
        update_card: bool | None = False,
    ):
        if repo_id is None and not self.config.curator_config.upload_to_hf:
            raise Exception("upload_to_hub is not set to true in config")
        path = (
            Path(path) if path else None
        ) or self.config.generation_output_dir / "valid" / "*jsonl"
        repo_id = repo_id or self.config.curator_config.upload_repo_id
        if repo_id is None:
            raise Exception("either pass repo_id or set `upload_repo_id` in config")

        update_card = (
            update_card
            if update_card is not None
            else self.config.curator_config.update_card
        )

        ds = datasets.Dataset.from_json(str(path))
        if len(ds) == 0:
            logger.warning("dataset has no rows. Not uploading to HuggingFace Hub")
            return
        ds = ds.remove_columns(["__meta"])
        ds.push_to_hub(repo_id=repo_id, commit_message=commit_message or "upload data")
        if update_card:
            self.config.create_hf_dataset_card().push_to_hub(repo_id)
