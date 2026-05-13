import abc
import itertools
import logging
from typing import (
    Generic,
    TypeVar,
    Dict,
    Any,
    Iterable,
    List,
    Optional,
    Union,
    Iterator,
)

import datasets
from pydantic import BaseModel  # noqa: F401  (re-exported usage by subclasses)

from datagen.core.gen_config import (
    DataSetConfig,
    DataSourceConfig,
    ModelConfig,
)
from datagen.llm import LLM

logger = logging.getLogger(__name__)

DatasetConfigT = TypeVar("DatasetConfigT", bound=DataSetConfig)


class BaseGeneratorConfig(Generic[DatasetConfigT], abc.ABC):
    def __init__(
        self,
        default_model: str,
        sources_config: Dict[str, DataSourceConfig],
        models_config: Dict[str, ModelConfig],
        source_datasets_config: List[DatasetConfigT],
        default_system_prompt: Optional[str] = None,
    ):
        self.default_model = default_model
        self.sources_config = sources_config
        self.models_config = models_config
        self.source_datasets_config = source_datasets_config
        self.default_system_prompt = default_system_prompt

    @classmethod
    def from_config(
        cls,
        generator_config: dict,
        sources_config: Dict[str, DataSourceConfig],
        models_config: Dict[str, ModelConfig],
    ):
        params = generator_config.get("params", {})
        default_model = params.get("default_model")
        if not default_model:
            raise ValueError("`default_model` must be set under params")

        if default_model not in models_config:
            raise ValueError(
                f"`{default_model}` model has not been defined in `models`"
            )

        source_datasets_config = params.get("source_datasets", [])
        if not source_datasets_config:
            raise ValueError(
                "At least one source_datasets must be defined in generator"
            )

        default_system_prompt = params.get("default_system_prompt")

        dataset_configs = []
        for src_ds in source_datasets_config:
            if src_ds["name"] not in sources_config:
                raise ValueError(
                    f"source_dataset with name {src_ds['name']} is not defined in sources section."
                )

            sys_prompt = src_ds.get("system_prompt")
            if sys_prompt is None and default_system_prompt is None:
                raise ValueError(
                    f"Either system_prompt for dataset `{src_ds['name']}` or `default_system_prompt` must be defined in generator.params"
                )

            model_name = src_ds.get("model", default_model)
            dataset_config = cls._create_dataset_config(
                src_ds=src_ds,
                sources_config=sources_config,
                models_config=models_config,
                model_name=model_name,
                default_system_prompt=default_system_prompt,
            )
            dataset_configs.append(dataset_config)

        return cls(
            default_model=default_model,
            sources_config=sources_config,
            models_config=models_config,
            source_datasets_config=dataset_configs,
            default_system_prompt=default_system_prompt,
        )

    @classmethod
    @abc.abstractmethod
    def _create_dataset_config(
        cls,
        src_ds: dict,
        sources_config: Dict[str, DataSourceConfig],
        models_config: Dict[str, ModelConfig],
        model_name: str,
        default_system_prompt: Optional[str],
    ) -> DatasetConfigT:
        pass

    def create_generator(self):
        # Import here to avoid circular imports
        from datagen.generators.templated import TemplatedGenerator

        # Cast self to the proper type since we only have one generator now
        return TemplatedGenerator(config=self)  # type: ignore


class BaseGenerator(Generic[DatasetConfigT], abc.ABC):
    def __init__(self, config: BaseGeneratorConfig[DatasetConfigT]):
        self.config = config
        self._llms: Dict[str, LLM] = {}

    def _get_llm(self, llm_key: str) -> LLM:
        llm = self._llms.get(llm_key)
        if llm is not None:
            return llm

        llm = self.config.models_config[llm_key].create_llm()
        self._llms[llm_key] = llm
        return llm

    def get_dataset(
        self,
        ds_config: DatasetConfigT,
        start_offset: int = 0,
    ) -> tuple[Iterable[Dict[str, Any]], Optional[int]]:
        """Return ``(rows_iterable, total_length_or_none)``.

        ``total_length_or_none`` is None for streaming sources (length unknown).
        ``start_offset`` is applied so callers consume rows starting at that
        original-dataset index — used to resume mid-run.
        """
        logger.info(f"Processing dataset {ds_config.source_name}")
        ds = ds_config.source_config.create_dataset()
        is_streaming = isinstance(ds, datasets.IterableDataset)

        if is_streaming:
            if ds_config.shuffle:
                raise ValueError(
                    f"`shuffle=true` is not supported for streaming source "
                    f"'{ds_config.source_name}'. Disable shuffle or set "
                    f"`streaming=false` on the source."
                )
            stop = (
                start_offset + ds_config.max_records
                if ds_config.max_records is not None
                else None
            )
            rows: Iterable[Dict[str, Any]] = itertools.islice(
                ds, start_offset, stop
            )
            logger.info(
                f"Dataset ready (streaming). Resuming from offset {start_offset}; "
                f"max_records={ds_config.max_records}."
            )
            return rows, None

        if ds_config.shuffle:
            logger.info("Shuffling dataset")
            ds = ds.shuffle(seed=10)
        full_len = len(ds)
        end = (
            min(start_offset + ds_config.max_records, full_len)
            if ds_config.max_records is not None
            else full_len
        )
        if start_offset >= end:
            logger.info(
                f"Dataset {ds_config.source_name} already fully processed "
                f"(offset={start_offset}, end={end}). Skipping."
            )
            return [], 0
        ds = ds.select(range(start_offset, end))
        logger.info(
            f"Dataset ready. {len(ds)} records will be processed "
            f"(start_offset={start_offset}, full_len={full_len})."
        )
        return ds, len(ds)

    def _create_messages(
        self, system_prompt: str, content: str
    ) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    def _handle_failure(
        self,
        e: Exception,
        content: str,
        ds_config: DataSetConfig,
        total_failures: int,
        max_failures: Union[int, float],
        dataset_length: Optional[int],
    ) -> tuple[int, bool]:
        total_failures += 1
        logger.warning(
            f"Unable to get response from llm for content {content[:100]}...",
            exc_info=True,
        )

        if isinstance(max_failures, int):
            max_failures_count: Optional[int] = max_failures
        elif dataset_length is not None:
            max_failures_count = int(dataset_length * max_failures)
        else:
            # Streaming source with float max_failures: cannot compute a
            # fraction without knowing total length. Skip the cap and warn.
            max_failures_count = None
            if total_failures == 1:
                logger.warning(
                    "max_failures is a fraction but dataset length is unknown "
                    "(streaming source). The failure cap is disabled; set "
                    "`max_failures` to an integer to enforce a cap."
                )

        should_stop = (
            max_failures_count is not None and total_failures >= max_failures_count
        )

        if should_stop:
            logger.warning(
                f"Number of failures {total_failures} exceeded maximum allowed "
                f"failures {max_failures_count}. Not processing dataset: "
                f"{ds_config.source_name}"
            )

        return total_failures, should_stop

    @abc.abstractmethod
    def process_dataset(
        self,
        rows: Iterable[Dict[str, Any]],
        ds_config: DatasetConfigT,
        llm: LLM,
        dataset_length: Optional[int] = None,
        start_offset: int = 0,
    ) -> Iterator[Dict[str, Any]]:
        pass

    def generate(
        self,
        start_offsets: Optional[Dict[str, int]] = None,
        completed_datasets: Optional[set[str]] = None,
        on_dataset_complete=None,
    ):
        """Yield rows across all configured datasets.

        ``start_offsets[name]`` is the original-dataset index to resume from
        for that dataset. ``completed_datasets`` is the set of dataset names
        the caller already considers done (skipped entirely).
        ``on_dataset_complete(name)`` is invoked after a dataset's rows are
        fully yielded (used by the pipeline to mark resume state).
        """
        start_offsets = start_offsets or {}
        completed_datasets = completed_datasets or set()
        for src_ds_config in self.config.source_datasets_config:
            name = src_ds_config.source_name
            if name in completed_datasets:
                logger.info(f"Dataset {name} already completed; skipping.")
                continue
            offset = start_offsets.get(name, 0)
            rows, length = self.get_dataset(
                ds_config=src_ds_config, start_offset=offset
            )
            llm = self._get_llm(llm_key=src_ds_config.model_name)
            yield from self.process_dataset(
                rows=rows,
                ds_config=src_ds_config,
                llm=llm,
                dataset_length=length,
                start_offset=offset,
            )
            logger.info(
                f"Finished processing dataset {src_ds_config.source_name}. "
                f"Total llm consumption: {llm.get_usage_stats()}"
            )
            if on_dataset_complete is not None:
                on_dataset_complete(name)
        logger.info("Finished processing all datasets")
