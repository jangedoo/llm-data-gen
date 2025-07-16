import abc
import json
import logging
import re
from typing import (
    Generic,
    TypeVar,
    Dict,
    Any,
    List,
    Optional,
    Union,
    Iterator,
)
from dataclasses import dataclass

import datasets
from pydantic import BaseModel

from datagen.core.gen_config import (
    DataSetConfig,
    DataSourceConfig,
    ModelConfig,
)
from datagen.llm import LLM
from datagen.core.registry import GeneratorRegistry

logger = logging.getLogger(__name__)

DatasetConfigT = TypeVar("DatasetConfigT", bound=DataSetConfig)


@dataclass
class BaseDatasetConfig(DataSetConfig):
    pass


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
        generator_class = self.__class__.__qualname__.replace(".Config", "")
        from datagen.core.registry import GeneratorRegistry

        # Find the generator by class name pattern
        for name, info in GeneratorRegistry.list_generators().items():
            if info.generator_class.__name__ == generator_class:
                return info.generator_class(config=self)

        raise ValueError(f"No generator found for config {self.__class__.__name__}")


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

    def get_dataset(self, ds_config: DatasetConfigT) -> datasets.Dataset:
        logger.info(f"Processing dataset {ds_config.source_name}")
        ds = ds_config.source_config.create_dataset()
        if ds_config.shuffle:
            logger.info("Shuffling dataset")
            ds = ds.shuffle(seed=10)
        ds = ds.select(range(ds_config.max_records))
        logger.info(
            f"Dataset ready for processing. {len(ds)} records will be processed."
        )
        return ds

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
        dataset_length: int,
    ) -> tuple[int, bool]:
        total_failures += 1
        logger.warning(
            f"Unable to get response from llm for content {content[:100]}...",
            exc_info=True,
        )

        max_failures_count = (
            max_failures
            if isinstance(max_failures, int)
            else int(dataset_length * max_failures)
        )
        should_stop = total_failures >= max_failures_count

        if should_stop:
            logger.warning(
                f"Number of failures {total_failures} exceeded maximum allowed failures {max_failures_count}. Not processing dataset: {ds_config.source_name}"
            )

        return total_failures, should_stop

    @abc.abstractmethod
    def process_dataset(
        self, ds: datasets.Dataset, ds_config: DatasetConfigT, llm: LLM
    ) -> Iterator[Dict[str, Any]]:
        pass

    def generate(self):
        for src_ds_config in self.config.source_datasets_config:
            ds = self.get_dataset(ds_config=src_ds_config)
            llm = self._get_llm(llm_key=src_ds_config.model_name)
            yield from self.process_dataset(ds=ds, ds_config=src_ds_config, llm=llm)
            logger.info(
                f"Finished processing dataset {src_ds_config.source_name}. Total llm consumption: {llm.get_usage_stats()}"
            )
        logger.info("Finished processing all datasets")
