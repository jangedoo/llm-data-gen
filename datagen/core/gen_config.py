import abc
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import datasets
from dotenv import load_dotenv

from datagen import llm

load_dotenv()


@dataclass
class DataSourceConfig(abc.ABC):
    @abc.abstractmethod
    def create_dataset(self) -> datasets.Dataset:
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def from_config(cls, source_config: dict):
        raise NotImplementedError()


@dataclass
class HFDataSourceConfig(DataSourceConfig):
    path: str
    subset: str | None = None
    split: str = "train"
    streaming: bool = False

    def create_dataset(self):
        # Returns Dataset when streaming=False, IterableDataset when True.
        return datasets.load_dataset(
            self.path, self.subset, split=self.split, streaming=self.streaming
        )  # type: ignore

    @classmethod
    def from_config(cls, source_config: dict):
        if not "path" in source_config:
            raise ValueError("`path` must be specified for the data source")
        hf_path = source_config["path"]
        subset = source_config.get("subset")
        split = source_config.get("split", "train")
        streaming = bool(source_config.get("streaming", False))
        return cls(path=hf_path, subset=subset, split=split, streaming=streaming)


@dataclass
class AutoDataSourceConfig:
    @classmethod
    def from_config(cls, source_config: dict):
        module = source_config.get("module", "hf")
        return HFDataSourceConfig.from_config(source_config)


@dataclass
class ModelConfig:
    @classmethod
    @abc.abstractmethod
    def from_config(cls, model_config: dict):
        raise NotImplementedError()

    @abc.abstractmethod
    def create_llm(self) -> llm.LLM:
        raise NotImplementedError()


@dataclass
class DummyModelConfig(ModelConfig):
    response: str

    @classmethod
    def from_config(cls, model_config: dict):
        params = model_config.get("params", {})
        if not params:
            raise ValueError(f"`params` must be defined")
        response = params.get("response")
        if not response:
            raise ValueError("`response` must be defined in params")

        return cls(response=response)

    def create_llm(self) -> llm.LLM:
        from datagen.llm.dummy import DummyLLM

        return DummyLLM(response=self.response)


def _resolve_env_param(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if set(value) != {"env"}:
            raise ValueError(
                f"`{field_name}` env reference must only contain an `env` key"
            )
        env_name = value["env"]
        if not isinstance(env_name, str) or not env_name.strip():
            raise ValueError(f"`{field_name}` env reference must be a non-empty string")
        env_value = os.environ.get(env_name)
        if env_value is None:
            raise ValueError(
                f"`{field_name}` references missing environment variable `{env_name}`"
            )
        return env_value
    raise ValueError(f"`{field_name}` must be a string or an env reference")


@dataclass
class OpenAIModelConfig(ModelConfig):
    model: str
    temperature: float = 0.3
    max_tokens: int = 1000
    top_p: float = 1
    frequency_penalty: float = 0
    presence_penalty: float = 0
    api_base: str | None = None
    api_key: str | None = None

    @classmethod
    def from_config(cls, model_config: dict):
        params = model_config.get("params", {})

        model = params.get("model")
        if not model:
            raise ValueError(
                "`model` must be defined under params for models with 'openai' backend"
            )

        temperature = float(params.get("temperature", 0.3))
        max_tokens = int(params.get("max_tokens", 1000))
        top_p = float(params.get("top_p", 1))
        frequency_penalty = float(params.get("frequency_penalty", 0))
        presence_penalty = float(params.get("presence_penalty", 0))
        api_base = _resolve_env_param(params.get("api_base"), "api_base")
        api_key = _resolve_env_param(params.get("api_key"), "api_key")

        return cls(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            api_base=api_base,
            api_key=api_key,
        )

    def create_llm(self) -> llm.LLM:
        import openai

        from datagen.llm.openai import OpenAILLM

        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        api_base = self.api_base or os.environ.get("OPENAI_API_BASE")
        client = openai.OpenAI(api_key=api_key, base_url=api_base)
        return OpenAILLM(
            client=client,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )


@dataclass
class AutoModelConfig:
    @classmethod
    def from_config(cls, model_config: dict):
        backend = model_config.get("backend")
        if not backend:
            raise ValueError("`backend` must be defined in the model config")
        if backend.lower() == "openai":
            return OpenAIModelConfig.from_config(model_config=model_config)
        elif backend.lower() == "dummy":
            return DummyModelConfig.from_config(model_config=model_config)
        raise ValueError(f"Unsupported model backend: {backend}")


@dataclass
class DataSetConfig:
    source_name: str
    source_config: DataSourceConfig
    model_name: str
    model_config: ModelConfig
    system_prompt: str
    max_records: int | None
    shuffle: bool
    max_failures: float | int


@dataclass
class AutoGeneratorConfig:
    @classmethod
    def from_config(
        cls,
        generator_config: dict,
        sources_config: dict[str, DataSourceConfig],
        models_config: dict[str, ModelConfig],
    ):
        from datagen.generators.templated import TemplatedGenerator

        # Since we only support templated generator now
        generator_name = generator_config.get("generator")
        if not generator_name:
            raise Exception("Generator config must have `generator` field.")

        if generator_name != "templated":
            raise ValueError(
                f"Only 'templated' generator is supported, got '{generator_name}'"
            )

        return TemplatedGenerator.Config.from_config(
            generator_config=generator_config,
            sources_config=sources_config,
            models_config=models_config,
        )


@dataclass
class CuratorConfig:
    upload_to_hf: bool = False
    upload_repo_id: str | None = None
    update_card: bool = True
    language: list = field(default_factory=list)
    license: str = "mit"
    train_test_split: bool = True
    task_categories: list = field(default_factory=list)
    task_ids: list = field(default_factory=list)
    citation_bibtex: str = ""
    # Periodically push the growing dataset (as a single 'train' split) to the
    # HF Hub every ``upload_every_n_rows`` valid rows. Required for long runs
    # where losing all progress on crash is unacceptable. When True,
    # train_test_split must be False (splits over a growing dataset are
    # unstable across uploads).
    incremental_upload: bool = False
    upload_every_n_rows: int = 1000


@dataclass
class GenerationPipelineConfig:
    authors: list[str]
    description: str

    generation_output_dir: Path
    sources_config: dict[str, DataSourceConfig]

    models_config: dict[str, ModelConfig]

    generator_config: Any  # BaseGeneratorConfig type from generators.py
    generation_logging_steps: int

    curator_config: CuratorConfig

    @classmethod
    def from_path(cls, path: Path | str, create_output_dir: bool = True):
        path = Path(path)
        config_dict = tomllib.load(path.open("rb"))
        return cls.from_dict(
            config_dict=config_dict,
            base_dir=path.parent,
            create_output_dir=create_output_dir,
        )

    @classmethod
    def from_dict(
        cls,
        config_dict: dict,
        base_dir: Path | str = Path("."),
        create_output_dir: bool = True,
    ):
        base_dir = Path(base_dir)
        description = config_dict.get(
            "description", "Dataset generated using datagen library!"
        )
        authors = config_dict.get("authors", [])

        sources_data = config_dict.get("sources", {})
        if not sources_data:
            raise ValueError("There must be at least one source defined in the config")

        sources_config = {}
        for source_key, source_data in sources_data.items():
            sources_config[source_key] = AutoDataSourceConfig.from_config(source_data)

        generator_data = config_dict.get("generator", {})
        if not generator_data:
            raise ValueError("`generator` must be defined in the config")
        generator_name = generator_data.get("generator")
        if generator_name != "templated":
            raise ValueError(
                f"Only 'templated' generator is supported, got '{generator_name}'"
            )

        models_data = config_dict.get("models", {})
        if not models_data:
            raise ValueError("`models` must be defined in the config")

        models_config = {}
        for model_key, model_data in models_data.items():
            model_config = AutoModelConfig.from_config(model_data)
            models_config[model_key] = model_config

        generator_config = AutoGeneratorConfig.from_config(
            generator_data, sources_config=sources_config, models_config=models_config
        )

        if not "generation_output_dir" in config_dict:
            raise ValueError("`generation_output_dir` must be specified in the config")

        generation_output_dir = Path(config_dict["generation_output_dir"])
        generation_output_dir = base_dir.joinpath(generation_output_dir).resolve()
        if create_output_dir and not generation_output_dir.exists():
            print(
                f"{generation_output_dir} does not exist. Will create output directory"
            )
            generation_output_dir.mkdir(exist_ok=True, parents=True)

        generation_logging_steps = int(config_dict.get("generation_logging_steps", 100))

        curator_data = config_dict.get("curator", {}).get("params", {})
        curator_config = CuratorConfig(
            upload_to_hf=curator_data.get("upload_to_hf", False),
            upload_repo_id=curator_data.get("upload_repo_id"),
            update_card=curator_data.get("update_card", True),
            language=curator_data.get("language"),
            train_test_split=curator_data.get("train_test_split", True),
            license=curator_data.get("license"),
            task_categories=curator_data.get("task_categories"),
            task_ids=curator_data.get("task_ids"),
            citation_bibtex=curator_data.get("citation_bibtex"),
            incremental_upload=curator_data.get("incremental_upload", False),
            upload_every_n_rows=int(curator_data.get("upload_every_n_rows", 1000)),
        )
        if curator_config.upload_to_hf and not curator_config.upload_repo_id:
            raise ValueError(f"`upload_to_hf` is True but no `upload_repo_id` defined.")
        if curator_config.incremental_upload:
            if not curator_config.upload_to_hf:
                raise ValueError(
                    "`incremental_upload` is True but `upload_to_hf` is False. "
                    "Set both to True to enable periodic uploads."
                )
            if curator_config.train_test_split:
                raise ValueError(
                    "`incremental_upload` is True but `train_test_split` is also True. "
                    "Splits over a growing dataset are unstable across uploads; "
                    "set `train_test_split = false` when using incremental uploads."
                )
        if curator_config.upload_every_n_rows < 1:
            raise ValueError("`upload_every_n_rows` must be >= 1")
        return cls(
            authors=authors,
            generation_output_dir=generation_output_dir,
            description=description,
            sources_config=sources_config,
            models_config=models_config,
            generator_config=generator_config,
            generation_logging_steps=generation_logging_steps,
            curator_config=curator_config,
        )

    def create_hf_dataset_card(self, aggregate: dict | None = None):
        from huggingface_hub import DatasetCard, DatasetCardData

        description = (
            self.description
            + "\nThis dataset was automatically generated using "
            "[llm-data-gen](https://github.com/jangedoo/llm-data-gen) library"
        )
        if aggregate:
            description += "\n\n" + _format_generation_stats_section(aggregate)

        card_data = DatasetCardData(
            language=self.curator_config.language,
            license=self.curator_config.license,
            annotations_creators=["machine-generated"],
            task_categories=self.curator_config.task_categories,
            task_ids=self.curator_config.task_ids,
            pretty_name=self.description,
            source_datasets=[
                c.path
                for c in self.sources_config.values()
                if isinstance(c, HFDataSourceConfig)
            ],
            curators=self.authors,
            dataset_description=description,
            citation_bibtex=self.curator_config.citation_bibtex,
            dataset_card_authors=self.authors,
        )
        card = DatasetCard.from_template(card_data=card_data)
        return card


def _format_generation_stats_section(aggregate: dict) -> str:
    totals = aggregate.get("totals", {}) or {}
    rows = int(totals.get("rows", 0))
    valid = int(totals.get("valid", 0))
    invalid = int(totals.get("invalid", 0))
    total_tokens = int(totals.get("total_tokens", 0))
    sessions = int(aggregate.get("total_sessions", 0))
    runs = int(aggregate.get("total_runs", 0))
    first = (aggregate.get("first_session_at") or "")[:10] or "n/a"
    last = (aggregate.get("last_session_at") or "")[:10] or "n/a"
    return (
        "## Generation Stats\n\n"
        f"- **Total rows:** {rows:,}\n"
        f"- **Valid / Invalid:** {valid:,} / {invalid:,}\n"
        f"- **Total tokens:** {total_tokens:,}\n"
        f"- **Generated across {sessions} session(s) over {runs} run(s)**\n"
        f"- **First session:** {first}\n"
        f"- **Last session:** {last}"
    )
