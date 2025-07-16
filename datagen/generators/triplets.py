from pydantic import BaseModel, Field
from dataclasses import dataclass
from datagen.core import (
    GeneratorRegistry,
    BaseGeneratorConfig,
    BaseDatasetConfig,
    BaseGenerator,
    DataSourceConfig,
    ModelConfig,
)
import datasets
from datagen.llm import LLM


@dataclass
class TripletsDatasetConfig(BaseDatasetConfig):
    sentence_column: str


class TripletResponse(BaseModel):
    positive_sentence: str = Field(
        description="A sentence that is semantically similar to the input sentence"
    )
    negative_sentence: str = Field(
        description="A sentence that is semantically different from the input sentence"
    )


@GeneratorRegistry.register(
    name="triplets",
    description="Generates triplets of sentences",
    default_config={
        "params": {
            "default_model": "gpt-4.1-mini",
            "default_system_prompt": "You will be given a sentence. Generate a triplet of sentences. The first sentence should be semantically similar to the input sentence. The second sentence should be semantically different from the input sentence. The third sentence should be semantically different from the input sentence.",
            "source_datasets": [],
        }
    },
)
class TripletsGenerator(BaseGenerator[TripletsDatasetConfig]):
    class Config(BaseGeneratorConfig[TripletsDatasetConfig]):
        @classmethod
        def _create_dataset_config(
            cls,
            src_ds: dict,
            sources_config: dict[str, DataSourceConfig],
            models_config: dict[str, ModelConfig],
            model_name: str,
            default_system_prompt: str | None,
        ):
            return TripletsDatasetConfig(
                source_name=src_ds["name"],
                sentence_column=src_ds["sentence_column"],
                source_config=sources_config[src_ds["name"]],
                model_name=model_name,
                model_config=models_config[model_name],
                system_prompt=src_ds.get("system_prompt", default_system_prompt),
                max_records=src_ds.get("max_records", 100),
                shuffle=src_ds.get("shuffle", False),
                max_failures=src_ds.get("max_failures", 0.5),
            )

    def process_dataset(
        self, ds: datasets.Dataset, ds_config: TripletsDatasetConfig, llm: LLM
    ):
        total_failures = 0
        max_failures = ds_config.max_failures

        for sentence in ds[ds_config.sentence_column]:
            truncated_sentence = sentence[:1500]
            messages = self._create_messages(
                system_prompt=ds_config.system_prompt, content=truncated_sentence
            )
            try:
                llm_response = llm.generate(
                    messages=messages, response_format=TripletResponse
                )
                if llm_response and isinstance(llm_response, TripletResponse):
                    yield {
                        "sentence": sentence,
                        "positive_sentence": llm_response.positive_sentence,
                        "negative_sentence": llm_response.negative_sentence,
                        "__meta": {
                            "dataset_name": ds_config.source_name,
                            "is_valid": True,
                            "reason": "",
                        },
                    }
                else:
                    raise Exception("Invalid response format")
            except Exception as e:
                total_failures, should_stop = self._handle_failure(
                    e,
                    content=truncated_sentence,
                    ds_config=ds_config,
                    total_failures=total_failures,
                    max_failures=max_failures,
                    dataset_length=len(ds),
                )
                yield {
                    "sentence": sentence,
                    "positive_sentence": None,
                    "negative_sentence": None,
                    "__meta": {
                        "dataset_name": ds_config.source_name,
                        "is_valid": False,
                        "reason": str(e),
                    },
                }
                if should_stop:
                    return
