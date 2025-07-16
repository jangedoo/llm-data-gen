import datasets
from dataclasses import dataclass
from datagen.core import (
    GeneratorRegistry,
    BaseGeneratorConfig,
    BaseDatasetConfig,
    BaseGenerator,
    DataSourceConfig,
    ModelConfig,
)
from datagen.llm import LLM
from typing import Dict, Optional
import re
import json


def extract_and_load_json(text):
    m = re.search(r"```(json)?(.*?)```", text, re.DOTALL | re.I)
    extracted_text = m.group(2) if m else text
    return json.loads(extracted_text)


@dataclass
class QuestionAnswerDatasetConfig(BaseDatasetConfig):
    passage_column: str


@GeneratorRegistry.register(
    name="question_answer",
    description="Generates question-answer pairs from passages",
    default_config={
        "params": {
            "default_model": "gpt-4.1-mini",
            "default_system_prompt": "Given a passage, generate question-answer pairs. Ensure answers are extractive (present in the passage).",
            "source_datasets": [],
        }
    },
)
class QuestionAnswerGenerator(BaseGenerator[QuestionAnswerDatasetConfig]):

    class Config(BaseGeneratorConfig[QuestionAnswerDatasetConfig]):
        @classmethod
        def _create_dataset_config(
            cls,
            src_ds: dict,
            sources_config: Dict[str, DataSourceConfig],
            models_config: Dict[str, ModelConfig],
            model_name: str,
            default_system_prompt: Optional[str],
        ) -> QuestionAnswerDatasetConfig:
            return QuestionAnswerDatasetConfig(
                source_name=src_ds["name"],
                passage_column=src_ds["passage_column"],
                source_config=sources_config[src_ds["name"]],
                model_config=models_config[model_name],
                model_name=model_name,
                system_prompt=src_ds.get("system_prompt", default_system_prompt),
                max_records=src_ds.get("max_records", 100),
                shuffle=src_ds.get("shuffle", False),
                max_failures=src_ds.get("max_failures", 0.5),
            )

    def get_dataset(self, ds_config: QuestionAnswerDatasetConfig) -> datasets.Dataset:
        ds = super().get_dataset(ds_config)
        return ds.select_columns([ds_config.passage_column])

    def is_question_valid(
        self, passage: str, question: Optional[str], answer: Optional[str]
    ) -> tuple[bool, str]:
        if question is None:
            return False, "Question is not present"
        if answer is None:
            return False, "Answer is not present"
        if answer not in passage:
            return False, "Answer is not present in passage"
        if len(question) < 10:
            return False, "Question is too short"
        return True, ""

    def process_dataset(
        self, ds: datasets.Dataset, ds_config: QuestionAnswerDatasetConfig, llm: LLM
    ):
        total_failures = 0
        max_failures = ds_config.max_failures

        for passage in ds[ds_config.passage_column]:
            truncated_passage = passage[:1500]
            messages = self._create_messages(
                system_prompt=ds_config.system_prompt, content=truncated_passage
            )
            llm_response = None
            try:
                llm_response = llm.generate(messages=messages)
                extracted_questions = extract_and_load_json(text=llm_response)
                for question_data in extracted_questions:
                    question = question_data.get("q") or question_data.get("question")
                    answer = question_data.get("a") or question_data.get("answer")
                    answer = answer.strip() if answer else None
                    question = question.strip() if question else None
                    is_valid, reason = self.is_question_valid(
                        passage=passage, question=question, answer=answer
                    )
                    yield {
                        "passage": passage,
                        "question": question,
                        "answer": answer,
                        "__meta": {
                            "dataset_name": ds_config.source_name,
                            "is_valid": is_valid,
                            "reason": reason,
                            "llm_response": llm_response,
                        },
                    }

            except Exception as e:
                total_failures, should_stop = self._handle_failure(
                    e, passage, ds_config, total_failures, max_failures, len(ds)
                )
                yield {
                    "passage": passage,
                    "question": None,
                    "answer": None,
                    "__meta": {
                        "dataset_name": ds_config.source_name,
                        "is_valid": False,
                        "reason": str(e),
                        "llm_response": llm_response,
                    },
                }
                if should_stop:
                    return
