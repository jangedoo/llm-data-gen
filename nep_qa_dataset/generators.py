import abc
import json
import logging
import re
from typing import Generic, TypeVar

import datasets

from nep_qa_dataset.gen_config import (
    DataSetConfig,
    GeneratorConfig,
    QuestionAnswerDatasetConfig,
    QuestionAnswerGeneratorConfig,
)
from nep_qa_dataset.llm import LLM

logger = logging.getLogger(__name__)

GeneratorConfigT = TypeVar("GeneratorConfigT", bound=GeneratorConfig)


def extract_and_load_json(text):
    # Use regex to extract the content inside triple backticks
    m = re.search(r"```(json)?(.*?)```", text, re.DOTALL | re.I)
    extracted_text = m.group(2) if m else text
    # Load the extracted text as JSON
    return json.loads(extracted_text)


class Generator(Generic[GeneratorConfigT], abc.ABC):
    def __init__(self, config: GeneratorConfigT):
        self.config = config
        self._llms: dict[str, LLM] = {}

    def _get_llm(self, llm_key: str):
        llm = self._llms.get(llm_key)
        if llm is not None:
            return llm

        llm = self.config.models_config[llm_key].create_llm()
        self._llms[llm_key] = llm
        return llm

    def get_dataset(self, ds_config: DataSetConfig) -> datasets.Dataset:
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

    def process_dataset(self, ds: datasets.Dataset, ds_config: DataSetConfig, llm: LLM):
        return []

    def generate(self):
        for src_ds_config in self.config.source_datasets_config:
            ds = self.get_dataset(ds_config=src_ds_config)
            llm = self._get_llm(llm_key=src_ds_config.model_name)
            yield from self.process_dataset(ds=ds, ds_config=src_ds_config, llm=llm)
            logger.info(
                f"Finished processing dataset {src_ds_config.source_name}. Total llm consumption: {llm.get_usage_stats()}"
            )
        logger.info(f"Finished processing all datasets")


class QuestionAnswerGenerator(Generator[QuestionAnswerGeneratorConfig]):
    def __init__(self, config: QuestionAnswerGeneratorConfig):
        super().__init__(config=config)

    def get_dataset(self, ds_config: QuestionAnswerDatasetConfig):
        ds = super().get_dataset(ds_config)
        return ds.select_columns([ds_config.passage_column])

    def _create_messages(self, system_prompt: str, passage: str) -> list[dict]:
        return [
            dict(role="system", content=system_prompt),
            dict(role="user", content=passage),
        ]

    def is_question_valid(
        self, passage: str, question: str | None, answer: str | None
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
        max_failures = (
            ds_config.max_failures
            if isinstance(ds_config.max_failures, int)
            else int(len(ds) * ds_config.max_failures)
        )

        for passage in ds[ds_config.passage_column]:
            truncated_passage = passage[:1500]
            messages = self._create_messages(
                system_prompt=ds_config.system_prompt, passage=truncated_passage
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
                    yield dict(
                        passage=passage,
                        question=question,
                        answer=answer,
                        __meta=dict(
                            dataset_name=ds_config.source_name,
                            is_valid=is_valid,
                            reason=reason,
                            llm_response=llm_response,
                        ),
                    )

            except Exception as e:
                total_failures += 1
                logger.warning(
                    f"Unable to get response from llm for passage {passage}",
                    exc_info=True,
                )
                yield dict(
                    passage=passage,
                    question=None,
                    answer=None,
                    __meta=dict(
                        dataset_name=ds_config.source_name,
                        is_valid=is_valid,
                        reason=str(e),
                        llm_response=llm_response,
                    ),
                )
                if total_failures >= max_failures:
                    logger.warning(
                        f"Number of failures {total_failures} exceeded maximum allowed failures {max_failures}. Not processing dataset: {ds_config.source_name}"
                    )
                    return
