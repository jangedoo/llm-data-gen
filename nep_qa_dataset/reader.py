import abc
import json
import logging
from pathlib import Path

import datasets
import pandas as pd

logger = logging.getLogger(__name__)


class RawDataReader(abc.ABC):
    @abc.abstractmethod
    def read(self) -> pd.DataFrame:
        raise NotImplementedError()

    def read_as_dataset(self):
        return datasets.Dataset.from_pandas(self.read())


class CSVReader(RawDataReader):
    def __init__(self, file_path: Path | str):
        self.file_path = Path(file_path)

    def read(self) -> pd.DataFrame:
        df = pd.read_csv(self.file_path)

        expected_col_names = ["passage", "question", "answer"]
        if not all(col in df.columns for col in expected_col_names):
            raise ValueError(
                f"csv file {self.file_path} must contain {expected_col_names} columns but has {df.columns.tolist()}"
            )

        return df


class JSONLReader(RawDataReader):
    def __init__(self, file_path: Path | str):
        self.file_path = Path(file_path)

    def read(self) -> pd.DataFrame:
        rows = []
        with open(self.file_path, "r") as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                except:
                    logger.warning(f"unable to parse line {i+1} as json: {line}")
                if not {"passage", "questions"}.issubset(data.keys()):
                    logger.warning(
                        f"passage and question must be present but was not found in line {i+1}: {line}"
                    )
                    continue
                for question_data in data["questions"]:
                    if not {"question", "answer"}.issubset(question_data.keys()):
                        logger.warning(
                            f"question and answer must be present but was not found in line {i+1}: {line}"
                        )
                        continue

                    question = question_data["question"].strip()
                    answer = question_data["answer"].strip()

                    rows.append(
                        dict(passage=data["passage"], question=question, answer=answer)
                    )

        return pd.DataFrame(rows)
