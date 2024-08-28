import hashlib

import datasets


def get_md5(text: str):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def clean_texts(row):
    passage = row["passage"].strip()
    question = row["question"].strip()
    id = get_md5(f"{passage}_{question}")
    return dict(id=id, passage=passage, question=question, answer=row["answer"].strip())


def should_keep_record(row, existing_ids: set[str]):
    id = row["id"]

    if id in existing_ids:
        return False

    passage = row["passage"]
    question = row["question"]
    answer = row["answer"]
    if len(question) < 10:
        return False
    if len(answer) > 50:
        return False
    return True


def preprocess_ds(*dss: datasets.Dataset, existing_ds: datasets.Dataset):
    existing_ids = set(existing_ds["id"])
    ds = datasets.concatenate_datasets(list(dss))
    ds = ds.map(clean_texts).filter(
        lambda row: should_keep_record(row, existing_ids=existing_ids)
    )
    return ds
