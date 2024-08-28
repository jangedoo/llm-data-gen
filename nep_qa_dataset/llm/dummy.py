from nep_qa_dataset.llm.base import LLM


class DummyLLM(LLM):
    def __init__(self, response: str):
        self.response = response

    def generate(
        self, messages: list[dict], response_format: dict = dict()
    ) -> str | None:
        return self.response
