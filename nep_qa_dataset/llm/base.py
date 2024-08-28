import abc


class LLM(abc.ABC):
    @abc.abstractmethod
    def generate(
        self,
        messages: list[dict],
        response_format: dict = {"type": "text"},
    ) -> str | None:
        raise NotImplementedError()

    def get_usage_stats(self) -> dict:
        completion_tokens = 0
        prompt_tokens = 0
        total_tokens = 0
        num_usage = 0

        return {
            "num_requests": num_usage,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
        }
