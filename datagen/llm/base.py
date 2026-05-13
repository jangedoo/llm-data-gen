import abc
import time
from typing import Any
from pydantic import BaseModel

from datagen.core.stats import CallStats


class LLM(abc.ABC):
    @abc.abstractmethod
    def generate(
        self,
        messages: list[dict],
        response_format: dict | type[BaseModel] | None = None,
    ) -> str | BaseModel | None:
        raise NotImplementedError()

    def generate_with_stats(
        self,
        messages: list[dict],
        response_format: dict | type[BaseModel] | None = None,
    ) -> tuple[Any, CallStats]:
        """Generate and return per-call timing/token stats.

        Default implementation times ``generate()`` and returns zeroed token
        counts. Subclasses with token info (e.g. OpenAI) should override.
        """
        start = time.perf_counter()
        out = self.generate(messages=messages, response_format=response_format)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return out, CallStats(
            latency_ms=elapsed_ms,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

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
