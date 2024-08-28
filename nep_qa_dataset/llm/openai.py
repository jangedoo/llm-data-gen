import openai

from nep_qa_dataset.llm.base import LLM


class OpenAILLM(LLM):
    def __init__(
        self,
        client: openai.Client,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens=1000,
        top_p: float = 1,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
    ):
        self.client = client
        self.model = model

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

        self.usage_history: list[openai.types.completion_usage.CompletionUsage] = []

    def generate(
        self,
        messages: list[dict],
        response_format: dict = {"type": "text"},
    ) -> str | None:
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            top_p=self.top_p,
            response_format=response_format,
        )
        if response.usage:
            self.usage_history.append(response.usage)
        return response.choices[0].message.content

    def get_usage_stats(self) -> dict:
        completion_tokens = 0
        prompt_tokens = 0
        total_tokens = 0
        num_usage = len(self.usage_history)
        for usage in self.usage_history:
            completion_tokens += usage.completion_tokens
            prompt_tokens += usage.prompt_tokens
            total_tokens += usage.total_tokens

        return {
            "num_requests": num_usage,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
        }
