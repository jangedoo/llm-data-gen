from pydantic import BaseModel
from litellm import responses, ResponseTextConfigParam, ResponsesAPIResponse
import litellm
from datagen.llm.base import LLM
import logging

logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)

# https://docs.litellm.ai/docs/completion/json_mode#validate-json-schema
litellm.enable_json_schema_validation = True


class LiteLLM(LLM):
    def __init__(
        self,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 4000,
        top_p: float = 1,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.usage_history: list = []

    def generate(
        self,
        messages: list[dict],
        response_format: dict | type[BaseModel] | None = None,
    ) -> str | BaseModel | None:
        response_text_config: ResponseTextConfigParam = {
            "format": {"type": "text"},
        }

        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            model_schema = response_format.model_json_schema()
            model_schema["additionalProperties"] = False
            response_text_config: ResponseTextConfigParam = {
                "format": {
                    "type": "json_schema",
                    "name": response_format.__class__.__name__,
                    "schema": model_schema,
                },
            }

        response: ResponsesAPIResponse = responses(
            model=self.model,
            input=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            text=response_text_config,
        )  # type: ignore

        self.usage_history.append(response.usage)
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            return response_format.model_validate_json(
                response.output[0].content[0].text
            )
        else:
            return response.output[0].content[0].text

    def get_usage_stats(self) -> dict:
        completion_tokens = 0
        prompt_tokens = 0
        total_tokens = 0
        num_usage = len(self.usage_history)
        for usage in self.usage_history:
            if usage is None:
                continue
            completion_tokens += getattr(usage, "output_tokens", 0)
            prompt_tokens += getattr(usage, "input_tokens", 0)
            total_tokens += getattr(usage, "total_tokens", 0)

        return {
            "num_requests": num_usage,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
        }
