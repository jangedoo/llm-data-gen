import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import tomli_w

from datagen.core.gen_config import AutoModelConfig


DEFAULT_SETTINGS_PATH = Path.home() / ".datagen" / "settings.toml"

OPENAI_PROVIDER_PRESETS = {
    "openai": {"api_base": None, "key_source": "omitted", "env": None, "literal": None},
    "openrouter": {
        "api_base": "https://openrouter.ai/api/v1",
        "key_source": "env",
        "env": "OPENROUTER_API_KEY",
        "literal": None,
    },
    "ollama": {
        "api_base": "http://localhost:11434/v1",
        "key_source": "literal",
        "env": None,
        "literal": "abc",
    },
    "custom": {"api_base": None, "key_source": "omitted", "env": None, "literal": None},
}


def _clean_string(value: Any) -> str:
    return str(value or "").strip()


def _optional_float(
    fields: Mapping[str, Any],
    key: str,
    default: float,
    label: str,
    minimum: float,
    maximum: float,
) -> float:
    raw = _clean_string(fields.get(key))
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{label} must be a number") from exc
    if value < minimum or value > maximum:
        raise ValueError(f"{label} must be between {minimum:g} and {maximum:g}")
    return value


def _optional_int(
    fields: Mapping[str, Any],
    key: str,
    default: int,
    label: str,
    minimum: int,
) -> int:
    raw = _clean_string(fields.get(key))
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def build_model_config_from_fields(fields: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    backend = _clean_string(fields.get("backend")).lower()
    if backend not in {"openai", "dummy"}:
        raise ValueError("Backend must be openai or dummy")

    if backend == "dummy":
        response = _clean_string(fields.get("dummy_response"))
        if not response:
            raise ValueError("Dummy response is required")
        params = {"response": response}
        return backend, params

    model = _clean_string(fields.get("openai_model"))
    if not model:
        raise ValueError("OpenAI model is required")

    params: dict[str, Any] = {
        "model": model,
        "temperature": _optional_float(
            fields, "openai_temperature", 0.3, "Temperature", 0, 2
        ),
        "max_tokens": _optional_int(
            fields, "openai_max_tokens", 1000, "Max tokens", 1
        ),
        "top_p": _optional_float(fields, "openai_top_p", 1, "Top P", 0, 1),
        "frequency_penalty": _optional_float(
            fields, "openai_frequency_penalty", 0, "Frequency penalty", -2, 2
        ),
        "presence_penalty": _optional_float(
            fields, "openai_presence_penalty", 0, "Presence penalty", -2, 2
        ),
    }

    preset_name = _clean_string(fields.get("openai_provider_preset")) or "openai"
    if preset_name not in OPENAI_PROVIDER_PRESETS:
        raise ValueError("Provider preset must be OpenAI, OpenRouter, Ollama/local, or Custom")
    preset = OPENAI_PROVIDER_PRESETS[preset_name]

    api_base = _clean_string(fields.get("openai_api_base"))
    if "openai_api_base" not in fields and preset["api_base"]:
        api_base = str(preset["api_base"])
    if api_base:
        params["api_base"] = api_base

    key_source = _clean_string(fields.get("openai_key_source"))
    if "openai_key_source" not in fields:
        key_source = str(preset["key_source"])
    if key_source not in {"omitted", "env", "literal"}:
        raise ValueError("API key source must be omitted, env, or literal")

    if key_source == "env":
        env_name = _clean_string(fields.get("openai_api_key_env"))
        if "openai_api_key_env" not in fields and preset["env"]:
            env_name = str(preset["env"])
        if not env_name:
            raise ValueError("API key environment variable name is required")
        params["api_key"] = {"env": env_name}
    elif key_source == "literal":
        literal = _clean_string(fields.get("openai_api_key_literal"))
        if "openai_api_key_literal" not in fields and preset["literal"]:
            literal = str(preset["literal"])
        if not literal:
            raise ValueError("Literal API key is required")
        params["api_key"] = literal

    return backend, params


@dataclass
class WebSettings:
    models: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WebSettings":
        models = data.get("models", {})
        if not isinstance(models, dict):
            raise ValueError("settings.models must be a table")
        return cls(models=models)

    def as_dict(self) -> dict[str, Any]:
        return {"models": self.models}


class SettingsStore:
    def __init__(self, path: Path | None = None):
        self.path = path or DEFAULT_SETTINGS_PATH

    def load(self) -> WebSettings:
        if not self.path.exists():
            return WebSettings()
        with self.path.open("rb") as f:
            return WebSettings.from_dict(tomllib.load(f))

    def save(self, settings: WebSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as f:
            tomli_w.dump(settings.as_dict(), f)

    def upsert_model(self, name: str, fields: Mapping[str, Any]) -> WebSettings:
        name = name.strip()
        if not name:
            raise ValueError("Model name is required")

        backend, params = build_model_config_from_fields(fields)
        candidate = {"backend": backend, "params": params}
        AutoModelConfig.from_config(candidate)

        settings = self.load()
        settings.models[name] = candidate
        self.save(settings)
        return settings

    def delete_model(self, name: str) -> WebSettings:
        settings = self.load()
        settings.models.pop(name, None)
        self.save(settings)
        return settings
