import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomli_w


DEFAULT_SETTINGS_PATH = Path.home() / ".datagen" / "settings.toml"


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

    def upsert_model(self, name: str, backend: str, params_json: str) -> WebSettings:
        name = name.strip()
        backend = backend.strip()
        if not name:
            raise ValueError("Model name is required")
        if not backend:
            raise ValueError("Backend is required")

        try:
            params = json.loads(params_json or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model params must be valid JSON: {exc}") from exc
        if not isinstance(params, dict):
            raise ValueError("Model params must be a JSON object")

        settings = self.load()
        settings.models[name] = {"backend": backend, "params": params}
        self.save(settings)
        return settings

    def delete_model(self, name: str) -> WebSettings:
        settings = self.load()
        settings.models.pop(name, None)
        self.save(settings)
        return settings
