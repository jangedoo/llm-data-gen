import json
import re
from pathlib import Path
from typing import Any

import tomli_w

from datagen.core.gen_config import GenerationPipelineConfig
from datagen.web.config_utils import CONFIG_DIR, ValidationResult
from datagen.web.settings import WebSettings


CONFIG_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+\.toml$")


def sanitize_config_name(name: str) -> str:
    name = name.strip()
    if not name.endswith(".toml"):
        name = f"{name}.toml"
    if not CONFIG_NAME_RE.match(name) or "/" in name or "\\" in name:
        raise ValueError("Config file name must be a simple .toml file name")
    return name


def parse_builder_payload(payload_json: str) -> dict[str, Any]:
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Builder payload must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Builder payload must be a JSON object")
    return payload


def _clean_list(value: Any) -> list:
    if isinstance(value, str):
        return [item.strip() for item in value.splitlines() if item.strip()]
    if isinstance(value, list):
        return [item for item in value if item not in ("", None)]
    return []


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
    return value.strip()


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _parse_params(value: Any, label: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} params must be valid JSON: {exc}") from exc
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"{label} params must be a JSON object")


def _strip_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_strip_none(item) for item in value]
    return value


def build_config_dict(payload: dict[str, Any], settings: WebSettings) -> dict[str, Any]:
    config: dict[str, Any] = {
        "dataset_name": _required_string(payload.get("dataset_name"), "Dataset name"),
        "description": _optional_string(payload.get("description")) or "",
        "authors": _clean_list(payload.get("authors")),
        "generation_output_dir": _required_string(
            payload.get("generation_output_dir"), "Output directory"
        ),
        "generation_logging_steps": int(payload.get("generation_logging_steps") or 100),
    }

    sources: dict[str, dict[str, Any]] = {}
    for source in payload.get("sources", []):
        source_name = _required_string(source.get("name"), "Source name")
        if source_name in sources:
            raise ValueError(f"Duplicate source name: {source_name}")
        source_config: dict[str, Any] = {
            "path": _required_string(source.get("path"), f"Source {source_name} path"),
            "split": _optional_string(source.get("split")) or "train",
        }
        subset = _optional_string(source.get("subset"))
        if subset:
            source_config["subset"] = subset
        sources[source_name] = source_config
    if not sources:
        raise ValueError("At least one source is required")
    config["sources"] = sources

    models: dict[str, dict[str, Any]] = {}
    for model in payload.get("models", []):
        model_name = _required_string(model.get("name"), "Model name")
        if model.get("kind") == "settings":
            if model_name not in settings.models:
                raise ValueError(f"Settings model not found: {model_name}")
            model_config = settings.models[model_name]
            models[model_name] = {
                "backend": _required_string(
                    model_config.get("backend"), f"Model {model_name} backend"
                ),
                "params": dict(model_config.get("params", {})),
            }
        else:
            models[model_name] = {
                "backend": _required_string(
                    model.get("backend"), f"Model {model_name} backend"
                ),
                "params": _parse_params(model.get("params", {}), model_name),
            }
    if not models:
        raise ValueError("At least one model is required")
    config["models"] = models

    generator_params: dict[str, Any] = {
        "default_model": _required_string(
            payload.get("default_model"), "Default model"
        ),
        "output_template": _required_string(
            payload.get("output_template"), "Output template"
        ),
    }
    if generator_params["default_model"] not in models:
        raise ValueError("Default model must be one of the selected models")

    for field in (
        "default_system_prompt",
        "default_prompt_template",
        "structured_output_schema",
    ):
        value = _optional_string(payload.get(field))
        if value:
            generator_params[field] = value

    source_datasets = []
    for item in payload.get("source_datasets", []):
        source_name = _required_string(item.get("name"), "Source dataset name")
        if source_name not in sources:
            raise ValueError(f"Source dataset references unknown source: {source_name}")
        source_dataset: dict[str, Any] = {
            "name": source_name,
            "max_records": int(item.get("max_records") or 100),
            "shuffle": bool(item.get("shuffle")),
            "max_failures": float(item.get("max_failures") or 0.5),
        }
        model_name = _optional_string(item.get("model"))
        if model_name:
            if model_name not in models:
                raise ValueError(
                    f"Source dataset {source_name} references unknown model: {model_name}"
                )
            source_dataset["model"] = model_name
        for field in ("system_prompt", "prompt_template"):
            value = _optional_string(item.get(field))
            if value:
                source_dataset[field] = value
        source_datasets.append(source_dataset)
    if not source_datasets:
        raise ValueError("At least one source dataset is required")
    generator_params["source_datasets"] = source_datasets

    aliases = []
    for alias in payload.get("aliases", []):
        source_name = _required_string(alias.get("source"), "Alias source")
        if source_name not in sources:
            raise ValueError(f"Alias references unknown source: {source_name}")
        column_map = alias.get("column_map") or {}
        if not isinstance(column_map, dict):
            raise ValueError("Alias column_map must be an object")
        if column_map:
            aliases.append({"source": source_name, "column_map": column_map})
    if aliases:
        generator_params["aliases"] = aliases

    config["generator"] = {"generator": "templated", "params": generator_params}

    curator = payload.get("curator", {})
    config["curator"] = {
        "params": {
            "upload_to_hf": bool(curator.get("upload_to_hf")),
            "upload_repo_id": _optional_string(curator.get("upload_repo_id")),
            "train_test_split": bool(curator.get("train_test_split", True)),
            "update_card": bool(curator.get("update_card", True)),
            "language": _clean_list(curator.get("language")),
            "license": _optional_string(curator.get("license")) or "mit",
            "task_categories": _clean_list(curator.get("task_categories")),
            "task_ids": _clean_list(curator.get("task_ids")),
        }
    }
    citation_bibtex = _optional_string(curator.get("citation_bibtex"))
    if citation_bibtex:
        config["curator"]["params"]["citation_bibtex"] = citation_bibtex

    return _strip_none(config)


def config_to_toml(config: dict[str, Any]) -> str:
    return tomli_w.dumps(config)


def validate_builder_config(config: dict[str, Any]) -> ValidationResult:
    try:
        parsed = GenerationPipelineConfig.from_dict(
            config, base_dir=CONFIG_DIR, create_output_dir=False
        )
    except Exception as exc:
        return ValidationResult(ok=False, errors=[str(exc)])
    return ValidationResult(ok=True, errors=[], config=parsed)


def payload_from_config_dict(
    config: dict[str, Any], settings: WebSettings
) -> dict[str, Any]:
    generator_params = config.get("generator", {}).get("params", {})
    curator = config.get("curator", {}).get("params", {})

    models = []
    for model_name, model_config in config.get("models", {}).items():
        settings_model = settings.models.get(model_name)
        if settings_model == model_config:
            models.append({"kind": "settings", "name": model_name})
        else:
            models.append(
                {
                    "kind": "custom",
                    "name": model_name,
                    "backend": model_config.get("backend", ""),
                    "params": model_config.get("params", {}),
                }
            )

    return {
        "dataset_name": config.get("dataset_name", ""),
        "description": config.get("description", ""),
        "authors": config.get("authors", []),
        "generation_output_dir": config.get("generation_output_dir", ""),
        "generation_logging_steps": config.get("generation_logging_steps", 100),
        "sources": [
            {"name": source_name, **source_config}
            for source_name, source_config in config.get("sources", {}).items()
        ],
        "models": models,
        "default_model": generator_params.get("default_model", ""),
        "default_system_prompt": generator_params.get("default_system_prompt", ""),
        "default_prompt_template": generator_params.get("default_prompt_template", ""),
        "output_template": generator_params.get("output_template", ""),
        "structured_output_schema": generator_params.get(
            "structured_output_schema", ""
        ),
        "source_datasets": generator_params.get("source_datasets", []),
        "aliases": generator_params.get("aliases", []),
        "curator": {
            "upload_to_hf": curator.get("upload_to_hf", False),
            "upload_repo_id": curator.get("upload_repo_id", ""),
            "train_test_split": curator.get("train_test_split", True),
            "update_card": curator.get("update_card", True),
            "language": curator.get("language", []),
            "license": curator.get("license", "mit"),
            "task_categories": curator.get("task_categories", []),
            "task_ids": curator.get("task_ids", []),
            "citation_bibtex": curator.get("citation_bibtex", ""),
        },
    }


def write_builder_config(
    name: str,
    payload_json: str,
    settings: WebSettings,
    overwrite: bool = False,
) -> tuple[Path, ValidationResult, str]:
    config_name = sanitize_config_name(name)
    config_path = (CONFIG_DIR / config_name).resolve()
    if config_path.parent != CONFIG_DIR.resolve():
        raise ValueError("Config must be written under gen_configs/")
    if config_path.exists() and not overwrite:
        raise ValueError(f"{config_name} already exists")

    payload = parse_builder_payload(payload_json)
    config = build_config_dict(payload, settings)
    validation = validate_builder_config(config)
    toml_text = config_to_toml(config)
    if not validation.ok:
        return config_path, validation, toml_text

    CONFIG_DIR.mkdir(exist_ok=True)
    config_path.write_text(toml_text, encoding="utf-8")
    return config_path, validation, toml_text
