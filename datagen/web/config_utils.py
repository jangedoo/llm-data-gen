import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets

from datagen.core.gen_config import GenerationPipelineConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "gen_configs"


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]
    config: GenerationPipelineConfig | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors}


def ensure_config_dir() -> Path:
    CONFIG_DIR.mkdir(exist_ok=True)
    return CONFIG_DIR


def list_config_files() -> list[Path]:
    return sorted(ensure_config_dir().glob("*.toml"))


def resolve_config_path(name: str) -> Path:
    candidate = (ensure_config_dir() / name).resolve()
    config_dir = ensure_config_dir().resolve()
    if candidate.parent != config_dir or candidate.suffix != ".toml":
        raise ValueError("Config must be a TOML file under gen_configs/")
    if not candidate.exists():
        raise FileNotFoundError(f"Config not found: {name}")
    return candidate


def load_config_dict(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def validate_config_path(path: Path) -> ValidationResult:
    try:
        config = GenerationPipelineConfig.from_path(path, create_output_dir=False)
    except Exception as exc:
        return ValidationResult(ok=False, errors=[str(exc)])
    return ValidationResult(ok=True, errors=[], config=config)


def get_config_summary(path: Path) -> dict[str, Any]:
    config_dict = load_config_dict(path)
    validation = validate_config_path(path)
    generator = config_dict.get("generator", {})
    params = generator.get("params", {})
    return {
        "name": path.name,
        "dataset_name": config_dict.get("dataset_name", path.stem),
        "description": config_dict.get("description", ""),
        "authors": config_dict.get("authors", []),
        "generation_output_dir": config_dict.get("generation_output_dir", ""),
        "sources": list(config_dict.get("sources", {}).keys()),
        "models": list(config_dict.get("models", {}).keys()),
        "generator": generator.get("generator"),
        "default_model": params.get("default_model"),
        "source_datasets": params.get("source_datasets", []),
        "valid": validation.ok,
        "errors": validation.errors,
    }


def render_prompt_from_row(
    path: Path,
    source_name: str | None,
    raw_row: str,
) -> dict[str, Any]:
    validation = validate_config_path(path)
    if not validation.ok or validation.config is None:
        return {"ok": False, "errors": validation.errors}

    try:
        row = json.loads(raw_row) if raw_row.strip() else {}
        if not isinstance(row, dict):
            raise ValueError("Row JSON must be an object")
    except Exception as exc:
        return {"ok": False, "errors": [f"Invalid row JSON: {exc}"]}

    from datagen.generators.templated import TemplatedGenerator

    generator = TemplatedGenerator(config=validation.config.generator_config)
    dataset_configs = validation.config.generator_config.source_datasets_config
    selected = None
    for dataset_config in dataset_configs:
        if source_name in (None, "", dataset_config.source_name):
            selected = dataset_config
            break
    if selected is None:
        return {"ok": False, "errors": [f"Unknown source dataset: {source_name}"]}

    prompt_template = (
        selected.prompt_template
        or validation.config.generator_config.default_prompt_template
    )
    if not prompt_template:
        return {"ok": False, "errors": ["No prompt template available"]}

    try:
        normalized_row = generator._apply_aliases(row, selected.aliases)
        generator._validate_row_against_template(
            normalized_row, prompt_template, selected.source_name
        )
        prompt = generator._render_template(prompt_template, {"input": normalized_row})
    except Exception as exc:
        return {"ok": False, "errors": [str(exc)]}

    return {
        "ok": True,
        "errors": [],
        "source_name": selected.source_name,
        "system_prompt": selected.system_prompt,
        "prompt": prompt,
        "normalized_row": normalized_row,
    }


def preview_source_dataset(
    path: Path, source_name: str | None = None, limit: int = 5
) -> dict[str, Any]:
    validation = validate_config_path(path)
    if not validation.ok or validation.config is None:
        return {"ok": False, "errors": validation.errors}

    dataset_configs = validation.config.generator_config.source_datasets_config
    selected = None
    for dataset_config in dataset_configs:
        if source_name in (None, "", dataset_config.source_name):
            selected = dataset_config
            break
    if selected is None:
        return {"ok": False, "errors": [f"Unknown source dataset: {source_name}"]}

    try:
        ds = selected.source_config.create_dataset()
        rows = [dict(row) for row in ds.select(range(min(limit, len(ds))))]
    except Exception as exc:
        return {"ok": False, "errors": [str(exc)]}

    return {
        "ok": True,
        "errors": [],
        "source_name": selected.source_name,
        "rows": rows,
    }


def _display_value(value: Any, max_chars: int = 120) -> dict[str, str]:
    if isinstance(value, (dict, list)):
        full = json.dumps(value, ensure_ascii=False)
    else:
        full = "" if value is None else str(value)
    display = full if len(full) <= max_chars else f"{full[: max_chars - 1]}…"
    return {"display": display, "full": full}


def _feature_metadata(features: Any, columns: list[str]) -> list[dict[str, str]]:
    metadata = []
    for column in columns:
        feature = features.get(column) if hasattr(features, "get") else None
        dtype = getattr(feature, "dtype", None) or getattr(feature, "_type", None) or ""
        metadata.append(
            {
                "name": column,
                "type": str(dtype) if dtype else type(feature).__name__,
                "feature": str(feature) if feature is not None else "",
            }
        )
    return metadata


def _dataset_metadata(
    ds: Any, path: str, subset: str | None, split: str
) -> dict[str, Any]:
    info = getattr(ds, "info", None)
    metadata = {
        "path": path,
        "subset": subset or "",
        "split": split,
        "num_rows": len(ds),
        "num_columns": len(getattr(ds, "column_names", []) or []),
    }
    if info is None:
        return metadata

    for key in (
        "dataset_name",
        "config_name",
        "builder_name",
        "version",
        "license",
        "homepage",
        "description",
        "citation",
    ):
        value = getattr(info, key, None)
        if value:
            metadata[key] = str(value)
    return metadata


def preview_hf_source(
    path: str,
    subset: str | None = None,
    split: str = "train",
    limit: int = 5,
) -> dict[str, Any]:
    if not path.strip():
        return {"ok": False, "errors": ["Dataset path is required"]}

    try:
        ds = datasets.load_dataset(path, subset or None, split=split or "train")  # type: ignore
        rows = [dict(row) for row in ds.select(range(min(limit, len(ds))))]
        columns = list(
            getattr(ds, "column_names", []) or (rows[0].keys() if rows else [])
        )
        table_rows = [
            {column: _display_value(row.get(column)) for column in columns}
            for row in rows
        ]
        column_metadata = _feature_metadata(getattr(ds, "features", {}), columns)
        dataset_metadata = _dataset_metadata(ds, path, subset, split or "train")
    except Exception as exc:
        return {"ok": False, "errors": [str(exc)]}

    return {
        "ok": True,
        "errors": [],
        "columns": columns,
        "rows": rows,
        "table_rows": table_rows,
        "column_metadata": column_metadata,
        "dataset_metadata": dataset_metadata,
    }


def resolve_output_dir(path: Path) -> Path:
    validation = validate_config_path(path)
    if not validation.ok or validation.config is None:
        raise ValueError("; ".join(validation.errors))
    return validation.config.generation_output_dir


def inspect_jsonl_file(path: Path, limit: int = 20) -> dict[str, Any]:
    rows = []
    invalid_reasons = []
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
                if len(rows) < limit:
                    rows.append(row)
                meta = row.get("__meta", {})
                if meta.get("is_valid") is False and meta.get("reason"):
                    invalid_reasons.append(str(meta["reason"]))
            except Exception as exc:
                invalid_reasons.append(f"line {line_number}: {exc}")
    return {
        "name": path.name,
        "path": str(path),
        "rows": total,
        "preview": rows,
        "invalid_reasons": invalid_reasons[:10],
    }


def list_outputs_for_config(path: Path) -> dict[str, Any]:
    output_dir = resolve_output_dir(path)
    files = []
    if output_dir.exists():
        for jsonl_path in sorted(output_dir.glob("**/*.jsonl"), reverse=True):
            if jsonl_path.is_file():
                files.append(inspect_jsonl_file(jsonl_path, limit=3))
    return {"output_dir": output_dir, "files": files}
