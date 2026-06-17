import json
from dataclasses import dataclass
from typing import Any

from jinja2 import Environment, TemplateSyntaxError, nodes


@dataclass
class TemplateReferences:
    input_fields: set[str]
    uses_llm_output: bool
    errors: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_fields": sorted(self.input_fields),
            "uses_llm_output": self.uses_llm_output,
            "errors": self.errors,
        }


def inspect_template(template: str) -> TemplateReferences:
    env = Environment()
    try:
        parsed = env.parse(template or "")
    except TemplateSyntaxError as exc:
        return TemplateReferences(
            input_fields=[],
            uses_llm_output=False,
            errors=[f"Template syntax error: {exc}"],
        )

    fields: set[str] = set()
    uses_llm_output = False
    for node in parsed.find_all(nodes.Getattr):
        if isinstance(node.node, nodes.Name) and node.node.name == "input":
            fields.add(node.attr)
    for node in parsed.find_all(nodes.Getitem):
        if not (isinstance(node.node, nodes.Name) and node.node.name == "input"):
            continue
        if isinstance(node.arg, nodes.Const) and isinstance(node.arg.value, str):
            fields.add(node.arg.value)
    for node in parsed.find_all(nodes.Name):
        if node.name == "llm_output":
            uses_llm_output = True
    return TemplateReferences(
        input_fields=fields,
        uses_llm_output=uses_llm_output,
        errors=[],
    )


def _source_columns(source: dict[str, Any]) -> list[str]:
    columns = source.get("columns", [])
    if isinstance(columns, list):
        return [str(column) for column in columns if str(column).strip()]
    return []


def _alias_map_for_source(payload: dict[str, Any], source_name: str) -> dict[str, str]:
    for alias in payload.get("aliases", []):
        if alias.get("source") != source_name:
            continue
        column_map = alias.get("column_map") or {}
        if isinstance(column_map, dict):
            return {str(k): str(v) for k, v in column_map.items()}
    return {}


def _template_for_source(payload: dict[str, Any], source_name: str) -> str:
    default_template = str(payload.get("default_prompt_template") or "")
    for item in payload.get("source_datasets", []):
        if item.get("name") != source_name:
            continue
        return str(item.get("prompt_template") or default_template)
    return default_template


def build_template_context(payload: dict[str, Any]) -> dict[str, Any]:
    sources = []
    for source in payload.get("sources", []):
        source_name = str(source.get("name") or "")
        if not source_name:
            continue
        columns = _source_columns(source)
        aliases = _alias_map_for_source(payload, source_name)
        available = sorted(set(columns) | set(aliases.keys()))
        prompt_refs = inspect_template(_template_for_source(payload, source_name))
        missing = sorted(field for field in prompt_refs.input_fields if field not in available)
        unused_aliases = sorted(alias for alias in aliases if alias not in prompt_refs.input_fields)
        alias_errors = [
            f"Alias input.{alias} points to missing source column '{column}'"
            for alias, column in aliases.items()
            if columns and column not in columns
        ]
        sources.append(
            {
                "name": source_name,
                "columns": columns,
                "aliases": aliases,
                "available_input_fields": available,
                "prompt": prompt_refs.as_dict(),
                "missing_input_fields": missing,
                "unused_aliases": unused_aliases,
                "errors": prompt_refs.errors + alias_errors,
            }
        )

    output_refs = inspect_template(str(payload.get("output_template") or ""))
    return {
        "sources": sources,
        "output": output_refs.as_dict(),
        "globals": ["llm_output"],
        "sample_output_context": json.dumps(
            {"input": {"field": "value"}, "llm_output": "model response"},
            indent=2,
        ),
    }
