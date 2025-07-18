import json
import re
from typing import Any, Dict, Iterator, List, Optional, Set
from datasets.arrow_dataset import Dataset
from pydantic import BaseModel
from dataclasses import dataclass
from jinja2 import Environment, Template, TemplateSyntaxError
from datagen.core import (
    BaseGeneratorConfig,
    BaseGenerator,
    DataSourceConfig,
    ModelConfig,
)
from datagen.core.gen_config import DataSetConfig
from datagen.llm import LLM


@dataclass
class TemplatedDatasetConfig(DataSetConfig):
    prompt_template: str | None = None
    aliases: Dict[str, str] | None = None


class TemplatedGenerator(BaseGenerator[TemplatedDatasetConfig]):
    class Config(BaseGeneratorConfig[TemplatedDatasetConfig]):
        def __init__(
            self,
            default_model: str,
            sources_config: dict[str, DataSourceConfig],
            models_config: dict[str, ModelConfig],
            source_datasets_config: list[TemplatedDatasetConfig],
            output_template: str,
            default_system_prompt: str | None = None,
            default_prompt_template: str | None = None,
            structured_output_schema: str | None = None,
            aliases: dict[str, dict[str, str]] | None = None,
        ):
            super().__init__(
                default_model,
                sources_config,
                models_config,
                source_datasets_config,
                default_system_prompt,
            )
            self.output_template = output_template
            self.default_prompt_template = default_prompt_template
            self.structured_output_schema = structured_output_schema
            self.aliases = aliases or {}

            # Setup Jinja2 environment
            self.jinja_env = Environment()

            # Setup structured output class if provided
            self.structured_output_cls = None
            if self.structured_output_schema:
                try:
                    # Create a safe environment for exec with necessary imports
                    import builtins

                    safe_globals = {
                        "BaseModel": BaseModel,
                        "__builtins__": dict(
                            builtins.__dict__
                        ),  # More permissive but safer than full access
                    }
                    # Import common modules that might be needed
                    try:
                        from typing import List, Dict, Optional, Union
                        from pydantic import Field

                        safe_globals.update(
                            {
                                "List": List,
                                "Dict": Dict,
                                "Optional": Optional,
                                "Union": Union,
                                "Field": Field,
                            }
                        )
                    except ImportError:
                        pass

                    # Execute the schema definition
                    exec(self.structured_output_schema, safe_globals)

                    # Find the defined class (look for BaseModel subclasses)
                    for name, obj in safe_globals.items():
                        if (
                            isinstance(obj, type)
                            and issubclass(obj, BaseModel)
                            and obj != BaseModel
                        ):
                            self.structured_output_cls = obj
                            break

                    if self.structured_output_cls is None:
                        raise ValueError(
                            "No BaseModel subclass found in structured_output_schema"
                        )

                except Exception as e:
                    raise ValueError(f"Invalid structured_output_schema: {e}")

            # Validate configuration
            self._validate_config()

        def _validate_config(self):
            """Validate the configuration for consistency and correctness."""
            # Validate output template
            try:
                self.jinja_env.from_string(self.output_template)
            except TemplateSyntaxError as e:
                raise ValueError(f"Invalid output_template syntax: {e}")

            # Validate default prompt template if provided
            if self.default_prompt_template:
                try:
                    self.jinja_env.from_string(self.default_prompt_template)
                except TemplateSyntaxError as e:
                    raise ValueError(f"Invalid default_prompt_template syntax: {e}")

            # Validate each dataset configuration
            for ds_config in self.source_datasets_config:
                self._validate_dataset_config(ds_config)

        def _validate_dataset_config(self, ds_config: TemplatedDatasetConfig):
            """Validate a single dataset configuration."""
            source_name = ds_config.source_name

            # Check if prompt template is provided (either default or per-dataset)
            prompt_template = ds_config.prompt_template or self.default_prompt_template
            if not prompt_template:
                raise ValueError(
                    f"No prompt_template defined for dataset '{source_name}' and no default_prompt_template provided"
                )

            # Validate prompt template syntax
            try:
                template = self.jinja_env.from_string(prompt_template)
            except TemplateSyntaxError as e:
                raise ValueError(
                    f"Invalid prompt_template syntax for dataset '{source_name}': {e}"
                )

            # Extract template variables
            template_vars = self._extract_template_variables(prompt_template)

            # Validate aliases if provided
            if ds_config.aliases:
                # Check that all aliased columns are used in templates
                for alias_name, original_column in ds_config.aliases.items():
                    if f"input.{alias_name}" not in template_vars:
                        print(
                            f"Warning: Alias '{alias_name}' defined but not used in template for dataset '{source_name}'"
                        )

            # Validate that template variables have corresponding aliases or columns
            # We'll do this validation at runtime when we have the actual dataset

        def _extract_template_variables(self, template_str: str) -> Set[str]:
            """Extract all template variables from a Jinja2 template."""
            # Simple regex-based extraction instead of AST parsing
            variables = set()
            for var in re.findall(r"\{\{\s*([^}]+)\s*\}\}", template_str):
                var = var.strip()
                # Extract the main variable name (before any filters or operations)
                main_var = var.split("|")[0].split(".")[0].strip()
                variables.add(var.strip())
            return variables

        @classmethod
        def from_config(
            cls,
            generator_config: dict,
            sources_config: Dict[str, DataSourceConfig],
            models_config: Dict[str, ModelConfig],
        ):
            params = generator_config.get("params", {})

            default_model = params.get("default_model")
            if not default_model:
                raise ValueError("`default_model` must be set under params")

            if default_model not in models_config:
                raise ValueError(
                    f"`{default_model}` model has not been defined in `models`"
                )

            source_datasets_config = params.get("source_datasets", [])
            if not source_datasets_config:
                raise ValueError(
                    "At least one source_datasets must be defined in generator"
                )

            default_system_prompt = params.get("default_system_prompt")
            output_template = params.get("output_template")
            if not output_template:
                raise ValueError("output_template must be defined in generator.params")

            default_prompt_template = params.get("default_prompt_template")
            structured_output_schema = params.get("structured_output_schema")

            # Parse aliases from config
            aliases = {}
            for alias_config in params.get("aliases", []):
                source = alias_config.get("source")
                column_map = alias_config.get("column_map", {})
                if source and column_map:
                    aliases[source] = column_map

            dataset_configs = []
            for src_ds in source_datasets_config:
                if src_ds["name"] not in sources_config:
                    raise ValueError(
                        f"source_dataset with name {src_ds['name']} is not defined in sources section."
                    )

                sys_prompt = src_ds.get("system_prompt")
                if sys_prompt is None and default_system_prompt is None:
                    raise ValueError(
                        f"Either system_prompt for dataset `{src_ds['name']}` or `default_system_prompt` must be defined in generator.params"
                    )

                model_name = src_ds.get("model", default_model)
                dataset_config = cls._create_dataset_config(
                    src_ds=src_ds,
                    sources_config=sources_config,
                    models_config=models_config,
                    model_name=model_name,
                    default_system_prompt=default_system_prompt,
                    aliases=aliases.get(src_ds["name"]),
                )
                dataset_configs.append(dataset_config)

            return cls(
                default_model=default_model,
                sources_config=sources_config,
                models_config=models_config,
                source_datasets_config=dataset_configs,
                default_system_prompt=default_system_prompt,
                output_template=output_template,
                default_prompt_template=default_prompt_template,
                structured_output_schema=structured_output_schema,
                aliases=aliases,
            )

        @classmethod
        def _create_dataset_config(
            cls,
            src_ds: dict,
            sources_config: dict[str, DataSourceConfig],
            models_config: dict[str, ModelConfig],
            model_name: str,
            default_system_prompt: str | None,
            aliases: dict[str, str] | None = None,
        ):
            return TemplatedDatasetConfig(
                source_name=src_ds["name"],
                source_config=sources_config[src_ds["name"]],
                model_name=model_name,
                model_config=models_config[model_name],
                system_prompt=src_ds.get("system_prompt", default_system_prompt),
                prompt_template=src_ds.get("prompt_template", None),
                max_records=src_ds.get("max_records", 100),
                shuffle=src_ds.get("shuffle", False),
                max_failures=src_ds.get("max_failures", 0.5),
                aliases=aliases,
            )

    def _apply_aliases(
        self, row: Dict[str, Any], aliases: Dict[str, str] | None
    ) -> Dict[str, Any]:
        """Apply column aliases to normalize the row data."""
        if not aliases:
            return row

        normalized_row = dict(row)  # Copy original row

        # Add aliased columns
        for alias_name, original_column in aliases.items():
            if original_column in row:
                normalized_row[alias_name] = row[original_column]
            else:
                print(
                    f"Warning: Original column '{original_column}' not found in row for alias '{alias_name}'"
                )

        return normalized_row

    def _render_template(self, template_str: str, context: Dict[str, Any]) -> str:
        """Render a Jinja2 template with the given context."""
        try:
            # Access the jinja_env from our specific config class
            jinja_env = getattr(self.config, "jinja_env")
            template = jinja_env.from_string(template_str)
            return template.render(**context)
        except Exception as e:
            raise ValueError(f"Template rendering failed: {e}")

    def _validate_row_against_template(
        self, row: Dict[str, Any], template_str: str, source_name: str
    ):
        """Validate that the row has all necessary fields for the template."""
        # Access the method from our specific config class
        extract_vars_method = getattr(self.config, "_extract_template_variables")
        template_vars = extract_vars_method(template_str)

        for var in template_vars:
            if var.startswith("input."):
                field_name = var[6:]  # Remove 'input.' prefix
                if field_name not in row:
                    raise ValueError(
                        f"Template variable '{var}' used in dataset '{source_name}' but field '{field_name}' not found in row"
                    )

    def _format_output(
        self,
        input_data: Dict[str, Any],
        llm_output: Any,
        ds_config: TemplatedDatasetConfig,
    ) -> Dict[str, Any]:
        """Format the final output using the output template."""
        context = {
            "input": input_data,
            "llm_output": llm_output,
        }

        try:
            # Access the output_template from our specific config class
            output_template = getattr(self.config, "output_template")
            formatted_output = self._render_template(output_template, context)
            # Try to parse as JSON if it looks like JSON
            formatted_output = formatted_output.strip()
            if (
                formatted_output.startswith("{") and formatted_output.endswith("}")
            ) or (formatted_output.startswith("[") and formatted_output.endswith("]")):
                try:
                    return json.loads(formatted_output)
                except json.JSONDecodeError:
                    pass

            # If not JSON, return as structured data
            return {
                "formatted_output": formatted_output,
                "__meta": {
                    "dataset_name": ds_config.source_name,
                    "is_valid": True,
                    "reason": "",
                },
            }
        except Exception as e:
            return {
                "error": str(e),
                "__meta": {
                    "dataset_name": ds_config.source_name,
                    "is_valid": False,
                    "reason": f"Output formatting failed: {e}",
                },
            }

    def process_dataset(
        self, ds: Dataset, ds_config: TemplatedDatasetConfig, llm: LLM
    ) -> Iterator[Dict[str, Any]]:
        total_failures = 0
        max_failures = ds_config.max_failures

        # Determine which template to use
        default_prompt_template = getattr(self.config, "default_prompt_template", None)
        prompt_template = ds_config.prompt_template or default_prompt_template
        if not prompt_template:
            raise ValueError(
                f"No prompt template available for dataset {ds_config.source_name}"
            )

        for row in ds:
            try:
                # Apply aliases to normalize the row data
                row_dict = dict(row)  # Convert to Dict[str, Any]
                normalized_row = self._apply_aliases(row_dict, ds_config.aliases)

                # Validate that the row has required fields for the template
                self._validate_row_against_template(
                    normalized_row, prompt_template, ds_config.source_name
                )

                # Render the prompt template
                context = {"input": normalized_row}
                prompt = self._render_template(prompt_template, context)

                # Create messages for LLM
                messages = self._create_messages(
                    system_prompt=ds_config.system_prompt, content=prompt
                )

                # Generate response from LLM
                structured_output_cls = getattr(
                    self.config, "structured_output_cls", None
                )
                llm_response = llm.generate(
                    messages=messages,
                    response_format=structured_output_cls,
                )

                # Format the output
                result = self._format_output(normalized_row, llm_response, ds_config)

                # Ensure __meta exists
                if "__meta" not in result:
                    result["__meta"] = {
                        "dataset_name": ds_config.source_name,
                        "is_valid": True,
                        "reason": "",
                    }

                yield result

            except Exception as e:
                total_failures, should_stop = self._handle_failure(
                    e,
                    content=str(row)[:100],
                    ds_config=ds_config,
                    total_failures=total_failures,
                    max_failures=max_failures,
                    dataset_length=len(ds),
                )

                yield {
                    "input_data": row,
                    "output": None,
                    "__meta": {
                        "dataset_name": ds_config.source_name,
                        "is_valid": False,
                        "reason": str(e),
                    },
                }

                if should_stop:
                    return
