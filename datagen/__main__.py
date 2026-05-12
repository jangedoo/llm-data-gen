import logging
from pathlib import Path

import click
import tomli_w

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


def write_toml(data, file_path):
    """Write data to TOML file"""
    with open(file_path, "wb") as f:
        tomli_w.dump(data, f)


@click.group()
def cli():
    """Nep QA Dataset Generation Tool"""
    pass


@cli.command()
@click.option(
    "--config_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the TOML configuration file",
)
def generate(config_file: Path):
    """Generate dataset using the specified config file"""
    from datagen.core.pipeline import GenerationPipeline, GenerationPipelineConfig

    pipeline_config = GenerationPipelineConfig.from_path(config_file)
    pipeline = GenerationPipeline(config=pipeline_config)
    pipeline.start()


@cli.command("push-to-hub")
@click.option(
    "--config_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the TOML configuration file",
)
@click.option(
    "--repo_id",
    type=str,
    required=False,
    help="HuggingFace repo id",
)
@click.option(
    "--commit_message",
    type=str,
    required=False,
    help="Commit message",
)
@click.option(
    "--update_card",
    type=bool,
    required=False,
    help="Update card",
)
def push_to_hub(
    config_file: Path,
    repo_id: str | None = None,
    commit_message: str | None = None,
    update_card: bool | None = None,
):
    from datagen.core.pipeline import GenerationPipeline, GenerationPipelineConfig

    pipeline_config = GenerationPipelineConfig.from_path(config_file)
    pipeline = GenerationPipeline(config=pipeline_config)
    pipeline.push_to_hub(
        repo_id=repo_id, commit_message=commit_message, update_card=update_card
    )


@cli.command("web")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, type=int, show_default=True)
def web(host: str, port: int):
    """Run the local Dataset Studio web interface."""
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise click.ClickException("Dataset Studio is local-only; bind to 127.0.0.1")

    import uvicorn

    uvicorn.run("datagen.web:create_app", factory=True, host=host, port=port)


@cli.command("generate-config")
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output file path for the generated config",
)
def generate_config(output: Path):
    """Generate a default config file for the templated generator"""

    # Create a basic config template for templated generator
    config = {
        "dataset_name": "Dataset generated with templated generator",
        "description": "Dataset created using the templated generator",
        "authors": ["Your Name <your.email@example.com>"],
        "generation_output_dir": "./output/templated",
        "generation_logging_steps": 100,
    }

    # Sources section (user needs to fill this)
    config["sources"] = {
        "example_source": {"path": "example/dataset", "split": "train"}
    }

    # Models section with some common examples
    config["models"] = {
        "gpt-4.1-mini": {
            "backend": "openai",
            "params": {"model": "gpt-4.1-mini", "temperature": 0.3, "max_tokens": 1000},
        },
        "claude": {
            "backend": "litellm",
            "params": {
                "model": "claude-3-sonnet-20240229",
                "temperature": 0.3,
                "max_tokens": 1000,
            },
        },
    }

    # Generator section with templated generator defaults
    config["generator"] = {
        "generator": "templated",
        "params": {
            "default_model": "gpt-4.1-mini",
            "default_system_prompt": "You are a helpful assistant.",
            "source_datasets": [
                {
                    "name": "example_source",
                    "prompt_template": "Generate a response for: {{ input.text }}",
                    "max_records": 100,
                    "shuffle": True,
                }
            ],
            "output_template": '{"input": "{{ input.text }}", "output": "{{ llm_output }}"}',
        },
    }

    # Curator section
    config["curator"] = {
        "params": {
            "upload_to_hf": False,
            "upload_repo_id": "your-username/your-dataset-name",
            "update_card": True,
            "language": ["en"],
            "license": "mit",
            "task_categories": [],
            "task_ids": [],
        }
    }

    # Write config file
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        write_toml(config, output)

        click.echo(f"Generated default config for templated generator at: {output}")
        click.echo(
            f"Please edit the config file to customize your settings before running generation."
        )

    except Exception as e:
        click.echo(f"Error writing config file: {e}", err=True)


if __name__ == "__main__":
    cli()
