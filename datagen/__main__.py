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
    import datagen.generators  # trigger registrations

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
    import datagen.generators  # trigger registrations

    pipeline_config = GenerationPipelineConfig.from_path(config_file)
    pipeline = GenerationPipeline(config=pipeline_config)
    pipeline.push_to_hub(
        repo_id=repo_id, commit_message=commit_message, update_card=update_card
    )


@cli.command("list-generators")
def list_generators():
    """List all available generators"""
    from datagen.core.registry import GeneratorRegistry
    import datagen.generators  # Import to trigger registrations

    generators = GeneratorRegistry.list_generators()

    if not generators:
        click.echo("No generators found.")
        return

    click.echo("Available Generators:")
    click.echo("=" * 50)

    for name, info in generators.items():
        click.echo(f"Name: {name}")
        click.echo(f"Description: {info.description}")
        click.echo(f"Class: {info.generator_class.__name__}")
        click.echo("-" * 30)


@cli.command("generate-config")
@click.argument("generator_name")
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output file path (default: {generator_name}_config.toml)",
)
def generate_config(generator_name: str, output: Path):
    """Generate a default config file for the specified generator"""
    from datagen.core.registry import GeneratorRegistry
    import datagen.generators  # Import to trigger registrations

    try:
        generator_info = GeneratorRegistry.get_generator_info(generator_name)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        return

    # Create a basic config template
    config = {
        "dataset_name": f"Dataset generated with {generator_name} generator",
        "description": f"Dataset created using the {generator_name} generator",
        "authors": ["Your Name <your.email@example.com>"],
        "generation_output_dir": f"./output/{generator_name}",
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

    # Generator section with defaults from registry
    config["generator"] = {"generator": generator_name, **generator_info.default_config}

    # Add generator-specific examples to source_datasets
    if generator_name == "paraphrase":
        config["generator"]["params"]["source_datasets"] = [
            {
                "name": "example_source",
                "sentence_column": "text",
                "max_records": 100,
                "shuffle": True,
            }
        ]
    elif generator_name == "question_answer":
        config["generator"]["params"]["source_datasets"] = [
            {
                "name": "example_source",
                "passage_column": "text",
                "max_records": 100,
                "shuffle": True,
            }
        ]

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

        click.echo(f"Generated default config for '{generator_name}' at: {output}")
        click.echo(
            f"Please edit the config file to customize your settings before running generation."
        )

    except Exception as e:
        click.echo(f"Error writing config file: {e}", err=True)


if __name__ == "__main__":
    cli()
