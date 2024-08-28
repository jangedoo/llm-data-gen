import logging
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--config_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
)
def generate(config_file: Path):
    from nep_qa_dataset.pipeline import GenerationPipeline, GenerationPipelineConfig

    pipeline_config = GenerationPipelineConfig.from_path(config_file)
    pipeline = GenerationPipeline(config=pipeline_config)
    pipeline.start()


if __name__ == "__main__":
    cli()
