from pathlib import Path

import click


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--config_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
)
def generate_raw_dataset(config_file: Path):
    import tomllib

    config = tomllib.load(config_file.open("rb"))
    click.echo(config)


@cli.command()
def generate_curated_dataset():
    click.echo(f"Generating curated dataset")


if __name__ == "__main__":
    cli()
