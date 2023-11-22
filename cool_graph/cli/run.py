import os
from itertools import product
from pathlib import Path
from typing import List

import click
from hydra import compose, core, initialize_config_dir
from omegaconf import DictConfig

from ..runners import ConfigRunner

default_config_path = (
    Path(__file__).parent.parent.parent / "tests" / "sample_config" / "full.yaml"
)


def create_config(config: str, overrides: List[str], path_base: str) -> DictConfig:
    core.global_hydra.GlobalHydra.instance().clear()
    if os.path.isabs(config):
        config_path = Path(config).parent
    else:
        config_path = Path(os.getcwd()) / Path(config).parent

    config_name = Path(config).name.replace(".yaml", "")
    initialize_config_dir(str(config_path), version_base=None)
    cfg = compose(config_name=config_name, overrides=overrides)

    pairs = product(
        ["train", "validation"], ["nodes_path", "edges_path", "labels_path"]
    )

    if path_base == "cfg":
        for part, name in pairs:
            path = cfg["data"][part][name]
            if not os.path.isabs(path):
                cfg["data"][part][name] = os.path.join(config_path, path)

    return cfg


@click.command()
@click.option("--config", default=default_config_path, help="Config Path")
@click.option(
    "--path_base",
    default="cfg",
    type=click.Choice(["cwd", "cfg"]),
    help="""How to resolve paths in the config:
              cwd -- based on script working directory,
              cfg -- based on config directory
              """,
)
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def main(config: str, path_base: str, overrides: List[str]):
    cfg = create_config(config, overrides, path_base=path_base)

    runner = ConfigRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
