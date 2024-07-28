from shutil import copytree
from site import getsitepackages

import click


@click.command()
@click.option(
    "--configs",
    default="./config/",
    help="Config Path to save.If Empty - creating a new folder ./config/ in the existing path",
)
def main(configs: str = "./config/") -> None:
    """
    This function allows you to upload configs
    to your own directory to make it easier
    to create your own configs.

    Args:
        configs (str, optional): path for configs. Defaults to "./config/".
    """
    src = getsitepackages()[-1] + "/cool_graph/config/"
    copytree(src=src, dst=configs, dirs_exist_ok=True)


if __name__ == "__main__":
    main()
