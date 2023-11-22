import pytest

from cool_graph.cli.run import create_config
from cool_graph.runners import ConfigRunner


def test_nnconv():
    try:
        cfg = create_config(
            "tests/sample_config/full.yaml",
            overrides=["model_params=nnconv", "logging.use_mlflow=False"],
            path_base="cfg",
        )
        runner = ConfigRunner(cfg)
        runner.run()
    except Exception as ex:
        raise pytest.fail(f"Exception in test_nnconv:\n {str(ex)}")


def test_grapconv():
    try:
        cfg = create_config(
            "tests/sample_config/full.yaml",
            overrides=["model_params=graphconv", "logging.use_mlflow=False"],
            path_base="cfg",
        )
        runner = ConfigRunner(cfg)
        runner.run()
    except Exception as ex:
        raise pytest.fail(f"Exception in test_graphconv:\n {str(ex)}")
