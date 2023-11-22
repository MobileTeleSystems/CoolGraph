import os
from typing import Dict

import mlflow
import urllib3


def setup_mlflow_from_config(config: Dict) -> None:
    """
    Setup mlflow using logging.mlflow section of a config
    """

    if config.get("MLFLOW_DISABLE_INSECURE_REQUEST_WARNING", False):
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    for key, value in config.items():
        os.environ[key] = str(value)

    mlflow.set_tracking_uri(config.get("MLFLOW_TRACKING_URI"))
