import random
import string
from typing import Dict, List, Literal, Union

import mlflow
import numpy as np
import optuna
from loguru import logger


def model_params_to_trial_params(
    **model_params: Dict[str, Union[Literal[str], int, float, List, Dict]]
) -> Dict[str, Union[Literal[str], int, float, List, Dict]]:
    """
    Convert readable model_params to trial_params
    for example to run study.enqueue_trial(trial_params)
    """
    trial = {}
    trial["activation"] = model_params["activation"]
    trial["lin_prep_len"] = model_params["lin_prep_len"]
    trial["lin_prep_dropout_rate"] = model_params["lin_prep_dropout_rate"]
    trial["lin_prep_weight_norm_flag"] = model_params["lin_prep_weight_norm_flag"]
    last_size = model_params["lin_prep_size_common"]
    trial["lin_prep_size_common"] = last_size
    for i in range(model_params["lin_prep_len"]):
        trial[f"lin_prep_size{i}_fraction"] = np.clip(
            model_params["lin_prep_sizes"][i] / last_size, 0.2, 1.0
        )
        last_size = model_params["lin_prep_sizes"][i]

    trial["conv1_aggrs_mean_fraction"] = np.clip(
        model_params["conv1_aggrs"]["mean"] / last_size, 0.1, 1.0
    )
    trial["conv1_aggrs_max_fraction"] = np.clip(
        model_params["conv1_aggrs"]["max"] / last_size, 0.05, 0.7
    )
    trial["conv1_aggrs_add_fraction"] = np.clip(
        model_params["conv1_aggrs"]["add"] / last_size, 0.05, 0.7
    )

    trial["conv1_dropout_rate"] = model_params["conv1_dropout_rate"]

    if model_params["n_hops"] == 2:
        last_size = sum(model_params["conv1_aggrs"].values())

        trial["conv2_aggrs_mean_fraction"] = np.clip(
            model_params["conv2_aggrs"]["mean"] / last_size, 0.1, 0.7
        )
        trial["conv2_aggrs_max_fraction"] = np.clip(
            model_params["conv2_aggrs"]["max"] / last_size, 0.05, 0.5
        )
        trial["conv2_aggrs_add_fraction"] = np.clip(
            model_params["conv2_aggrs"]["add"] / last_size, 0.05, 0.5
        )

        trial["conv2_dropout_rate"] = model_params["conv2_dropout_rate"]

    if model_params["conv_type"] == "GraphConv":
        trial["graph_conv_weight_norm_flag"] = model_params[
            "graph_conv_weight_norm_flag"
        ]

    if model_params["conv_type"] == "NNConv":
        trial["edge_attr_repr_len"] = model_params["edge_attr_repr_len"]
        for i in range(model_params["edge_attr_repr_len"] - 1):
            if i == 0:
                trial[f"edge_attr_repr_size{i}"] = model_params["edge_attr_repr_sizes"][
                    i
                ]

            else:
                trial[f"edge_attr_repr_size{i}_fraction"] = np.clip(
                    model_params["edge_attr_repr_sizes"][i]
                    / model_params["edge_attr_repr_sizes"][i - 1],
                    0.2,
                    1.0,
                )

        trial["edge_attr_repr_size_last"] = model_params["edge_attr_repr_sizes"][-1]

        trial["edge_attr_repr_dropout_rate"] = model_params[
            "edge_attr_repr_dropout_rate"
        ]

        trial["edge_attr_repr_last_dropout_rate_zero"] = (
            model_params["edge_attr_repr_last_dropout_rate"] == 0
        )
        if not trial["edge_attr_repr_last_dropout_rate_zero"]:
            trial["edge_attr_repr_last_dropout_rate"] = model_params[
                "edge_attr_repr_last_dropout_rate"
            ]

        trial["edge_attr_repr_weight_norm_flag"] = model_params[
            "edge_attr_repr_weight_norm_flag"
        ]

    return trial


def sample_model_params(trial: optuna.Trial, conv_type: str = "GraphConv") -> Dict:
    params = {}
    params["conv_type"] = conv_type
    params["activation"] = trial.suggest_categorical(
        "activation",
        [
            "relu",  # 1st place
            "prelu",  # 2nd place
            "leakyrelu",
            "elu",
            "gelu",
        ],
    )
    # NODE FEATURES PREP params
    params["lin_prep_len"] = trial.suggest_int("lin_prep_len", low=0, high=2)
    params["lin_prep_dropout_rate"] = trial.suggest_uniform(
        "lin_prep_dropout_rate", low=0, high=0.5
    )
    params["lin_prep_weight_norm_flag"] = trial.suggest_categorical(
        "lin_prep_weight_norm_flag", [False, True]
    )

    min_lin_prep_size_common = 32
    max_lin_prep_size_common = 1024

    last_size = trial.suggest_int(
        "lin_prep_size_common",
        min_lin_prep_size_common,
        max_lin_prep_size_common,
        log=True,
    )
    params["lin_prep_size_common"] = last_size
    params["lin_prep_sizes"] = []
    for i in range(params["lin_prep_len"]):
        fraction = trial.suggest_loguniform(
            f"lin_prep_size{i}_fraction", low=0.2, high=1.0
        )
        last_size = max(16, int(np.round(last_size * fraction)))
        params["lin_prep_sizes"].append(last_size)
    params["n_hops"] = 2

    # CONV1 params

    params["conv1_aggrs"] = {}
    fraction = trial.suggest_loguniform("conv1_aggrs_mean_fraction", low=0.1, high=1.0)
    params["conv1_aggrs"]["mean"] = max(8, int(np.round(last_size * fraction)))

    fraction = trial.suggest_loguniform("conv1_aggrs_max_fraction", low=0.05, high=0.7)
    params["conv1_aggrs"]["max"] = int(np.round(last_size * fraction))

    fraction = trial.suggest_loguniform("conv1_aggrs_add_fraction", low=0.05, high=0.7)
    params["conv1_aggrs"]["add"] = int(np.round(last_size * fraction))

    params["conv1_dropout_rate"] = trial.suggest_uniform(
        "conv1_dropout_rate", low=0, high=0.5
    )

    #     return params
    # CONV2 params
    if params["n_hops"] == 2:
        last_size = sum(params["conv1_aggrs"].values())
        params["conv2_aggrs"] = {}
        fraction = trial.suggest_loguniform(
            "conv2_aggrs_mean_fraction", low=0.1, high=0.7
        )
        params["conv2_aggrs"]["mean"] = max(8, int(np.round(last_size * fraction)))

        fraction = trial.suggest_loguniform(
            "conv2_aggrs_max_fraction", low=0.05, high=0.5
        )
        params["conv2_aggrs"]["max"] = int(np.round(last_size * fraction))

        fraction = trial.suggest_loguniform(
            "conv2_aggrs_add_fraction", low=0.05, high=0.5
        )
        params["conv2_aggrs"]["add"] = int(np.round(last_size * fraction))

        params["conv2_dropout_rate"] = trial.suggest_uniform(
            "conv2_dropout_rate", low=0, high=0.5
        )
    if params["conv_type"] == "GraphConv":
        params["graph_conv_weight_norm_flag"] = trial.suggest_categorical(
            "graph_conv_weight_norm_flag", [False, True]
        )

    # EDGE ATTR params
    if params["conv_type"] == "NNConv":
        params["edge_attr_repr_len"] = trial.suggest_int(
            "edge_attr_repr_len", low=1, high=3
        )
        params["edge_attr_repr_sizes"] = []
        for i in range(params["edge_attr_repr_len"] - 1):
            if i == 0:
                params["edge_attr_repr_sizes"].append(
                    trial.suggest_int(
                        f"edge_attr_repr_size{i}", low=4, high=40, log=True
                    )
                )
            else:
                fraction = trial.suggest_loguniform(
                    f"edge_attr_repr_size{i}_fraction", low=0.2, high=1.0
                )
                params["edge_attr_repr_sizes"].append(
                    max(4, int(np.round(params["edge_attr_repr_sizes"][-1] * fraction)))
                )
        params["edge_attr_repr_sizes"].append(
            trial.suggest_int("edge_attr_repr_size_last", low=1, high=5, log=True)
        )

        params["edge_attr_repr_dropout_rate"] = trial.suggest_uniform(
            "edge_attr_repr_dropout_rate", low=0, high=0.5
        )
        if trial.suggest_categorical(
            "edge_attr_repr_last_dropout_rate_zero", [True, False]
        ):
            params["edge_attr_repr_last_dropout_rate"] = 0.0
        else:
            params["edge_attr_repr_last_dropout_rate"] = trial.suggest_uniform(
                "edge_attr_repr_last_dropout_rate", low=0, high=0.5
            )

        params["edge_attr_repr_weight_norm_flag"] = trial.suggest_categorical(
            "edge_attr_repr_weight_norm_flag", [False, True]
        )

        params["edge_attr_repr_last_activation"] = "sigmoid"

    return params
