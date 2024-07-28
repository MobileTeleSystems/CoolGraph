import os
import pathlib
import traceback
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Literal, Optional

import hydra
import numpy as np
import optuna
import pandas as pd
import torch
from hydra import (
    compose,
    core,
    initialize,
    initialize_config_dir,
    initialize_config_module,
)
from omegaconf import DictConfig, OmegaConf
from optuna.trial import TrialState
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, NeighborSampler
from tqdm import tqdm

from cool_graph.data import RawDataProcessor
from cool_graph.data.batch import get_auto_batch_size
from cool_graph.data.loaders import create_loaders
from cool_graph.data.utils import preprocessing_data
from cool_graph.data.utils import compress_x_cat
from cool_graph.data.utils import add_graph_node_features
from cool_graph.data.utils import add_graph_edge_features
from cool_graph.logging import setup_mlflow_from_config
from cool_graph.parameter_search import (
    model_params_to_trial_params,
    sample_model_params,
)
from cool_graph.train import Trainer
from cool_graph.train.helpers import eval_epoch
from cool_graph.train.metrics import get_metric


def create_config2(config: str, overrides: List[str], path_base: str = "cfg") -> Dict:
    core.global_hydra.GlobalHydra.instance().clear()
    if os.path.isabs(config):
        config_path = pathlib.Path(config).parent
    else:
        config_path = pathlib.Path(os.getcwd()) / pathlib.Path(config).parent

    config_name = pathlib.Path(config).name.replace(".yaml", "")
    initialize_config_dir(str(config_path), version_base=None)
    cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


class ConfigRunner:
    r"""Runner for cli mode. Using only in cli.
    This class allows to load data + split data per batchs + split data per train/val + training.
    See the config full.yaml in ./config for knowing what excactly using as data/logging/model_params/training/metrics.
    You can use default params, but also you can change it.
    Steps for changing confis:
    - make get_config --configs path_where_you_need_configs (default: new path ./configs by itself)
    """

    def __init__(self, config: Optional[DictConfig]) -> None:
        cfg = OmegaConf.to_container(config, resolve=True)
        self.cfg = cfg
        self.target_names = cfg["training"]["targets"]
        self.groups_names = cfg["data"]["groups_names"]
        self.target_weights = cfg["training"]["loss"]["target_weights"]
        self.read_edge_attr = cfg["data"].get("read_edge_attr", True)
        self.batch_size = cfg["training"]["batch_size"]
        self.group_mask_col = cfg["data"]["group_mask_col"]
        self.label_mask_col = cfg["data"]["label_mask_col"]
        self.label_cols = cfg["data"]["label_cols"]
        self.label_index_col = cfg["data"]["label_index_col"]
        self.edge_index_cols = cfg["data"]["edge_index_cols"]
        self.num_neighbors = cfg["training"]["num_neighbors"]
        self.features_edges_names = cfg["data"].get("features_edges")
        self.group_names_node_features = cfg["data"]["features"]
        self.train_paths = cfg["data"]["train"]
        self.val_paths = cfg["data"]["validation"]
        self.metrics = cfg["metrics"]
        self.chkpt_dir = (
            pathlib.Path(cfg["logging"]["checkpoint_dir"]) / str(datetime.now())[:19]
        )
        os.makedirs(self.chkpt_dir, exist_ok=True)

        if self.cfg["logging"].get("use_mlflow", False):
            setup_mlflow_from_config(cfg["logging"]["mlflow"])

    def init_loaders(self) -> None:
        """
        Using RawDataProcessor from cool_graph.data for preprocessing data from disk.
        """
        self.train_sampler = RawDataProcessor(
            self.groups_names,
            self.group_names_node_features,
            mon_nodes_path=self.train_paths["nodes_path"],
            mon_edges_path=self.train_paths["edges_path"],
            mon_labels_path=self.train_paths["labels_path"],
            edge_index_cols=self.edge_index_cols,
            label_index_col=self.label_index_col,
            label_mask_col=self.label_mask_col,
            read_edge_attr=self.read_edge_attr,
            group_mask_col=self.group_mask_col,
            features_edges_names=self.features_edges_names,
            label_cols=self.label_cols,
            target_names=self.target_names,
        )

        self.val_sampler = RawDataProcessor(
            self.groups_names,
            self.group_names_node_features,
            mon_nodes_path=self.val_paths["nodes_path"],
            mon_edges_path=self.val_paths["edges_path"],
            mon_labels_path=self.val_paths["labels_path"],
            edge_index_cols=self.edge_index_cols,
            label_index_col=self.label_index_col,
            label_mask_col=self.label_mask_col,
            read_edge_attr=self.read_edge_attr,
            group_mask_col=self.group_mask_col,
            features_edges_names=self.features_edges_names,
            label_cols=self.label_cols,
            target_names=self.target_names,
        )

    def sample_data(
        self, seed=0
    ) -> Dict[Literal["train", "validation"], List[torch.utils.data.DataLoader]]:
        """
        Sampling data in batches.
        """
        if self.batch_size == "auto":
            self._batch_size = get_auto_batch_size(
                [len(v) for _, v in self.group_names_node_features.items()],
                conv_type=self.cfg["model_params"]["conv_type"],
                conv1_aggrs=self.cfg["model_params"]["conv1_aggrs"],
                conv2_aggrs=self.cfg["model_params"].get("conv2_aggrs"),
                conv3_aggrs=self.cfg["model_params"].get("conv3_aggrs"),
                n_hops=self.cfg["model_params"]["n_hops"],
                lin_prep_size_common=self.cfg["model_params"]["lin_prep_size_common"],
                lin_prep_sizes=self.cfg["model_params"]["lin_prep_sizes"],
                edge_attr_repr_sizes=self.cfg["model_params"].get(
                    "edge_attr_repr_sizes"
                ),
                num_edge_features=len(self.cfg["data"].get("features_edges", [])),
                device=self.cfg["training"]["device"],
                num_neighbors=self.cfg["training"]["num_neighbors"],
            )
        else:
            self._batch_size = self.batch_size

        train_loaders = self.train_sampler.sample_data(
            self.num_neighbors, self._batch_size, seed=seed
        )

        val_loaders = self.val_sampler.sample_data(
            self.num_neighbors, self._batch_size, seed=seed
        )

        return {"train": train_loaders, "validation": val_loaders}

    def run(self, seed: int = 0) -> Dict[str, float]:
        """
        Train model for train_samples and val_sampler.

        Args:
            seed (int): seed for training. Default to 0.

        Returns:
            result (dict): Result of training for each 5 epochs with metrics from config.
        """
        if not (hasattr(self, "train_sampler") and hasattr(self, "val_sampler")):
            self.init_loaders()
        sampled = self.sample_data(seed=seed)
        train_loaders = sampled["train"]
        val_loaders = sampled["validation"]

        self.trainer = Trainer(
            train_loaders,
            val_loaders,
            self.chkpt_dir,
            device=self.cfg["training"]["device"],
            eval_freq=self.cfg["training"]["eval_freq"],
            fill_value=self.cfg["training"]["loss"].get("fill_value"),
            initial_lr=self.cfg["training"].get("initial_lr", 0.01),
            weight_decay=self.cfg["training"].get("weight_decay", 0.0),
            loss_name=self.cfg["training"]["loss"]["name"],
            loss_label_smoothing=self.cfg["training"]["loss"].get(
                "label_smoothing", False
            ),
            loss_target_weights=self.cfg["training"]["loss"].get("target_weights"),
            loss_group_weights=self.cfg["training"]["loss"].get("group_weights"),
            groups_names=self.cfg["data"]["groups_names"],
            mlflow_experiment_name=self.cfg["logging"].get("mlflow_experiment_name"),
            n_epochs=self.cfg["training"].get("n_epochs"),
            scheduler_params=self.cfg["training"].get("scheduler_params", {}),
            scheduler_type=self.cfg["training"].get("scheduler_type"),
            target_names=self.cfg["training"]["targets"],
            use_mlflow=self.cfg["logging"].get("use_mlflow", False),
            tqdm_disable=False,
            **self.cfg["model_params"],
            groups_names_num_features={
                k: len(v) for k, v in self.group_names_node_features.items()
            },
            num_edge_features=len(self.cfg["data"].get("features_edges", [])),
            metrics=self.metrics,
        )
        result = self.trainer.train()
        return result


class BaseRunner:
    def __init__(
        self,
        data: Data,
        config: Optional[DictConfig] = None,
        config_path: Optional[str] = None,
        overrides: Optional[List] = None,
        train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        seed: Optional[int] = None,
        train_idx: Optional[List[int]] = None,
        test_idx: Optional[List[int]] = None,
        use_edge_attr: bool = False,
        use_graph_node_features = False,
        use_graph_edge_features = False,
        use_index_as_feature: bool = False,
        verbose: bool = True,
        embedding_data: bool = False,
        **kwargs,
    ) -> None:
        """
        Main class for Basic runner and Runner with Optuna.

        Args:
            data (Data): A data object describing a homogeneous graph. The data object can hold node-level,
            link-level and graph-level attributes. In general, Data tries to mimic the behavior of a regular Python
            dictionary. In addition, it provides useful functionality for analyzing graph structures, and provides basic
            PyTorch tensor functionalities.
            https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs
            config (DictConfig): Config. Defaults to None.
            config_path (str): Path to config. Defaults to None.
            overrides (list): Own params. Can ba params from configs and overrides. Defaults to None.
            train_size (int): Size for train data. Defaults to None.
            test_size (int): Size for test data. Defaults to None.
            seed (int): Seed param for training. Defaults to None.
            train_idx (list): Indices for train data. Defaults to None.
            test_idx (list): Indices for test data. Defaults to None.
            use_edge_attr (bool): If attributes exist on edges, it can be used in training. Defaults to False.

        """
        if config is None:
            if config_path is None:
                if use_edge_attr:
                    config_path = "./config/in_memory_data2.yaml"
                else:
                    config_path = "./config/in_memory_data.yaml"
                config_path = os.path.join(os.path.dirname(__file__), config_path)
            config = create_config2(
                config=config_path, overrides=overrides, path_base="cfg"
            )
        cfg = OmegaConf.to_container(config, resolve=True)
        self.data = data
        self.cfg = cfg
        self.test_size = test_size
        self.train_size = train_size
        self.seed = seed
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.use_edge_attr = use_edge_attr
        self.embedding_data = embedding_data
        self.use_cat_features = hasattr(data, 'x_cat')
        self.verbose = verbose
        
        optuna.logging.set_verbosity(
            optuna.logging.INFO if self.verbose
            else optuna.logging.CRITICAL
        )

        if use_graph_node_features:
            self.data = add_graph_node_features(self.data)

        if use_graph_edge_features:
            self.data = add_graph_edge_features(self.data)
            use_edge_attr = True
            
        if use_edge_attr and data.edge_attr is None:
            raise BaseException(
                "data does not contain edge_attr, please set use_edge_attr=False"
            )

        self.target_weights = cfg["training"]["loss"]["target_weights"]
        self.batch_size = cfg["training"]["batch_size"]
        self.num_neighbors = cfg["training"]["num_neighbors"]
        self.metrics = cfg["metrics"]
        self.data.group_mask = torch.zeros(len(data.x), dtype=torch.int8)
        self.groups_names = {0: "x"}
        self.groups_names_num_features = {"x": data.x.shape[1]}
        
        if use_index_as_feature:
            index = torch.tensor(list(range(0, self.data.x.shape[0]))).reshape(-1, 1)
            if self.use_cat_features:
                self.data.x_cat = torch.hstack((self.data.x_cat, index))
            else:
                self.data.x_cat = index
                self.use_cat_features=True
        
        if self.use_cat_features:
            self.groups_cat_names = {0: "x_cat"}
            self.groups_names_num_cat_features = {"x_cat": data.x_cat.shape[1]}
            
            if type(self.data.x_cat) != torch.Tensor:
                self.data.x_cat = torch.tensor(self.data.x_cat)
            self.data = compress_x_cat(self.data)
            self.data.x_cat = self.data.x_cat.long()
            
            self.cat_features_sizes = torch.max(self.data.x_cat, 0).values + 1
        else:
            self.groups_names_num_cat_features = 0
            self.cat_features_sizes = None
    
        if len(data.y.shape) == 2:
            (
                self.target_names,
                self.target_weights,
                self.target_sizes,
            ) = preprocessing_data(self.data)

        else:
            self.target_names = cfg["training"]["targets"]
            self.classes = self.data.y.unique()
            self.target_sizes = [
                len(self.classes[self.classes != cfg["training"]["loss"]["fill_value"]])
            ]
            self.target_weights = cfg["training"]["loss"]["target_weights"]

        setattr(
            self.data,
            "label_mask",
            (
                torch.vstack([getattr(data, name) for name in self.target_names])
                != cfg["training"]["loss"]["fill_value"]
            )
            .max(dim=0)
            .values,
        )

        self.data.label_mask = data.label_mask

        if use_edge_attr:
            self.num_edge_features = data.edge_attr.shape[1]
        else:
            self.num_edge_features = 0

        self.chkpt_dir = (
            pathlib.Path(cfg["logging"]["checkpoint_dir"]) / str(datetime.now())[:19]
        )
        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.cfg["logging"].get("use_mlflow", False):
            setup_mlflow_from_config(cfg["logging"]["mlflow"])

    def sample_data(self) -> None:
        """
        Sampling data into batches and sampling data with NeighborLoader into list loaders.
        """
        if self.batch_size == "auto":
            self._batch_size = get_auto_batch_size(
                [
                    self.groups_names_num_features[self.groups_names[i]]
                    for i in range(len(self.groups_names))
                ],
                conv_type=self.cfg["model_params"]["conv_type"],
                conv1_aggrs=self.cfg["model_params"]["conv1_aggrs"],
                conv2_aggrs=self.cfg["model_params"].get("conv2_aggrs"),
                conv3_aggrs=self.cfg["model_params"].get("conv3_aggrs"),
                n_hops=self.cfg["model_params"]["n_hops"],
                lin_prep_size_common=self.cfg["model_params"]["lin_prep_size_common"],
                lin_prep_sizes=self.cfg["model_params"]["lin_prep_sizes"],
                edge_attr_repr_sizes=self.cfg["model_params"].get(
                    "edge_attr_repr_sizes"
                ),
                num_edge_features=self.num_edge_features,
                device=self.cfg["training"]["device"],
                num_neighbors=self.num_neighbors,
            )
        else:
            self._batch_size = self.batch_size

        if (self.train_idx is None) or (self.test_idx is None):
            train_idx, test_idx = train_test_split(
                torch.nonzero(self.data.label_mask)[:, 0],
                train_size=self.train_size,
                test_size=self.test_size,
                random_state=self.seed,
                shuffle=True,
            )
            self.train_idx = train_idx
            self.test_idx = test_idx

        unique_groups = np.unique(self.data.group_mask)

        self.train_loader = create_loaders(
            data=self.data,
            num_neighbors=self.num_neighbors,
            batch_size=self._batch_size,
            input_nodes=self.train_idx,
            groups_names=self.groups_names,
            group_mask=self.data.group_mask,
            unique_groups=unique_groups,
            disable= not self.verbose,
        )

        self.test_loader = create_loaders(
            data=self.data,
            num_neighbors=self.num_neighbors,
            batch_size=self._batch_size,
            input_nodes=self.test_idx,
            groups_names=self.groups_names,
            group_mask=self.data.group_mask,
            unique_groups=unique_groups,
            disable= not self.verbose,
        )
        return {"train_loader": self.train_loader, "test_loader": self.test_loader}


class Runner(BaseRunner):
    """
    Runner for notebook launch.

    Args:
    data (Data): A data object describing a homogeneous graph. The data object can hold node-level,
    link-level and graph-level attributes. In general, Data tries to mimic the behavior of a regular Python
    dictionary. In addition, it provides useful functionality for analyzing graph structures, and provides basic
    PyTorch tensor functionalities.
    https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs
    config (DictConfig): Config. Defaults to None.
    config_path (str): Path to config. Defaults to None.
    overrides (list): Own params. Can ba params from configs and overrides. Defaults to None.
    train_size (int): Size for train data. Defaults to None.
    test_size (int): Size for test data. Defaults to None.
    seed (int): Seed param for training. Defaults to None.
    train_idx (int): Indices for train data. Defaults to None.
    test_idx (int): Indices for test data. Defaults to None.
    use_edge_attr (bool): If attributes exist on edges, it can be used in training. Defaults to False.

    Examples
    --------
    >>> from cool_graph.runners import Runner
    >>> from torch_geometric import datasets
    >>> # loading amazon dataset
    >>> data = datasets.Amazon(root="./data/Amazon", name="Computers").data
    >>> runner = Runner(data)
    >>> result = runner.run()
    >>> result["best_loss"]
        {'accuracy': 0.916,
        'cross_entropy': 0.286,
        'f1_micro': 0.916,
        'calc_time': 0.004,
        'main_metric': 0.916,
        'epoch': 10}
    Also you can override params in Runner:
    runner = Runner(data, metrics=['accuracy'],
    batch_size='auto', train_size=0.7, test_size=0.3,
    overrides=['training.n_epochs=1'], config_path=path/to/config)
    result = runner.run()

    """

    def __init__(
        self,
        data: Data,
        config: Optional[DictConfig] = None,
        config_path: Optional[str] = None,
        overrides: Optional[List] = None,
        train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        seed: Optional[int] = None,
        train_idx: Optional[List[int]] = None,
        test_idx: Optional[List[int]] = None,
        use_edge_attr: bool = False,
        use_graph_node_features = False,
        use_graph_edge_features = False,
        use_index_as_feature: bool = False,
        verbose: bool = True,
        embedding_data: bool = False,
        **kwargs,
    ):
        super().__init__(
            data,
            config,
            config_path,
            overrides,
            train_size,
            test_size,
            seed,
            train_idx,
            test_idx,
            use_edge_attr,
            use_graph_node_features,
            use_graph_edge_features,
            use_index_as_feature,
            verbose,
            embedding_data,
            **kwargs,
        )
        self.trainer = None

    def run(self, train_loader=None, test_loader=None) -> Dict[str, float]:
        """
        Training model with params in_memory_data/in_memory_data2 config.
        See the configs in ./config for knowing what excactly using as logging/model_params/training/metrics.
        You can use default params, but also you can change it.
        Steps for changing confis:
        - make get_config --configs path_where_you_need_configs (default: new path ./configs by itself)
        """
        self.train_loader = train_loader
        self.test_loader = test_loader

        if (self.train_loader is None) and (self.test_loader is None):
            self.sample_data()
        elif "index" not in self.data.keys:
            raise Exception("please, add field 'index' to your data before initializing loaders. Look at notebooks for Example")

        self.trainer = Trainer(
            self.train_loader,
            self.test_loader,
            self.chkpt_dir,
            device=self.cfg["training"]["device"],
            eval_freq=self.cfg["training"]["eval_freq"],
            fill_value=self.cfg["training"]["loss"].get("fill_value"),
            initial_lr=self.cfg["training"].get("initial_lr", 0.01),
            weight_decay=self.cfg["training"].get("weight_decay", 0.0),
            loss_name=self.cfg["training"]["loss"]["name"],
            loss_label_smoothing=self.cfg["training"]["loss"].get(
                "label_smoothing", False
            ),
            loss_target_weights=self.target_weights,
            loss_group_weights=self.cfg["training"]["loss"].get("group_weights"),
            groups_names=self.groups_names,
            mlflow_experiment_name=self.cfg["logging"].get("mlflow_experiment_name"),
            n_epochs=self.cfg["training"].get("n_epochs"),
            scheduler_params=self.cfg["training"].get("scheduler_params", {}),
            scheduler_type=self.cfg["training"].get("scheduler_type"),
            target_names=self.target_names,
            use_mlflow=self.cfg["logging"].get("use_mlflow", False),
            tqdm_disable=not self.verbose,
            target_sizes=self.target_sizes,
            **self.cfg["model_params"],
            groups_names_num_features=self.groups_names_num_features,
            groups_names_num_cat_features=self.groups_names_num_cat_features,
            cat_features_sizes=self.cat_features_sizes,
            num_edge_features=self.num_edge_features,
            metrics=self.metrics,
            log_all_metrics=False,
            log_metric=self.verbose,
            embedding_data=self.embedding_data,
        )
        result = self.trainer.train()
        return result

    def predict_proba(
        self,
        data,
        test_mask=None,
        predict_loader=None,
        batch_size: Optional[int] = None,
        num_neighbors: Optional[int] = None
    ):
        """
        Args:
        data (Data): A data object describing a homogeneous graph. The data object can hold node-level,
        link-level and graph-level attributes. In general, Data tries to mimic the behavior of a regular Python
        dictionary. In addition, it provides useful functionality for analyzing graph structures, and provides basic
        PyTorch tensor functionalities.
        https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs
        test_mask: mask of nodes from data.x on which predictions will be made. Default to range(0, len(data.x)), meaning
        all of nodes from data

        Returns:
            preds (Dict[str, float]): Dict with predictions of probabilities on each task
            indices (torch.tensor(int)): indices matching nodes from preds to labels from data

        Examples
        --------
        >>> from cool_graph.runners import Runner
        >>> from torch_geometric import datasets
        >>> # loading amazon dataset
        >>> data = datasets.Amazon(root="./data/Amazon", name="Computers").data
        >>> runner = Runner(data)
        >>> result = runner.run()
        >>> result["best_loss"]
            {'accuracy': 0.916,
            'cross_entropy': 0.286,
            'f1_micro': 0.916,
            'calc_time': 0.004,
            'main_metric': 0.916,
            'epoch': 10}
        >>> preds, indices = runner.predict_proba(data, test_mask=runner.test_idx,
            batch_size='auto')
        >>>preds
            'y': array([[1.93610392e-03, 5.08281529e-01, 3.10600898e-03, ...,
             7.42573151e-03, 3.01667452e-01, 1.28832005e-03],
            [1.47580403e-11, 9.99880552e-01, 1.03485095e-10, ...,
             6.51596277e-11, 2.65427474e-07, 6.00029034e-15],
            [9.98413205e-01, 2.00255581e-08, 6.93480979e-06, ...,
             2.96822922e-08, 3.92414279e-08, 3.74058899e-08],
            ...,
            [3.35409859e-05, 1.15004822e-03, 2.12268560e-06, ...,
             2.37720778e-05, 8.57518315e-01, 1.21086057e-04],
            [8.91507423e-11, 9.99903917e-01, 3.53415408e-10, ...,
             7.17140583e-11, 8.49195416e-08, 2.95486618e-15],
            [6.94501125e-07, 1.11814063e-06, 1.24563138e-11, ...,
             7.48802895e-11, 4.13234375e-06, 4.48725057e-10]], dtype=float32)}
        >>>preds["y"].shape
            (3438, 10) # preds for 3438 nodes in test sample of each of 10 classes
        >>>len(indices)
            3438
        """
        if self.trainer is None:
            raise Exception("runner is not trained, please do run()")

        if test_mask is None:
            test_mask = range(0, len(data.x))

        if batch_size is None:
            batch_size = self.batch_size

        if num_neighbors is None:
            num_neighbors = self.num_neighbors

        #preprocessing_data(data)

        data.group_mask = torch.zeros(len(data.x), dtype=torch.int8)
        unique_groups = np.unique(data.group_mask)
        
        self.predict_loader = predict_loader
        if predict_loader is None:
            self.predict_loader = create_loaders(
                data=data,
                num_neighbors=num_neighbors,
                batch_size=batch_size,
                input_nodes=test_mask,
                groups_names=self.groups_names,
                group_mask=data.group_mask,
                unique_groups=unique_groups,
                disable=not self.verbose,
            )
        elif "index" not in self.data.keys:
            raise Exception("please, add field 'index' to your data before initializing loaders. Look at notebooks for Example")

        test_metric, test_preds, test_indices = eval_epoch(
            self.trainer._model,  # _model from trainer
            self.predict_loader,  # loader for prediction
            self.trainer.device,
            self.target_names,
            self.groups_names,
            postfix="predict",
            use_edge_attr=self.trainer._use_edge_attr,
            tqdm_disable=not self.verbose,
            fill_value=self.trainer.fill_value,
            count_metrics=False,
            log_all_metrics=False,
            log_metric=self.verbose,
            embedding_data=self.embedding_data,
        )

        return test_preds, test_indices


class HypeRunner(BaseRunner):
    """
    Runner for optimization model with Optuna.
    https://optuna.readthedocs.io/en/stable/reference/index.html
    1st trial - with default config params (hyper_params).
    Also, 2nd trial - you can add own trial as argument enqueue_trial in optimazire_run method, and next
    trial optuna optimize model params randomly,
    if set None randomly optimization after 1st default trial.

    Args:
        data (Data): Loaded dataset.
        config (DictConfig): Confif with patams (model_params, logging, training, metrics). Default to None.
        config_path (str): Path with config structure (can be loaded with cli get_config). Default to None.
        overrides (list): Own params in list. Default to None.
        train_size (int): Own train size. Default to None.
        test (int): Own test size. Default to None.
        seed  (int): The desired seed. Default to None.
        train_idx (list): List of train indices.
        test_idx (list): List of test indices.

    Examples
    --------
    >>> from cool_graph.runners import HypeRunner
    >>> from torch_geometric import datasets
    >>> # loading amazon dataset
    >>> data = datasets.Amazon(root="./data/Amazon", name="Computers").data
    >>> runner = HypeRunner(data)
    >>> result = runner.optimize_run()
    Study statistics:
      Number of finished trials:  5
      Number of complete trials:  5
    Best trial:
      Value:  0.922
      Params:
      {'conv_type': 'GraphConv', 'activation': 'leakyrelu', 'lin_prep_len': 1, 'lin_prep_dropout_rate': 0.4,
      'lin_prep_weight_norm_flag': True, 'lin_prep_size_common': 512, 'lin_prep_sizes': [256],
      'n_hops': 2, 'conv1_aggrs': {'mean': 128, 'max': 64, 'add': 32},
      'conv1_dropout_rate': 0.2, 'conv2_aggrs': {'mean': 64, 'max': 32, 'add': 16},
      'conv2_dropout_rate': 0.2, 'graph_conv_weight_norm_flag': True}

    """

    def __init__(
        self,
        data: Data,
        config: Optional[DictConfig] = None,
        config_path: Optional[str] = None,
        overrides: Optional[List] = None,
        train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        seed: Optional[int] = None,
        train_idx: Optional[List[int]] = None,
        test_idx: Optional[List[int]] = None,
        use_edge_attr: bool = False,
        use_graph_node_features = False,
        use_graph_edge_features = False,
        use_index_as_feature: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__(
            data,
            config,
            config_path,
            overrides,
            train_size,
            test_size,
            seed,
            train_idx,
            test_idx,
            use_edge_attr,
            use_graph_node_features,
            use_graph_edge_features,
            use_index_as_feature,
            verbose,
            **kwargs,
        )

        self.best_trainer = None
        self.best_model_loss = None

        if config is None:
            if config_path is None:
                if use_edge_attr:
                    config_path = "./config/in_memory_data2.yaml"
                else:
                    config_path = "./config/in_memory_data.yaml"
                config_path = os.path.join(os.path.dirname(__file__), config_path)
            config = create_config2(
                config=config_path, overrides=overrides, path_base="cfg"
            )

        if use_edge_attr and data.edge_attr is None:
            raise BaseException(
                "data does not contain edge_attr, please set use_edge_attr=False"
            )

        self.study = optuna.study
        self.use_edge_attr = use_edge_attr

        for k, v in kwargs.items():
            setattr(self, k, v)

    def optimize_run(
        self,
        n_trials: int = 100,
        storage: Optional[str] = None,
        study_name: Optional[str] = None,
        enqueue_trial: Optional[List[Dict]] = None,
        train_loader=None,
        test_loader=None,
    ) -> pd.DataFrame:
        """
        Method for running objective function in Optuna.

        Args:
            n_trials (int, optional): The number of trials for each process.
            None represents no limit in terms of the number of trials. Defaults to 100.
            storage (Optional[str], optional): Database URL. If this argument is set to None,
            in-memory storage is used, and the Study will not be persistent. Defaults to None.
            study_name (Optional[str], optional): Study name.
            If this argument is set to None, a unique name is generated automatically. Defaults to None.
            enqueue_trial (Optional[List[Dict]], optional): Enqueue a trial with given parameter values. Defaults to None.

        Returns:
            trials_dataset (pd.DataFrame): Result dataframe with trial params.
        """

        self.train_loader = train_loader
        self.test_loader = test_loader

        if (self.train_loader is None) and (self.test_loader is None):
            self.sample_data()
        elif "index" not in self.data.keys:
            raise Exception("please, add field 'index' to your data before initializing loaders. Look at notebooks for Example")

        list_with_params = []

        def objective(trial) -> float:
            self.cfg["model_params"] = sample_model_params(
                trial, conv_type=self.cfg["model_params"]["conv_type"]
            )

            list_with_params.append(self.cfg["model_params"])

            self.trainer = Trainer(
                self.train_loader,
                self.test_loader,
                self.chkpt_dir,
                device=self.cfg["training"]["device"],
                eval_freq=self.cfg["training"]["eval_freq"],
                fill_value=self.cfg["training"]["loss"].get("fill_value"),
                initial_lr=self.cfg["training"].get("initial_lr", 0.01),
                weight_decay=self.cfg["training"].get("weight_decay", 0.0),
                loss_name=self.cfg["training"]["loss"]["name"],
                loss_label_smoothing=self.cfg["training"]["loss"].get(
                    "label_smoothing", False
                ),
                loss_target_weights=self.target_weights,
                loss_group_weights=self.cfg["training"]["loss"].get("group_weights"),
                groups_names=self.groups_names,
                mlflow_experiment_name=self.cfg["logging"].get(
                    "mlflow_experiment_name"
                ),
                n_epochs=self.cfg["training"].get("n_epochs"),
                scheduler_params=self.cfg["training"].get("scheduler_params", {}),
                scheduler_type=self.cfg["training"].get("scheduler_type"),
                target_names=self.target_names,
                use_mlflow=self.cfg["logging"].get("use_mlflow", False),
                tqdm_disable=not self.verbose,
                target_sizes=self.target_sizes,
                **self.cfg["model_params"],
                groups_names_num_features=self.groups_names_num_features,
                groups_names_num_cat_features=self.groups_names_num_cat_features,
                cat_features_sizes=self.cat_features_sizes,
                num_edge_features=self.num_edge_features,
                metrics=self.metrics,
                log_all_metrics=False,
                log_metric=self.verbose,
            )
            result = self.trainer.train()
            output = result["best_loss"]["main_metric"]
            output = round(output, 3)

            if self.best_trainer is None:
                self.best_trainer = self.trainer
                self.best_model_loss = output

            if output > self.best_model_loss:
                self.best_model_loss = output
                self.best_trainer = self.trainer

            return output

        # default params for the 1st trial in Optuna optimization
        trial_params = model_params_to_trial_params(**self.cfg["model_params"])
        trial_params["weight_decay"] = self.cfg["training"].get("weight_decay", 0.0)

        self.study = optuna.create_study(
            storage=storage,
            study_name=study_name,
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.RandomSampler(seed=120),
        )

        # adding a trial_params as a default one to optuna optimization
        self.study.enqueue_trial(trial_params)

        # users params for the 2nd trial,
        # if None use optuna random params to trial
        if enqueue_trial:
            for param in enqueue_trial:
                user_params = model_params_to_trial_params(**param)
                self.study.enqueue_trial(user_params)

        self.study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=1,
            show_progress_bar=False,
            gc_after_trial=True,
            catch=(RuntimeError, ValueError),
        )

        complete_trials = self.study.get_trials(
            deepcopy=False, states=[TrialState.COMPLETE]
        )
        
        trial = self.study.best_trial
        dict_with_params = dict(enumerate(list_with_params))
        
        trials_dataset = self.study.trials_dataframe()
        trials_dataset = trials_dataset[
            [
                "number",
                "value",
                "datetime_start",
                "datetime_complete",
                "duration",
                "system_attrs_fixed_params",
                "state",
            ]
        ]
        trial_dataset = pd.concat(
            [trials_dataset, pd.DataFrame(dict_with_params).T], axis=1
        )
        if self.verbose:
            print("Study statistics: ")
            print("  Number of finished trials: ", len(self.study.trials))
            print("  Number of complete trials: ", len(complete_trials))
            print("Best trial:")
            print("  Value: ", trial.value)
            print("  Params: ")
            for i in trial_dataset["number"].tolist():
                if trial_dataset["value"][i] == trial_dataset["value"].max():
                    print(dict_with_params[i])

        return trial_dataset

    def predict_proba(
        self,
        data,
        test_mask=None,
        predict_loader=None,
        batch_size: Optional[int] = None,
        num_neighbors: Optional[int] = None
    ):
        """
        Args:
        data (Data): A data object describing a homogeneous graph. The data object can hold node-level,
        link-level and graph-level attributes. In general, Data tries to mimic the behavior of a regular Python
        dictionary. In addition, it provides useful functionality for analyzing graph structures, and provides basic
        PyTorch tensor functionalities.
        https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs
        test_mask: mask of nodes from data.x on which predictions will be made. Default to range(0, len(data.x)), meaning
        all of nodes from data

        Returns:
            preds (Dict[str, float]): Dict with predictions probabilities on each task
            indices (torch.tensor(int)): indices matching nodes from preds to labels from data

        Examples
        --------
        >>> from cool_graph.runners import HypeRunner
        >>> from torch_geometric import datasets
        >>> # loading amazon dataset
        >>> data = datasets.Amazon(root="./data/Amazon", name="Computers").data
        >>> runner = HypeRunner(data)
        >>> result = runner.optitmize_run()
        >>> preds, indices = runner.predict_proba(data, test_mask=runner.test_idx metrics=['accuracy'],
            batch_size='auto')
        >>>preds
            'y': array([[1.93610392e-03, 5.08281529e-01, 3.10600898e-03, ...,
             7.42573151e-03, 3.01667452e-01, 1.28832005e-03],
            [1.47580403e-11, 9.99880552e-01, 1.03485095e-10, ...,
             6.51596277e-11, 2.65427474e-07, 6.00029034e-15],
            [9.98413205e-01, 2.00255581e-08, 6.93480979e-06, ...,
             2.96822922e-08, 3.92414279e-08, 3.74058899e-08],
            ...,
            [3.35409859e-05, 1.15004822e-03, 2.12268560e-06, ...,
             2.37720778e-05, 8.57518315e-01, 1.21086057e-04],
            [8.91507423e-11, 9.99903917e-01, 3.53415408e-10, ...,
             7.17140583e-11, 8.49195416e-08, 2.95486618e-15],
            [6.94501125e-07, 1.11814063e-06, 1.24563138e-11, ...,
             7.48802895e-11, 4.13234375e-06, 4.48725057e-10]], dtype=float32)}
        >>>preds["y"].shape
            (3438, 10) # preds for 3438 nodes in test sample of each of 10 classes
        >>>len(indices)
            3438
        """

        if self.trainer is None:
            raise Exception("runner is not trained, please do optimize_run()")

        if test_mask is None:
            test_mask = range(0, len(data.x))

        if batch_size is None:
            batch_size = self.batch_size

        if num_neighbors is None:
            num_neighbors = self.num_neighbors

        #preprocessing_data(data)

        data.group_mask = torch.zeros(len(data.x), dtype=torch.int8)
        unique_groups = np.unique(data.group_mask)
        
        self.predict_loader = predict_loader
        if predict_loader is None:
            self.predict_loader = create_loaders(
                data=data,
                num_neighbors=num_neighbors,
                batch_size=batch_size,
                input_nodes=test_mask,
                groups_names=self.groups_names,
                group_mask=data.group_mask,
                unique_groups=unique_groups,
                disable=not self.verbose,
            )
        elif "index" not in self.data.keys:
            raise Exception("please, add field 'index' to your data before initializing loaders. Look at notebooks for Example")

        test_metric, test_preds, test_indices = eval_epoch(
            self.best_trainer._model,  # _model from best_trainer
            self.predict_loader,  # loader for prediction
            self.best_trainer.device,
            self.target_names,
            self.groups_names,
            postfix="predict",
            use_edge_attr=self.best_trainer._use_edge_attr,
            tqdm_disable=False,
            fill_value=self.best_trainer.fill_value,
            count_metrics=False,
            log_all_metrics=False,
            log_metric=self.verbose,
            embedding_data=self.embedding_data,
        )

        return test_preds, test_indices


class MultiRunner:
    """
    Runner for heterogeneous graph

    Args:
    data (Data): A data object describing a homogeneous graph. The data object can hold node-level,
    link-level and graph-level attributes. In general, Data tries to mimic the behavior of a regular Python
    dictionary. In addition, it provides useful functionality for analyzing graph structures, and provides basic
    PyTorch tensor functionalities.
    https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs
    config (DictConfig): Config. Defaults to None.
    config_path (str): Path to config. Defaults to None.
    overrides (list): Own params. Can ba params from configs and overrides. Defaults to None.
    train_size (int): Size for train data. Defaults to None.
    test_size (int): Size for test data. Defaults to None.
    seed (int): Seed param for training. Defaults to None.
    train_idx (list): Indices for train data. Defaults to None.
    test_idx (list): Indices for test data. Defaults to None.

    """

    def __init__(
        self,
        data: Optional[Data] = None,
        config: Optional[DictConfig] = None,
        config_path: Optional[str] = None,
        overrides: Optional[List] = None,
        train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        seed: Optional[int] = None,
        train_idx: Optional[List[int]] = None,
        test_idx: Optional[List[int]] = None,
        embedding_data: bool = False,
        **kwargs,
    ) -> None:
        if config is None:
            if config_path is None:
                config_path = os.path.join(
                    os.path.dirname(__file__), "./config/full.yaml"
                )
            config = create_config2(
                config=config_path, overrides=overrides, path_base="cfg"
            )

        cfg = OmegaConf.to_container(config, resolve=True)
        self.cfg = cfg
        self.data = data
        self.test_size = test_size
        self.train_size = train_size
        self.seed = seed
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.node_feature_indices = cfg["data"]["node_feature_indices"]
        self.target_names = cfg["training"]["targets"]
        self.groups_names = cfg["data"]["groups_names"]
        self.target_weights = cfg["training"]["loss"]["target_weights"]
        self.read_edge_attr = cfg["data"].get("read_edge_attr", True)
        self.batch_size = cfg["training"]["batch_size"]
        self.group_mask_col = cfg["data"]["group_mask_col"]
        self.label_mask_col = cfg["data"]["label_mask_col"]
        self.label_cols = cfg["data"]["label_cols"]
        self.label_index_col = cfg["data"]["label_index_col"]
        self.edge_index_cols = cfg["data"]["edge_index_cols"]
        self.num_neighbors = cfg["training"]["num_neighbors"]
        self.features_edges_names = cfg["data"].get("features_edges")
        self.group_names_node_features = cfg["data"]["features"]
        self.metrics = cfg["metrics"]
        self.embedding_data = embedding_data
        self.chkpt_dir = (
            pathlib.Path(cfg["logging"]["checkpoint_dir"]) / str(datetime.now())[:19]
        )
        os.makedirs(self.chkpt_dir, exist_ok=True)

        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.cfg["logging"].get("use_mlflow", False):
            setup_mlflow_from_config(cfg["logging"]["mlflow"])

    def init_loaders(self) -> None:
        if self.batch_size == "auto":
            self._batch_size = get_auto_batch_size(
                [len(v) for _, v in self.group_names_node_features.items()],
                conv_type=self.cfg["model_params"]["conv_type"],
                conv1_aggrs=self.cfg["model_params"]["conv1_aggrs"],
                conv2_aggrs=self.cfg["model_params"].get("conv2_aggrs"),
                conv3_aggrs=self.cfg["model_params"].get("conv3_aggrs"),
                n_hops=self.cfg["model_params"]["n_hops"],
                lin_prep_size_common=self.cfg["model_params"]["lin_prep_size_common"],
                lin_prep_sizes=self.cfg["model_params"]["lin_prep_sizes"],
                edge_attr_repr_sizes=self.cfg["model_params"].get(
                    "edge_attr_repr_sizes"
                ),
                num_edge_features=len(self.cfg["data"].get("features_edges", [])),
                device=self.cfg["training"]["device"],
                num_neighbors=self.cfg["training"]["num_neighbors"],
            )
        else:
            self._batch_size = self.batch_size

        if (self.train_idx is None) or (self.test_idx is None):
            train_idx, test_idx = train_test_split(
                torch.nonzero(self.data.label_mask)[:, 0],
                train_size=self.train_size,
                test_size=self.test_size,
                random_state=self.seed,
                shuffle=True,
            )
            self.train_idx = train_idx
            self.test_idx = test_idx

        unique_groups = np.unique(self.data.group_mask)

        self.train_loader = create_loaders(
            data=self.data,
            num_neighbors=self.num_neighbors,
            batch_size=self._batch_size,
            input_nodes=self.train_idx,
            groups_names=self.groups_names,
            group_mask=self.data.group_mask,
            node_feature_indices=self.node_feature_indices,
            unique_groups=unique_groups,
            groups_features=self.node_feature_indices,
        )

        self.test_loader = create_loaders(
            data=self.data,
            num_neighbors=self.num_neighbors,
            batch_size=self._batch_size,
            input_nodes=self.test_idx,
            groups_names=self.groups_names,
            group_mask=self.data.group_mask,
            node_feature_indices=self.node_feature_indices,
            unique_groups=unique_groups,
            groups_features=self.node_feature_indices,
        )

    def run(self) -> Dict[str, float]:
        if not (hasattr(self, "train_loader") and hasattr(self, "test_loader")):
            self.init_loaders()

        self.trainer = Trainer(
            self.train_loader,
            self.test_loader,
            self.chkpt_dir,
            device=self.cfg["training"]["device"],
            eval_freq=self.cfg["training"]["eval_freq"],
            fill_value=self.cfg["training"]["loss"].get("fill_value"),
            initial_lr=self.cfg["training"].get("initial_lr", 0.01),
            weight_decay=self.cfg["training"].get("weight_decay", 0.0),
            loss_name=self.cfg["training"]["loss"]["name"],
            loss_label_smoothing=self.cfg["training"]["loss"].get(
                "label_smoothing", False
            ),
            loss_target_weights=self.cfg["training"]["loss"].get("target_weights"),
            loss_group_weights=self.cfg["training"]["loss"].get("group_weights"),
            groups_names=self.cfg["data"]["groups_names"],
            mlflow_experiment_name=self.cfg["logging"].get("mlflow_experiment_name"),
            n_epochs=self.cfg["training"].get("n_epochs"),
            scheduler_params=self.cfg["training"].get("scheduler_params", {}),
            scheduler_type=self.cfg["training"].get("scheduler_type"),
            target_names=self.cfg["training"]["targets"],
            use_mlflow=self.cfg["logging"].get("use_mlflow", False),
            tqdm_disable=False,
            **self.cfg["model_params"],
            groups_names_num_features={
                k: len(v) for k, v in self.group_names_node_features.items()
            },
            num_edge_features=len(self.cfg["data"].get("features_edges", [])),
            metrics=self.metrics,
            embedding_data=self.embedding_data,
        )
        result = self.trainer.train()
        return result
