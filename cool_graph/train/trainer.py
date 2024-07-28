import gc
import json
import os
import pathlib
import time
import traceback
from typing import Any, Dict, List, Literal, Optional, Union

import mlflow
import numpy as np
import pandas as pd
import torch
from loguru import logger
from mlflow.exceptions import MlflowException

from cool_graph.models import GraphConvGNN, NNConvGNN
from cool_graph.train.helpers import add_prefix_to_dict_keys, eval_epoch, train_epoch
from cool_graph.train.metrics import get_metric


class Trainer(object):
    def __init__(
        self,
        list_loader_train: List[torch.utils.data.DataLoader],
        list_loader_test: List[torch.utils.data.DataLoader],
        checkpoint_dir: Union[str, pathlib.PosixPath],
        device: str = "cuda:0",
        eval_freq: int = 5,
        fill_value: Union[int, float] = -100,
        initial_lr: float = 0.0023,
        weight_decay: float = 0.001,
        loss_name: str = "CrossEntropyLoss",
        loss_label_smoothing: bool = False,
        loss_target_weights: Optional[Dict[str, Union[int, float]]] = None,
        loss_group_weights: Optional[List[float]] = None,
        groups_names: Optional[Dict[int, str]] = None,
        groups_names_num_features: Optional[Dict[str, int]] = None,
        groups_names_num_cat_features: Optional[Dict[str, int]] = None,
        num_edge_features: Optional[int] = None,
        main_metric_name: str = "main_metric",
        mlflow_experiment_name: Optional[str] = None,
        n_epochs: int = 10,
        scheduler_params: Dict[Literal["milestones", "gamma"], int] = {
            "milestones": [10, 20, 35, 50, 70, 90, 105],
            "gamma": 0.25,
        },
        scheduler_type: str = "MultiStepLR",
        target_names: List[str] = ["y"],
        target_sizes: Optional[List[int]] = None,
        cat_features_sizes: Optional[List[int]] = None,
        use_mlflow: bool = False,
        tqdm_disable=False,
        conv_type: Literal["NNConv", "GraphConv"] = "NNConv",
        metrics: Optional[float] = None,
        log_all_metrics: bool = True,
        log_metric: bool = True,
        embedding_data: bool = False,
        **model_params,
    ) -> None:
        """
        Training model (GraphConv or NNConv).
        Class that training / logging / saving model. Using train_epoch
        and eval_epoch from helpers.py in training loop below.

        Args:
            list_loader_train (List[torch.utils.data.DataLoader]): Train list with Data loader. Combines a dataset
            and a sampler, and provides an iterable over the given dataset.
            https://pytorch.org/docs/stable/data.html
            list_loader_test (List[torch.utils.data.DataLoader]): Test list with Data loader. Combines a dataset
            and a sampler, and provides an iterable over the given dataset.
            https://pytorch.org/docs/stable/data.html
            checkpoint_dir (Union[str, pathlib.PosixPath]): Path for training checkpoints
            device (_type_, optional): The device is an object representing the device on
            which a torch.Tensor is or will be allocated.. Defaults to "cuda:0".
            eval_freq (int, optional): Number of epoch group. Defaults to 5.
            fill_value (Union[int, float], optional): If value is None. Defaults to -100.
            initial_lr (float, optional): The learning rate param for Optimization. Defaults to 0.0023.
            weight_decay (float, optional): weight decay (L2 penalty). Defaults to 0.001.
            loss_name (str, optional): This criterion computes the cross entropy loss between
            input logits and target. Defaults to "CrossEntropyLoss".
            https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            loss_label_smoothing (bool, optional): If set True, use label smoothing. Defaults to False.
            loss_target_weights (Optional[Dict[str, Union[int, float]]], optional): Weights for targets. Defaults to None.
            loss_group_weights (Optional[List[float]], optional): Weights for groups. Defaults to None.
            groups_names (Optional[Dict[int, str]], optional): List with group names in nodes. Defaults to None.
            groups_names_num_features (Optional[Dict[str, int]], optional): Number of feats in groups in nodes. Defaults to None.
            num_edge_features (Optional[int], optional): Number of feats on edges. Defaults to None.
            main_metric_name (str, optional): Main metric for maximaze. Defaults to "main_metric".
            mlflow_experiment_name (Optional[str], optional): Name of mlflow experiment. Defaults to None.
            n_epochs (int, optional): Number of epochs. Defaults to 10.
            scheduler_params (Dict, optional): Milestones (list) – List of epoch indices. Must be increasing.
            gamma (float) – Multiplicative factor of learning rate decay.
            Defaults to { "milestones": [10, 20, 35, 50, 70, 90, 105], "gamma": 0.25, }.
            scheduler_type (str, optional): Decays the learning rate of each parameter group
            by gamma once the number of epoch reaches one of the milestones. Defaults to "MultiStepLR".
            target_names (List[str], optional): List of target names. Defaults to ["y"].
            target_sizes (Optional[List[int]], optional): Size of list with target. Defaults to None.
            use_mlflow (bool, optional): If set True, use MLFlow. Defaults to False.
            tqdm_disable (bool, optional): Display progress. Defaults to False.
            conv_type (Literal[NNConv, GraphConv], optional): The graph neural network operator. Defaults to "NNConv".
            metrics (float, optional): Metrics. Defaults to None.
            log_all_metrics (bool, optional): If set True, logging all metrics. Defaults to True.

        Raises:
            NotImplementedError: _description_
        """
        for key, value in locals().items():
            setattr(self, key, value)

        self._metrics = {}
        self._main_metric = {}
        if isinstance(metrics, str):
            metrics = [metrics]
        if isinstance(
            metrics,
            (
                list,
                tuple,
            ),
        ):
            metrics = {name: metrics for name in target_names}

        for k, names in metrics.items():
            self._metrics[k] = {name: get_metric(name) for name in names}
            self._main_metric[k] = names[0]

        os.makedirs(checkpoint_dir, exist_ok=True)

        torch.cuda.empty_cache()
        gc.collect()

        if conv_type == "NNConv":
            self._model = NNConvGNN(
                **model_params,
                target_names=target_names,
                target_sizes=target_sizes,
                cat_features_sizes=cat_features_sizes,
                groups_names=groups_names,
                groups_names_num_features=groups_names_num_features,
                groups_names_num_cat_features=groups_names_num_cat_features,
                num_edge_features=num_edge_features,
            )
        elif conv_type == "GraphConv":
            self._model = GraphConvGNN(
                **model_params,
                target_names=target_names,
                target_sizes=target_sizes,
                cat_features_sizes=cat_features_sizes,
                groups_names=groups_names,
                groups_names_num_features=groups_names_num_features,
                groups_names_num_cat_features=groups_names_num_cat_features,
                num_edge_features=num_edge_features,
            )
        else:
            raise NotImplementedError(f"{conv_type} is not implemented")

        self._model.to(device)

        self._optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=initial_lr,
            weight_decay=weight_decay,
        )

        self._loss_criteria = getattr(torch.nn, loss_name)(
            reduction="none", label_smoothing=loss_label_smoothing
        )
        self._use_edge_attr = conv_type == "NNConv"

        self._scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(
            self._optimizer, **scheduler_params
        )

        self._best_loss = {main_metric_name: -np.inf}

        self._train_run_lst = []
        self._test_metric_lst = []
        self._train_metric_lst = []

    def train(self, start_epoch: int = 0, end_epoch: Optional[int] = None) -> Dict[
        Literal[
            "best_loss", "global_calc_time", "train_loss", "test_metric", "train_metric"
        ],
        float,
    ]:
        """
        Training model and logging metrics.
        """
        if end_epoch is None:
            end_epoch = self.n_epochs

        self.global_start_time = time.time()

        if self.use_mlflow:
            mlflow.end_run()
            mlflow.set_experiment(self.mlflow_experiment_name)
            mlflow.start_run()
            mlflow.log_params(
                {
                    "LossCriteria": self._loss_criteria,
                    "checkpoint_dir": self.checkpoint_dir,
                    **self.model_params,
                }
            )

        for epoch in range(start_epoch, end_epoch):
            self.epoch = epoch
            # TRAIN
            train_run = train_epoch(
                self._model,
                self.list_loader_train,
                self.device,
                self._optimizer,
                self._use_edge_attr,
                target_weights=self.loss_target_weights,
                loss_criteria=self._loss_criteria,
                group_weights=self.loss_group_weights,
                tqdm_disable=self.tqdm_disable,
            )
            train_run["lr"] = self._optimizer.param_groups[0]["lr"]
            self.mlflow_log_metrics(
                metrics=add_prefix_to_dict_keys(train_run, "run_"), step=epoch
            )
            train_run["epoch"] = epoch
            self._train_run_lst.append(train_run)
            with open(
                os.path.join(self.checkpoint_dir, "train_running_loss.txt"), "a"
            ) as f:
                json.dump(train_run, f)
                f.write("\n")

            # calc metrics and perform scheduler step
            if (epoch - 0) % self.eval_freq == 0:
                # calc metrics
                # test
                #                 logger.info("\nEpoch {:03d}: ".format(epoch))
                test_metric, test_preds, indeces = eval_epoch(
                    self._model,
                    self.list_loader_test,
                    self.device,
                    self.target_names,
                    self.groups_names,
                    postfix=f"epoch {epoch} test",
                    use_edge_attr=self._use_edge_attr,
                    tqdm_disable=self.tqdm_disable,
                    fill_value=self.fill_value,
                    metrics=self._metrics,
                    main_metric=self._main_metric,
                    log_all_metrics=self.log_all_metrics,
                    log_metric=self.log_metric,
                    embedding_data=self.embedding_data,
                )
                test_tasks = test_metric["tasks"]
                self.mlflow_log_metrics(
                    metrics=add_prefix_to_dict_keys(test_metric, "test_"), step=epoch
                )
                test_metric["epoch"] = epoch
                self._test_metric_lst.append(test_metric)
                with open(
                    os.path.join(self.checkpoint_dir, "test_metric.txt"), "a"
                ) as f:
                    json.dump(test_metric, f)
                    f.write("\n")

                # train
                #                 logger.info("Epoch {:03d}: ".format(epoch))
                train_metric, train_preds, indeces = eval_epoch(
                    self._model,
                    self.list_loader_train,
                    self.device,
                    self.target_names,
                    self.groups_names,
                    postfix=f"epoch {epoch} train",
                    use_edge_attr=self._use_edge_attr,
                    tqdm_disable=self.tqdm_disable,
                    metrics=self._metrics,
                    main_metric=self._main_metric,
                    log_all_metrics=self.log_all_metrics,
                    log_metric=self.log_metric,
                    embedding_data=self.embedding_data,
                )
                train_tasks = train_metric["tasks"]
                self.mlflow_log_metrics(
                    metrics=add_prefix_to_dict_keys(train_metric, "train_"), step=epoch
                )
                train_metric["epoch"] = epoch
                self._train_metric_lst.append(train_metric)
                with open(
                    os.path.join(self.checkpoint_dir, "train_metric.txt"), "a"
                ) as f:
                    json.dump(train_metric, f)
                    f.write("\n")

                # save model
                checkpoint_file = os.path.join(
                    self.checkpoint_dir, f"state_dict_{epoch:0>4d}.pt"
                )
                torch.save(self._model.cpu().state_dict(), checkpoint_file)
                self._model.to(self.device)

                if (
                    test_metric[self.main_metric_name]
                    > self._best_loss[self.main_metric_name]
                ):
                    self._best_loss = test_metric
                    self._best_loss["epoch"] = epoch
                    checkpoint_file = os.path.join(
                        self.checkpoint_dir, "state_dict_best.pt"
                    )
                    torch.save(self._model.cpu().state_dict(), checkpoint_file)
                    self._model.to(self.device)
                    with open(
                        os.path.join(self.checkpoint_dir, "best_loss.txt"), "w"
                    ) as f:
                        json.dump(self._best_loss, f, indent=4)

                self.mlflow_log_metrics(
                    {
                        "best_epoch": self._best_loss["epoch"],
                        f"best_{self.main_metric_name}": self._best_loss[
                            self.main_metric_name
                        ],
                    },
                    step=epoch,
                )

            if self.scheduler_type == "ReduceLROnPlateau":
                self._scheduler.step(train_run["total_loss"])
                if (
                    self._optimizer.param_groups[0]["lr"]
                    <= self.scheduler_params["min_lr"]
                ):
                    break
            else:
                self._scheduler.step()

        self.global_calc_time = time.time() - self.global_start_time
        train_loss = pd.DataFrame(self._train_run_lst)
        test_metric = pd.DataFrame(self._test_metric_lst)
        train_metric = pd.DataFrame(self._train_metric_lst)

        self.mlflow_log_metrics(
            metrics=add_prefix_to_dict_keys(self._best_loss, "best_")
        )
        self.mlflow_log_metrics({"global_calc_time": self.global_calc_time})

        if self.use_mlflow:
            mlflow.end_run()
        torch.cuda.empty_cache()

        return {
            "best_loss": self._best_loss,
            "global_calc_time": self.global_calc_time,
            "train_loss": train_loss,
            "test_metric": test_metric,
            "train_metric": train_metric,
            "train_preds": train_preds,
            "test_preds": test_preds,
            "train_tasks": train_tasks,
            "test_tasks": test_tasks,
        }

    def mlflow_log_metrics(
        self, metrics: Dict[str, Any], step: Optional[int] = None
    ) -> None:
        if self.use_mlflow:
            try:
                mlflow.log_metrics(metrics, step)
            except MlflowException as e:
                save_str_e = traceback.format_exc()
                logger.info(
                    "Epoch {:03d}::\nCaught exception:\n{}".format(
                        self.epoch, save_str_e
                    )
                )
                with open(
                    os.path.join(self.checkpoint_dir, "MlflowExceptions.txt"), "a"
                ) as f:
                    f.write(
                        "Epoch {:03d}::\nCaught exception:\n{}".format(
                            self.epoch, save_str_e
                        )
                    )
