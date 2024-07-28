from collections import defaultdict
from time import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor, cat, nn, no_grad, optim, unique, utils
from tqdm import tqdm

from cool_graph.train.loss import get_train_loss


def train_epoch(
    model: nn.Module,
    list_loader: List[utils.data.DataLoader],
    device: str,
    optimizer: optim.Optimizer,
    use_edge_attr: bool = True,
    target_weights: Optional[Dict[str, Union[int, float]]] = None,
    group_weights: Optional[List[float]] = None,
    loss_criteria: nn.modules.loss._Loss = nn.CrossEntropyLoss(),
    tqdm_disable: bool = False,
) -> Dict[Literal["total_loss", "calc_time"], float]:
    """
    Training one epoch. Using in training loop.

    Args:
        model (nn.Module): Neural Network model type.
        list_loader (List[utils.data.DataLoader]): List with Data loader. Data Loader combines a dataset
        and a sampler, and provides an iterable over the given dataset.
        https://pytorch.org/docs/stable/data.html
        device (str): The device is an object representing the device on which a torch.Tensor is or will be allocated.
        optimizer (optim.Optimizer): torch.optim is a package implementing various optimization algorithms.
        use_edge_attr (bool, optional): If graph edges have features, it can be use in training. Defaults to True.
        target_weights (Optional[Dict[str, Union[int, float]]]): Weights for targets. Defaults to None.
        group_weights  (Optional[List[float]]): Weights for groups. Defaults to None.
        loss_criteria: This criterion computes the cross entropy loss between input logits and target.
        Defaults to "CrossEntropyLoss". https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        tqdm_disable (bool, optional): Display progress. Defaults to True.

    Returns:
        Dict[Literal["total_loss", "calc_time"], float]: Total loss and time info.
    """

    start_time = time()
    model.train()
    total_loss = 0
    indices = np.arange(len(list_loader))
    indices = np.random.permutation(indices)
    for i in tqdm(indices, leave=False, disable=tqdm_disable):
        data_cut = list_loader[i]
        sampled_data = data_cut.detach().clone().to(device)
        optimizer.zero_grad()
        out, scores = model(sampled_data)
        loss = get_train_loss(
            sampled_data,
            out,
            target_weights=target_weights,
            loss_criteria=loss_criteria,
            group_weights=group_weights,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        total_loss += loss.detach().item() / len(list_loader)

    calc_time = time() - start_time
    return {"total_loss": total_loss, "calc_time": calc_time}


def eval_epoch(
    model: nn.Module,
    list_loader: List[utils.data.DataLoader],
    device: str,
    target_names: List[str],
    groups_names: List[str],
    mode: str = "eval",
    postfix: str = "",
    use_edge_attr: bool = True,
    log_metric: bool = True,
    count_metrics: bool = True,
    tqdm_disable: bool = True,
    fill_value: Any = -100,
    metrics: Optional[Dict[str, Dict[str, Callable]]] = None,
    main_metric: Optional[Dict[str, str]] = None,
    log_all_metrics: bool = True,
    embedding_data: bool = False,
) -> Dict[str, float]:
    """
    Getting metrics. Using in training loop.

    Args:
        model (nn.Module): Neural Network model type.
        list_loader (List[utils.data.DataLoader]): Data loader. Combines a dataset
        and a sampler, and provides an iterable over the given dataset.
        https://pytorch.org/docs/stable/data.html
        device (str): The device is an object representing the device on which a torch.Tensor is or will be allocated.
        target_names (List[str]): List with target names.
        groups_names (List[str]): List with group names in nodes.
        mode (str, optional): Dropout and batch normalization layers
        to evaluation mode before running. Defaults to "eval".
        postfix (str, optional): Postfix for logging. Defaults to "".
        use_edge_attr (bool, optional): If graph edges have features, it can be use in training. Defaults to True.
        log_metric (bool, optional): If set True, logging metrics. Defaults to True.
        tqdm_disable (bool, optional): Display progress. Defaults to True.
        fill_value (Any, optional): If value is None. Defaults to -100.
        metrics (Optional[Dict[str, Dict[str, Callable]]], optional): Dict with training metrics. Defaults to None.
        main_metric (Optional[Dict[str, str]], optional): Main metric if training. Defaults to None.
        log_all_metrics (bool, optional): If set True, all metrics are logging. Defaults to False.

    Returns:
        results (Dict[str, float]): Dict with metrics
        preds (Dict[str, float]): Dict with predictions probabilities on each task
        indices (torch.tensor(int)): indices matching nodes from preds to labels from data
    """

    logger.remove()
    logger.add(
        sink=sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss} - {message}",
    )
    
    start_time = time()
    if mode == "eval":
        model.eval()
    else:
        model.train()

    outs = defaultdict(list)
    ys = defaultdict(list)
    preds = defaultdict(list)
    scores = defaultdict(list)
    group_masks = []
    indices = []

    with no_grad():
        for data_cut in tqdm(list_loader, leave=False, disable=tqdm_disable):
            sampled_data = data_cut.detach().clone().to(device)
            wh = data_cut.label_mask
            indices.extend(sampled_data.index.detach().cpu().numpy()[wh])
            if embedding_data:
                out, scores, embs = model.forward_with_embs(sampled_data)
                for key in embs.keys():
                    preds[key].extend(embs[key][wh])
                for key in scores.keys():
                    preds[key].extend(scores[key][wh])
                preds = dict(preds)
            else:
                out, scores = model(sampled_data)
                for key in scores.keys():
                    preds[key].extend(np.array(scores[key][wh]))
                preds = dict(preds)

            for key in out.keys():
                outs[key].append(out[key].detach().cpu()[wh])
                if count_metrics:
                    ys[key].append(data_cut.__getattr__(key)[wh])
            group_masks.append(data_cut.group_mask[wh])

    for key in preds.keys():
        preds[key] = np.array([pred for pred in preds[key]])
        
    if not count_metrics:
        return {}, preds, indices
        
    for key in outs.keys():
        outs[key] = cat(outs[key])
        ys[key] = cat(ys[key])
    
    group_masks = cat(group_masks)
    unique_groups = unique(group_masks).detach().cpu().numpy()
    results = {key: defaultdict(dict) for key in metrics.keys()}
    tasks = {key: defaultdict(dict) for key in metrics.keys()}

    for key, metric_dict in metrics.items():  # key = target_name
        y_wh = ys[key] != fill_value
        for group in unique_groups:  # by groups
            groups_name = groups_names[group]
            wh = (group_masks == group) & y_wh
            y = ys[key][wh]
            out = outs[key][wh]
            for metric_name, func in metric_dict.items():
                if wh.any().item() is False:
                    value = np.nan
                else:
                    value = float(func(out, y))
                    if len(groups_name) >= 2:
                        results[key][metric_name][groups_name] = value
                    else:
                        results[key][metric_name][key] = value
                        tasks[key][metric_name] = value

    for key in tasks.keys():
        tasks[key] = dict(tasks[key])

    main_metric_val = []
    for key, metric_name in main_metric.items():
        main_metric_val.extend(list(results[key][metric_name].values()))
    main_metric_val = np.nanmean(main_metric_val)

    if not log_all_metrics:
        res = {}
        for metric_name in list(metrics.values())[0].keys():
            metric_val = []
            for key in metrics.keys():
                metric_val.extend(list(results[key][metric_name].values()))
            metric_val = np.nanmean(metric_val)
            res[metric_name] = metric_val
        results = res
    else:
        results = flat_metric(results)

    calc_time = (time() - start_time) / 60
    results["calc_time"] = calc_time
    results["main_metric"] = main_metric_val

    for i in results.keys():
        results[i] = round(results[i], 3)

    if log_metric:
        logger.info(f"{postfix}:\n {results}")

    results["tasks"] = tasks

    return results, preds, indices


def flat_metric(results) -> Dict[str, Union[int, float]]:
    out = {}
    for key in results.keys():
        for metric_name in results[key].keys():
            for groups_name, value in results[key][metric_name].items():
                out[f"{key}__{metric_name}__{groups_name}"] = value
    return out


def add_prefix_to_dict_keys(d: dict, prefix: str = ""):
    return {prefix + k: v for k, v in d.items()}
