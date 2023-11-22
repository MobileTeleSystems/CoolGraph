from typing import Dict, List, Optional

import torch
import torch_geometric
from loguru import logger


def get_train_loss(
    data: torch_geometric.data.Data,
    outs: Dict[str, torch.FloatTensor],
    target_weights: Optional[Dict[str, float]] = None,
    loss_criteria: torch.nn.modules.loss._Loss = torch.nn.CrossEntropyLoss(
        reduction="none"
    ),
    group_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """Multi-target Batch loss.

    Args:
        data (torch_geometric.data.Data): The data object can hold node-level, link-level and graph-level attributes.
        outs (Dict[str, torch.FloatTensor]): Out tensor.
        target_weights (Optional[Dict[str, float]], optional): Weights per target. Defaults to None.
        loss_criteria (torch.nn.modules.loss._Loss, optional): This criterion computes the cross entropy loss between
        input logits and target. Defaults to torch.nn.CrossEntropyLoss(reduction="none").
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        group_weights (Optional[List[float]], optional): Weights per groups in node. Defaults to None.

    Returns:
        torch.Tensor: Loss
    """
    loss = 0
    for trgt in target_weights:
        loss += get_train_loss_product(
            data,
            outs,
            loss_criteria,
            trgt,
            target_weights,
            group_weights,
        )
    loss /= data.batch_size
    return loss


def get_train_loss_product(
    data: torch_geometric.data.Data,
    outs: Dict[str, torch.FloatTensor],
    loss_criteria: torch.nn.modules.loss._Loss,
    key: str,
    target_weights: Dict[str, float],
    group_weights: List[float],
    fill_value: Optional[int] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Loss per target

    Args:
        data (torch_geometric.data.Data): The data object can hold node-level, link-level and graph-level attributes.
        outs (Dict[str, torch.FloatTensor]): Out tensor.
        loss_criteria (torch.nn.modules.loss._Loss, optional): This criterion computes the cross entropy loss between
        input logits and target. Defaults to torch.nn.CrossEntropyLoss(reduction="none").
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        key (str): [description]
        target_weights (Dict[str, float]): [description]
        group_weights (List[float]): [description]

    Returns:
        torch.Tensor: [description]
    """
    y = data[key]
    if target_weights[key] == 0:
        return 0
    out = outs[key]
    wh = data.label_mask & (y != fill_value)

    if wh.any() is False:
        return 0

    y = y[wh]
    if len(y) == 0:
        return 0
    out = out[wh]
    mask = data.group_mask[wh]

    losses = loss_criteria(out, y)
    unique = torch.unique(mask).detach().cpu().numpy()
    if group_weights is None:
        try:
            group_weights = [1] * (max(unique) + 1)
        except Exception as ex:
            logger.error(f"unique {unique}")
            logger.error(f"mask {mask}")
            logger.error(f"len {len(y)}")
            raise ex

    for group in unique:
        losses[mask == group] *= group_weights[group]

    return losses.sum() * target_weights[key]
