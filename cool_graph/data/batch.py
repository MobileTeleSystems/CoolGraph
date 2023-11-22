from typing import Dict, List, Literal, Optional

import numpy as np
import torch


def get_auto_batch_size(
    groups_num_features: List[int],
    conv_type: Optional[Literal["NNConv", "GraphConv"]] = None,
    conv1_aggrs: Optional[Dict[Literal["mean", "max", "add"], int]] = None,
    conv2_aggrs: Optional[Dict[Literal["mean", "max", "add"], int]] = None,
    conv3_aggrs: Optional[Dict[Literal["mean", "max", "add"], int]] = None,
    n_hops: Optional[int] = None,
    lin_prep_size_common: Optional[int] = None,
    lin_prep_sizes: Optional[List[int]] = None,
    edge_attr_repr_sizes: Optional[List[int]] = None,
    num_edge_features: Optional[int] = None,
    device: str = "cuda:0",
    num_neighbors: Optional[List[int]] = None,
) -> int:
    """
    Аutomatic batch size calculation.
    Depending on model size and free GPU memory.

    Args:
        groups_num_features (List[int]): Number of feats in groups on nodes.
        conv_type (Literal[NNConv, GraphConv]): Model type
        conv1_aggrs (Dict[Literal[mean, max, add], int]]):
        An aggregation per features across a set of elements in conv layer 1. Defaults to None.
        conv2_aggrs (Dict[Literal[mean, max, add], int]]):
        An aggregation per features across a set of elements in conv layer 2. Defaults to None.
        conv3_aggrs (Dict[Literal[mean, max, add], int]]):
        An aggregation per features across a set of elements in conv layer 3. Defaults to None.
        n_hops (int): Hop with neighbors. Defaults to None.
        lin_prep_size_common (int): Size of linear layer (in). Defaults to None.
        lin_prep_sizes (int): Size of linear layer (out). Defaults to None.
        edge_attr_repr_sizes (List[int]): Size of layer of edges attributes. Defaults to None.
        num_edge_features (int): Number of feats on edges. Defaults to None.
        device (str): The current GPU memory usage. Defaults to "cuda:0".
        num_neighbors (List[int]): Number of neighbors are sampled for each node in each iteration. Defaults to None.

    Returns:
        batch_size (int): Numbers of samples per batch to load.
    """
    if lin_prep_sizes is None:
        lin_prep_sizes = []
    if device is None:
        device = "cuda:0"

    hop1_size = sum(conv1_aggrs.values())
    hop2_size = sum(conv2_aggrs.values()) if n_hops >= 2 else 0
    hop3_size = sum(conv3_aggrs.values()) if n_hops == 3 else 0

    max_size_node = max(
        *groups_num_features,
        lin_prep_size_common,
        *lin_prep_sizes,
        hop1_size,
        hop2_size,
        hop3_size,
    )

    max_size_edge = 0
    if conv_type == "NNConv":
        max_size_edge = max(
            *edge_attr_repr_sizes,
            num_edge_features,
        )

    max_size = max_size_node + max_size_edge * 1.5

    try:
        all([n != -1 for n in num_neighbors])
    except Exception as ex:
        raise ValueError(
            f"""
            Found -1, Need to know max neighbors per hop.
            """
        )
    m_neighbors = np.prod(num_neighbors)

    free_memory = torch.cuda.mem_get_info(device=device)[0] / (1024**3)  # GB

    floats_per_node_ = 320000
    batch_size_ = 250
    memory_reserved_max_ = 3.8

    batch_size = (
        0.5
        * batch_size_
        * floats_per_node_
        / (m_neighbors * max_size)
        * (free_memory / memory_reserved_max_)
    )

    if conv_type == "NNConv":
        batch_size /= edge_attr_repr_sizes[-1] * 4

    batch_size = int(batch_size)

    return batch_size
