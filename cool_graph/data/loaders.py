from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm


def create_loaders(
    data: Data = None,
    node_features: torch.FloatTensor = None,
    edge_features: torch.FloatTensor = None,
    edge_index: torch.LongTensor = None,
    read_edge_attr: bool = None,
    num_neighbors: List[int] = None,
    batch_size: int = None,
    group_mask: torch.LongTensor = None,
    groups_features: Dict[int, List[int]] = None,
    groups_names: Dict[int, str] = None,
    label_mask: torch.BoolTensor = None,
    index: torch.LongTensor = None,
    targets: Dict[str, torch.Tensor] = None,
    input_nodes: Optional[List] = None,
    node_feature_indices: Optional[List] = None,
    unique_groups: Optional[int] = None,
) -> List[torch.utils.data.DataLoader]:
    """
    Creating list loaders.

    Args:
        node_features (torch.FloatTensor): features on nodes on FloatTensor
        edge_features (torch.FloatTensor): features on edge on FloatTensor
        edge_index (torch.LongTensor): edge indices
        read_edge_attr (bool): if set True - read edge features.
        num_neighbors (List[int]): Number of neighbors are sampled for each node in each iteration.
        batch_size (int): Numbers of samples per batch to load.
        group_mask (torch.LongTensor): Mask for groups in nodes.
        groups_features (Dict[int, List[int]]): Features in groups in nodes.
        groups_names (Dict[int, str]): Name of featutes in groups in nodes.
        label_mask (torch.BoolTensor): Mask for label.
        index (torch.LongTensor): index
        targets (Dict[str, torch.Tensor]): Labels.

    Returns:
        List[torch.utils.data.DataLoader]: Created DataLoader object. https://pytorch.org/docs/stable/data.html
    """
    unique_groups = np.unique(group_mask)
    try:
        set(unique_groups).issubset(set(groups_features.keys()))
    except Exception as ex:
        raise ValueError(
            f"""Group mask values should be a subset of feature groups keys"""
        )

    try:
        set(groups_features).issubset(set(groups_names.keys()))
    except Exception as ex:
        raise ValueError(
            f"""Feature groups keys should be a subset of feature_groups_names"""
        )
    if data is None:
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features if read_edge_attr else None,
            group_mask=group_mask,
            label_mask=label_mask,
            index=index,
            **targets,
        )
        input_nodes = torch.nonzero(label_mask)[:, 0]

    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=True,
        input_nodes=input_nodes,
    )

    list_loader = []
    for sampled_data in tqdm(loader, desc="Sample data"):
        sampled_data.label_mask[sampled_data.batch_size :] = False

        for group in unique_groups:
            name = groups_names[group]
            mask = sampled_data.group_mask == group
            features = groups_features[group]
            setattr(sampled_data, name, sampled_data.x[mask][:, features])

        del sampled_data.x

        list_loader.append(sampled_data)

    return list_loader
