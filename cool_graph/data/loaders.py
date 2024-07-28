from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm


def create_loaders(
    data: Optional[Data] = None,
    node_features: Optional[torch.FloatTensor] = None,
    edge_features: Optional[torch.FloatTensor] = None,
    edge_index: Optional[torch.LongTensor] = None,
    read_edge_attr: Optional[bool] = None,
    num_neighbors: Optional[List[int]] = None,
    batch_size: Optional[int] = None,
    group_mask: Optional[torch.LongTensor] = None,
    groups_features: Optional[Dict[int, List[int]]] = None,
    groups_names: Optional[Dict[int, str]] = None,
    label_mask: Optional[torch.BoolTensor] = None,
    index: Optional[torch.LongTensor] = None,
    targets: Optional[Dict[str, torch.Tensor]] = None,
    input_nodes: Optional[List] = None,
    node_feature_indices: Optional[List] = None,
    unique_groups: Optional[int] = None,
    hetero_data: Optional[bool] = False,
    disable: bool = False,
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

    if hetero_data:
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
        
    if "index" not in data.keys:
        data.index = torch.tensor(range(0, len(data.x)))

    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=True,
        input_nodes=input_nodes,
    )

    list_loader = []

    for sampled_data in tqdm(loader, desc="Sample data", disable=disable):
        sampled_data.label_mask[sampled_data.batch_size :] = False
        for group in unique_groups:
            name = groups_names[group]
            mask = sampled_data.group_mask == group
            if hetero_data:
                features = groups_features[group]
                setattr(sampled_data, name, sampled_data.x[mask][:, features])
            else:
                setattr(sampled_data, name, sampled_data.x[mask])

        list_loader.append(sampled_data)

    return list_loader
