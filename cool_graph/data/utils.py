from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from sklearn.preprocessing import QuantileTransformer


def preprocessing_data(
    data: Optional[Data], fill_value=-100, target_names=None
) -> Dict:

    target_names = []
    target_weights = {}

    for i in range(data.y.shape[1]):
        y_sub = data.y[:, i]
        setattr(data, f"y{i}", y_sub)
        target_names.append(f"y{i}")
        target_weights.update({f"y{i}": 1})

        _target_sizes = {}

        for target_name in target_names:
            classes = getattr(data, target_name).max() + 1
            _target_sizes[target_name] = len(classes[classes != fill_value]) + 1
            target_sizes = list(_target_sizes.values())

    return target_names, target_weights, target_sizes

def compress_x_cat(data: Data) -> Data:

    if not hasattr(data, 'x_cat'):
        return data
    
    def compress(col: torch.Tensor) -> torch.Tensor:
        coords = dict()
        pos = 0
        for val in col:
            if val.item() not in coords:
                coords[val.item()] = pos
                pos += 1
        col = torch.tensor([coords[val.item()] for val in col])
        return col
    
    data.x_cat = torch.transpose(data.x_cat, 0, 1)
    data.x_cat = torch.stack([compress(col) for col in data.x_cat])
    data.x_cat = torch.transpose(data.x_cat, 0, 1)
    
    return data

def add_graph_node_features(
    data: Data,
    stats: List[callable] = [nx.degree_centrality, nx.pagerank],
) -> Data:
    
    G = to_networkx(data)
    columns = []

    for foo in stats:
        column = torch.tensor([value for value in foo(G).values()], dtype=torch.float)
        columns.append(column)
            
    columns = torch.transpose(torch.stack(columns), 0, 1)
    qt = QuantileTransformer(output_distribution='normal')
    columns = torch.tensor(qt.fit_transform(columns), dtype=torch.float)
    
    if hasattr(data, 'x') and data.x is not None:
        data.x = torch.hstack((data.x, columns))
    else:
        data.x = columns
        
    return data

def count_degree(
    G: nx.MultiGraph
) -> List[float]:
    
    degrees = {}
    
    for v, to in G.edges():
        if not v in degrees:
            degrees[v] = 0
        if not to in degrees:
            degrees[to] = 0
        degrees[v] += 1
        degrees[to] += 1
    
    result = []
    for v, to in G.edges():
        result.append(degrees[v] + degrees[to])
    
    return result

def count_common_neighbors(
    G: nx.MultiGraph
) -> List[float]:
    
    graph = {}
    
    for v, to in G.edges():
        if not v in graph:
            graph[v] = set()
        if not to in graph:
            graph[to] = set()
        graph[v].add(to)
        graph[to].add(v)
    
    def common_neighbors(v, to):
        return len(graph[v].intersection(graph[to]))
    
    result = []
    for v, to in G.edges():
        num_common_neighbors = common_neighbors(v, to)
        result.append(num_common_neighbors)
    
    return result

def add_graph_edge_features(
    data: Data,
    stats: List[callable] = [count_degree, count_common_neighbors]
) -> Data:
    
    G = to_networkx(data)
    columns = []

    for foo in stats:
        res = foo(G)
        column = torch.tensor([value for value in res], dtype=torch.float)
        columns.append(column)
            
    columns = torch.transpose(torch.stack(columns), 0, 1)
    qt = QuantileTransformer(output_distribution='normal')
    columns = torch.tensor(qt.fit_transform(columns), dtype=torch.float)
    
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        data.edge_attr = torch.hstack((data.edge_attr, columns))
    else:
        data.edge_attr = columns
        
    return data