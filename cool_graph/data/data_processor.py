import glob
import os
from typing import Dict, List, Optional

import pyarrow.parquet as pq
import torch

from cool_graph.data.loaders import create_loaders


class RawDataProcessor:
    """
    Preprocessing datasets.

    Args:
        groups_names (Dict[int, str]): Name of groups in nodes.
        group_names_node_features (Dict[str, List[str]]): Name of features in groups in nodes.
        mon_nodes_path (str): path to nodes
        mon_edges_path (str): path to edges
        mon_labels_path (str): path to labels
        edge_index_cols (List[str]): columns of edge index in dataset
        label_index_col (str): columns of label index in dataset
        label_mask_col (str): mask of label columns
        read_edge_attr (bool): is set True - read edge features. Default to True.
        group_mask_col (str): Mask for group in data. Default to None.
        features_edges_names (List[str]): List of features on edge. Default to None.
        label_cols (List[str]): List of label columns. Default to None.
        target_names (List[str]): List of target names. Default to None.
    """

    @staticmethod
    def _check_cols_in_parquet(columns: List[str], path: str) -> bool:
        """Cheking colomns in parquet files.

        Args:
            columns (List[str]): columns of dataset
            path (str): path to dataset

        Raises:
            ValueError: if there is no any files with parquet extension
            ValueError: if there is no path with parquet extension

        Returns:
            bool: True if columns and path are right
        """
        if columns:
            set_cols = set(columns if type(columns) == list else [columns])
            try:
                parquet_file = [path] if path.endswith(".parquet") else []
                parquet_file = (
                    parquet_file
                    + glob.glob(os.path.join(path, "*.parquet"), recursive=True)
                    + glob.glob(os.path.join(path, "**/*.parquet"), recursive=True)
                )
                parquet_file = parquet_file[0]
            except Exception as ex:
                raise ValueError(
                    f"""
                    Couldn't find any files with parquet extension in {path}\n
                    Original exception: \n
                    {str(ex)}
                """
                )
            pqt_cols = set(pq.read_schema(parquet_file).names)
            if not set_cols.issubset(pqt_cols):
                diff = set_cols - pqt_cols
                raise ValueError(
                    f"""
                    "{'", "'.join(diff)}" were not found in {path}
                """
                )
        return True

    def __init__(
        self,
        groups_names: Dict[int, str],
        group_names_node_features: Dict[str, List[str]],
        mon_nodes_path: str,
        mon_edges_path: str,
        mon_labels_path: str,
        edge_index_cols: List[str],
        label_index_col: str,
        label_mask_col: Optional[str] = None,
        read_edge_attr: bool = True,
        group_mask_col: Optional[str] = None,
        features_edges_names: Optional[List[str]] = None,
        label_cols: Optional[List[str]] = None,
        target_names: Optional[List[str]] = None,
    ) -> None:
        self._check_cols_in_parquet(group_mask_col, mon_nodes_path)
        self._check_cols_in_parquet(label_cols, mon_labels_path)
        self._check_cols_in_parquet([label_mask_col], mon_labels_path)
        self._check_cols_in_parquet([label_index_col], mon_labels_path)

        for key, val in group_names_node_features.items():
            try:
                self._check_cols_in_parquet(val, mon_nodes_path)
            except Exception as ex:
                raise ValueError(
                    f"""
                    {str(ex)} for group {key} aka {groups_names[key]}
                """
                )

        df_node_feats = pq.read_table(mon_nodes_path).to_pandas()
        df_labels = pq.read_table(mon_labels_path, columns=label_cols).to_pandas()
        df_edge_index = pq.read_table(
            mon_edges_path, columns=edge_index_cols
        ).to_pandas()

        # Nodes
        node_features = torch.FloatTensor(df_node_feats.values)
        group_mask = torch.IntTensor(df_node_feats[group_mask_col].values)
        node_features_names_fixed = df_node_feats.columns.tolist()

        # Labels
        df_labels.set_index(label_index_col, inplace=True)
        df_labels.sort_index(inplace=True)
        df_labels.reset_index(inplace=True)
        targets = {t: torch.LongTensor(df_labels[t].values) for t in target_names}
        label_mask = torch.BoolTensor(df_labels[label_mask_col].values)
        index = torch.LongTensor(df_labels[label_index_col].values)

        try:
            df_node_feats.shape[0] == df_labels.shape[0]
        except Exception as ex:
            raise ValueError(
                f"""
                Length of features must be equal to the length of labels.
                """
            )

        # Edges
        edge_index = torch.LongTensor(df_edge_index.values).T

        # Nodes
        self.node_features = node_features
        self.group_mask = group_mask
        self.targets = targets
        self.label_mask = label_mask
        self.index = index
        self.edge_index = edge_index

        # Edge features
        if read_edge_attr:
            df_edge_feats = pq.read_table(
                mon_edges_path, columns=features_edges_names
            ).to_pandas()

            self.edge_features = torch.FloatTensor(df_edge_feats.values)
            self.edge_features_names = df_edge_feats.columns.tolist()
        else:
            self.edge_features = None
            self.edge_features_names = None

        self.read_edge_attr = read_edge_attr

        # Mappings
        inverse = {v: k for k, v in groups_names.items()}
        self.group_indices_node_findex = {
            inverse[key]: [node_features_names_fixed.index(f) for f in value]
            for key, value in group_names_node_features.items()
        }
        self.groups_names = groups_names

    def sample_data(
        self, num_neighbors: int, batch_size: int, seed: int = 0
    ) -> Dict[str, List[torch.utils.data.DataLoader]]:
        """Samling data.

        Args:
            num_neighbors (int): Number of neighbors are sampled for each node in each iteration.
            batch_size (int): Numbers of samples per batch to load.
            seed (int, optional): Number of seed of samples. Defaults to 0.

        Returns:
            Dict[str, List[torch.utils.data.DataLoader]]: Sampled data.
        """

        return create_loaders(
            self.node_features,
            self.edge_features,
            self.edge_index,
            self.read_edge_attr,
            num_neighbors,
            batch_size,
            self.group_mask,
            self.group_indices_node_findex,
            self.groups_names,
            self.label_mask,
            self.index,
            targets=self.targets,
        )
