from typing import Dict, List, Literal, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn import (
    ELU,
    GELU,
    Dropout,
    Identity,
    LeakyReLU,
    Linear,
    PReLU,
    ReLU,
    Sequential,
    Sigmoid,
    Tanh,
)
from torch_geometric.nn import NNConv

from cool_graph.models.categorical_embeddings import MultiEmbedding


class NNConvGNN(torch.nn.Module):
    def __init__(
        self,
        activation: Literal["elu", "relu", "prelu", "leakyrelu", "gelu"],
        groups_names_num_features: Dict[str, int],
        groups_names_num_cat_features: Dict[str, int],
        groups_names: Dict[int, str],
        lin_prep_size_common: int,
        lin_prep_len: int,
        lin_prep_sizes: List[int],
        lin_prep_dropout_rate: float,
        lin_prep_weight_norm_flag: bool,
        num_edge_features: int,
        edge_attr_repr_len: int,
        edge_attr_repr_sizes: List[int],
        edge_attr_repr_dropout_rate: float,
        edge_attr_repr_last_dropout_rate: float,
        edge_attr_repr_weight_norm_flag: bool,
        edge_attr_repr_last_activation: Literal[
            "elu", "relu", "prelu", "leakyrelu", "gelu", "identity", "sigmoid"
        ],
        n_hops: int,
        conv1_aggrs: Dict[Literal["mean", "max", "add"], Union[int, float]],
        conv1_dropout_rate: float,
        conv2_aggrs: Optional[
            Dict[Literal["mean", "max", "add"], Union[int, float]]
        ] = None,
        conv2_dropout_rate: Optional[float] = None,
        groups_cat_names_dim_embeds: Optional[Dict[str, List[int]]] = None,
        groups_cat_names: Optional[Dict[int, str]] = None,
        target_names: Optional[List[str]] = None,
        target_sizes: Optional[List[int]] = None,
        cat_features_sizes: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        super(NNConvGNN, self).__init__()

        self.groups_names = groups_names
        self.n_hops = n_hops
        self.lin_prep_size_common = lin_prep_size_common
        self.target_names = target_names or ["y"]
        self.target_sizes = target_sizes or [2] * len(self.target_names)

        self.groups_cat_names_num_embeds = groups_cat_names_dim_embeds or {}
        self.groups_cat_names = groups_cat_names or {}

        if activation == "relu":  # 1st place
            act = ReLU
        elif activation == "prelu":  # 2nd place
            act = PReLU
        elif activation == "leakyrelu":
            act = LeakyReLU
        elif activation == "elu":
            act = ELU
        elif activation == "gelu":
            act = GELU

        if edge_attr_repr_last_activation == "relu":
            act_edge_last = ReLU
        elif edge_attr_repr_last_activation == "prelu":
            act_edge_last = PReLU
        elif edge_attr_repr_last_activation == "leakyrelu":
            act_edge_last = LeakyReLU
        elif edge_attr_repr_last_activation == "tanh":  # 2nd place
            act_edge_last = Tanh
        elif edge_attr_repr_last_activation == "elu":
            act_edge_last = ELU
        elif edge_attr_repr_last_activation == "identity":
            act_edge_last = Identity
        elif edge_attr_repr_last_activation == "sigmoid":  # 1st place
            act_edge_last = Sigmoid
        elif activation == "gelu":
            act_edge_last = GELU

        if cat_features_sizes is not None:
            self.multi_embed = MultiEmbedding(cat_features_sizes, None)

        for cat_name, num_embeddings in self.groups_cat_names_num_embeds.items():
            setattr(self, f"emb_{cat_name}", MultiEmbedding(num_embeddings))

        for group in groups_names:
            name = groups_names[group]
            num = groups_names_num_features[name]
            if group in self.groups_cat_names:
                num += getattr(self, f"emb_{self.groups_cat_names[group]}").out_features

            linear = Linear(num, lin_prep_size_common)
            if lin_prep_weight_norm_flag:
                linear = torch.nn.utils.weight_norm(linear)
            setattr(self, f"lin_prep_{name}", Linear(num, lin_prep_size_common))

        self.lin_prep_tube = Sequential()
        self.lin_prep_tube.add_module("act0", act())
        self.lin_prep_tube.add_module("dropout0", Dropout(lin_prep_dropout_rate))
        lin_prep_sizes = [lin_prep_size_common] + lin_prep_sizes

        lin_prep_sizes[0] += self.multi_embed.out_features if cat_features_sizes is not None else 0

        for i in range(lin_prep_len):
            lin = Linear(lin_prep_sizes[i], lin_prep_sizes[i + 1])
            if lin_prep_weight_norm_flag:
                lin = torch.nn.utils.weight_norm(lin)
            self.lin_prep_tube.add_module(f"lin_prep{i+1}", lin)
            self.lin_prep_tube.add_module(f"act{i+1}", act())
            self.lin_prep_tube.add_module(
                f"dropout{i+1}", Dropout(lin_prep_dropout_rate)
            )

        self.nn_edge_attr_repr = Sequential()
        edge_attr_repr_sizes = [num_edge_features] + edge_attr_repr_sizes

        for i in range(edge_attr_repr_len):
            lin = Linear(edge_attr_repr_sizes[i], edge_attr_repr_sizes[i + 1])
            if edge_attr_repr_weight_norm_flag:
                lin = torch.nn.utils.weight_norm(lin)
            self.nn_edge_attr_repr.add_module(f"edge_attr_repr{i+1}", lin)
            if i != edge_attr_repr_len - 1:
                self.nn_edge_attr_repr.add_module(f"act{i+1}", act())
                self.nn_edge_attr_repr.add_module(
                    f"dropout{i+1}", Dropout(edge_attr_repr_dropout_rate)
                )
            else:
                self.nn_edge_attr_repr.add_module(f"act{i+1}", act_edge_last())
                self.nn_edge_attr_repr.add_module(
                    f"dropout{i+1}", Dropout(edge_attr_repr_last_dropout_rate)
                )

        input_size = lin_prep_sizes[-1]
        self.conv1 = torch.nn.ModuleDict()
        for aggr, output_size in conv1_aggrs.items():
            if output_size > 0:
                self.conv1[aggr] = NNConv(
                    input_size,
                    output_size,
                    nn=Sequential(
                        self.nn_edge_attr_repr,
                        Linear(edge_attr_repr_sizes[-1], input_size * output_size),
                    ),
                    aggr=aggr,
                )
        self.conv1_activation = act()
        self.conv1_dropout = Dropout(conv1_dropout_rate)

        input_size = sum(conv1_aggrs.values())
        if n_hops == 2:
            self.conv2 = torch.nn.ModuleDict()
            for aggr, output_size in conv2_aggrs.items():
                if output_size > 0:
                    self.conv2[aggr] = NNConv(
                        input_size,
                        output_size,
                        nn=Sequential(
                            self.nn_edge_attr_repr,
                            Linear(edge_attr_repr_sizes[-1], input_size * output_size),
                        ),
                        aggr=aggr,
                    )
            self.conv2_activation = act()
            self.conv2_dropout = Dropout(conv2_dropout_rate)

        if n_hops == 2:
            input_size = sum(conv2_aggrs.values())
        else:
            input_size = sum(conv1_aggrs.values())

        self.lin_out = torch.nn.ModuleDict()
        for name, size in zip(self.target_names, self.target_sizes):
            self.lin_out[name] = Linear(input_size, size)

    def forward(self, data: torch.utils.data.DataLoader) -> Dict[str, torch.Tensor]:
        tensors = {v: getattr(data, v) for v in self.groups_names.values()}
        tensors_cat = {v: getattr(data, v) for v in self.groups_cat_names.values()}
        edge_index = data.edge_index
        mask = data.group_mask
        edge_attr = data.edge_attr

        x = torch.zeros(
            len(mask),
            self.lin_prep_size_common,
            device=list(tensors.values())[0].device,
        )

        for group in self.groups_names:
            name = self.groups_names[group]
            tensor = tensors[name]
            if group in self.groups_cat_names:
                cat_name = self.groups_cat_names[group]
                tensor_cat = getattr(self, f"emb_{cat_name}")(tensors_cat[cat_name])
                tensor = torch.cat([tensor, tensor_cat], dim=1)

            branch = getattr(self, f"lin_prep_{name}")
            x[mask == group] = branch(tensor)

        if hasattr(data, 'x_cat'):
            if type(data.x_cat) != torch.Tensor:
                print(type(data.x_cat))
                data.x_cat = torch.tensor(data.x_cat)
            data.x_cat = data.x_cat.long()
            x_cat = self.multi_embed(data.x_cat)
            x = torch.cat([x, x_cat], dim=1)

        x = self.lin_prep_tube(x)

        x_out = []
        for conv in self.conv1.values():
            x_out.append(conv(x, edge_index, edge_attr))
        x = torch.cat(x_out, dim=1)
        x = self.conv1_dropout(self.conv1_activation(x))

        if self.n_hops == 2:
            x_out = []
            for conv in self.conv2.values():
                x_out.append(conv(x, edge_index, edge_attr))
            x = torch.cat(x_out, dim=1)
            x = self.conv2_dropout(self.conv2_activation(x))

        outs = {name: self.lin_out[name](x) for name in self.target_names}
        scores = {
            name: F.softmax(self.lin_out[name](x), dim=1).detach().cpu().numpy()
            for name in self.target_names
        }

        return outs, scores

    def forward_with_embs(
        self, data: torch.utils.data.DataLoader
    ) -> Dict[str, torch.Tensor]:
        tensors = {v: getattr(data, v) for v in self.groups_names.values()}
        tensors_cat = {v: getattr(data, v) for v in self.groups_cat_names.values()}
        edge_index = data.edge_index
        mask = data.group_mask
        edge_attr = data.edge_attr

        x = torch.zeros(
            len(mask),
            self.lin_prep_size_common,
            device=list(tensors.values())[0].device,
        )

        for group in self.groups_names:
            name = self.groups_names[group]
            tensor = tensors[name]
            if group in self.groups_cat_names:
                cat_name = self.groups_cat_names[group]
                tensor_cat = getattr(self, f"emb_{cat_name}")(tensors_cat[cat_name])
                tensor = torch.cat([tensor, tensor_cat], dim=1)

            branch = getattr(self, f"lin_prep_{name}")
            x[mask == group] = branch(tensor)

        x = self.lin_prep_tube(x)

        x_out = []
        for conv in self.conv1.values():
            x_out.append(conv(x, edge_index, edge_attr))
        x = torch.cat(x_out, dim=1)
        x = self.conv1_dropout(self.conv1_activation(x))

        if self.n_hops == 2:
            x_out = []
            for conv in self.conv2.values():
                x_out.append(conv(x, edge_index, edge_attr))
            x = torch.cat(x_out, dim=1)
            x = self.conv2_dropout(self.conv2_activation(x))

        outs = {name: self.lin_out[name](x) for name in self.target_names}
        scores = {
            name: F.softmax(self.lin_out[name](x), dim=1).detach().cpu().numpy()
            for name in self.target_names
        }
        embs = {"emb": x.detach().cpu().numpy()}

        return outs, scores, embs
