from typing import Dict, List, Literal, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn import ELU, GELU, Dropout, LeakyReLU, Linear, PReLU, ReLU, Sequential
from torch_geometric.nn import GraphConv

from cool_graph.models.categorical_embeddings import MultiEmbedding


class GraphConvGNN(torch.nn.Module):
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
        n_hops: int,
        graph_conv_weight_norm_flag: bool,
        conv1_aggrs: Dict[Literal["mean", "max", "add"], Union[int, float]],
        conv1_dropout_rate: float,
        conv2_aggrs: Union[
            Dict[Literal["mean", "max", "add"], Union[int, float]], None
        ],
        conv2_dropout_rate: Optional[float] = None,
        groups_cat_names_dim_embeds: Optional[Dict[str, List[int]]] = None,
        groups_cat_names: Optional[Dict[int, str]] = None,
        target_names: Optional[List[str]] = None,
        target_sizes: Optional[List[int]] = None,
        cat_features_sizes: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        super(GraphConvGNN, self).__init__()

        self.groups_names = groups_names
        self.groups_names_num_features = groups_names_num_features
        self.n_hops = n_hops
        self.lin_prep_size_common = lin_prep_size_common
        self.target_names = target_names or ["y"]
        self.target_sizes = target_sizes or [2] * len(self.target_names)

        self.groups_cat_names_dim_embeds = groups_cat_names_dim_embeds or {}
        self.groups_cat_names = groups_cat_names or {}

        if activation == "relu":
            act = ReLU
        elif activation == "prelu":
            act = PReLU
        elif activation == "leakyrelu":
            act = LeakyReLU
        elif activation == "elu":
            act = ELU
        elif activation == "gelu":
            act = GELU

        if cat_features_sizes is not None:
            self.multi_embed = MultiEmbedding(cat_features_sizes, None)

        for cat_name, num_embeddings in self.groups_cat_names_dim_embeds.items():
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

        input_size = lin_prep_sizes[-1]
        self.conv1 = torch.nn.ModuleDict()
        for aggr, output_size in conv1_aggrs.items():
            if output_size > 0:
                conv = GraphConv(input_size, output_size, aggr)
                if graph_conv_weight_norm_flag:
                    conv.lin_rel = torch.nn.utils.weight_norm(conv.lin_rel)
                    conv.lin_root = torch.nn.utils.weight_norm(conv.lin_root)
                self.conv1[aggr] = conv
        self.conv1_activation = act()
        self.conv1_dropout = Dropout(conv1_dropout_rate)

        input_size = sum(conv1_aggrs.values())
        if conv2_aggrs:
            self.conv2 = torch.nn.ModuleDict()
            for aggr, output_size in conv2_aggrs.items():
                if output_size > 0:
                    conv = GraphConv(input_size, output_size, aggr)
                    if graph_conv_weight_norm_flag:
                        conv.lin_rel = torch.nn.utils.weight_norm(conv.lin_rel)
                        conv.lin_root = torch.nn.utils.weight_norm(conv.lin_root)
                    self.conv2[aggr] = conv
            self.conv2_activation = act()
            self.conv2_dropout = Dropout(conv2_dropout_rate)

        if conv2_aggrs:
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
            x_out.append(conv(x, edge_index))
        x = torch.cat(x_out, dim=1)
        x = self.conv1_dropout(self.conv1_activation(x))

        if self.n_hops == 2:
            x_out = []
            for conv in self.conv2.values():
                x_out.append(conv(x, edge_index))
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
        embedding_size = 112
        emb_names = ["emb_{:>02}".format(i) for i in range(embedding_size)]

        edge_index = data.edge_index
        mask = data.group_mask

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
            x_out.append(conv(x, edge_index))
        x = torch.cat(x_out, dim=1)
        x = self.conv1_dropout(self.conv1_activation(x))

        if self.n_hops == 2:
            x_out = []
            for conv in self.conv2.values():
                x_out.append(conv(x, edge_index))
            x = torch.cat(x_out, dim=1)
            x = self.conv2_dropout(self.conv2_activation(x))

        outs = {name: self.lin_out[name](x) for name in self.target_names}
        scores = {
            name: F.softmax(self.lin_out[name](x), dim=1).detach().cpu().numpy()
            for name in self.target_names
        }
        embs = {"emb": x.detach().cpu().numpy()}

        return outs, scores, embs
