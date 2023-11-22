from typing import Callable, Dict, List, Union

import torch


class MultiEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: List[int],
        embedding_dim: Union[List[int], Dict[int, int], Callable[[int], int], None],
        **kwargs
    ) -> None:
        """
        Creating and concating embeddings of categorical features.

        Args:
            num_embeddings (List): Number of unique value per feature
            embedding_dim: Embedding size per feature. If set None -> use default func.
        """
        super(MultiEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self.kwargs = kwargs
        self.default_emb_dim_calc = lambda x: max(2, int(x**0.27)) if x > 2 else 1

        if isinstance(embedding_dim, List):
            self.embedding_dim = embedding_dim

        elif isinstance(embedding_dim, Callable):
            self.embedding_dim = [embedding_dim(num) for num in self.num_embeddings]

        elif embedding_dim is None:
            self.embedding_dim = [
                self.default_emb_dim_calc(num) for num in self.num_embeddings
            ]

        elif isinstance(embedding_dim, Dict):
            self.embedding_dim = [embedding_dim[num] for num in self.num_embeddings]

        self.multi_embedding = torch.nn.ModuleList()
        for num, dim in zip(self.num_embeddings, self.embedding_dim):
            self.multi_embedding.append(torch.nn.Embedding(num, dim, **kwargs))

    def forward(self, x) -> torch.Tensor:
        x = [self.multi_embedding[i](x[:, i]) for i in range(len(self.multi_embedding))]
        return torch.cat(x, dim=1)

    @property
    def in_features(self):
        return len(self.multi_embedding)

    @property
    def out_features(self):
        return sum([emb.embedding_dim for emb in self.multi_embedding])
