from cool_graph.data import Dataset
from torch_geometric.data import Data
import torch
import sys
import os.path as osp
from typing import Literal
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset


class OgbnProteins(Dataset):
    r'''
    The ogbn-proteins dataset is an undirected, weighted, and typed (according to species) graph.
    Nodes represent proteins, and edges indicate different types of biologically meaningful associations between proteins, e.g., physical interactions, co-expression or homology.
    All edges come with 8-dimensional features, where each dimension represents the approximate confidence of a single association type and takes values between 0 and 1 (the larger the value is, the more confident we are about the association).
    The proteins come from 8 species.
    The task is to predict the presence of protein functions in a multi-label binary classification setup, where there are 112 kinds of labels to predict in total.
    
    "Open Graph Benchmark: Datasets for Machine Learning on Graphs"
    https://arxiv.org/pdf/2005.00687v7
    
    
    Args:
        root (str): Root directory where the dataset should be saved.
        log (bool): Whether to print logs during processing the dataset (default: True)
        
    Stats
        * - Name
          - #nodes
          - #edges
          - #features
          - #edge_features
          - #classes
        * - ogbn-proteins
          - 132,534
          - 79,122,504
          - 8
          - 8
          - 2 * 112
    
    '''
    
    def __init__(
        self,
        root: str,
        log: bool = True
    ) -> None:        
        super().__init__(root, 'ogbn-proteins', '', log)
                
        if osp.exists(osp.join(root, f'{self.filename}.pt')): 
            return
                
        self.data = self.process(log)
        
        torch.save(self.data, osp.join(root, f'{self.filename}.pt'))
        if log:
            print(f"dataset saved as {osp.join(root, f'{self.filename}.pt')}", file=sys.stderr)
            
    def process(
        self,
        log: bool = True
    ) -> Data:
        if log:        
            print(f'Preprocessing ', file=sys.stderr)
        pyg_dataset = PygNodePropPredDataset(name = "ogbn-proteins")
        dataset = pyg_dataset[0]
        
        x = []
        encoding_dict = {
            3702: 0,
            4932: 1,
            6239: 2,
            7227: 3,
            7955: 4,
            9606: 5,
            10090: 6,
            511145: 7
        }
        for i in range(dataset.node_species.shape[0]):
            x.append(encoding_dict[dataset.node_species[i][0].item()])
        x = torch.tensor(x)
        x = F.one_hot(x, num_classes=8)
        dataset.x = x.type(torch.FloatTensor)
        return dataset