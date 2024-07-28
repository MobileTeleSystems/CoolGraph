import sys
import os.path as osp
from scipy.io import loadmat
import torch
import requests
from torch_geometric.data import Data
from cool_graph.data import Dataset
import numpy as np
from typing import Literal


class NonHomophilous(Dataset):
    r"""Penn94 is a friendship network from the
    Facebook 100 networks of university students from 2005, where nodes represent students.
    The node features are major, second major/minor, dorm/house, year, and high school.
    Each node is labeled with the reported gender of the user.
    Nodes without reported gender are labeled -100.
    The task is to predict the gender label.
    
    The genius is a subset of the social network on
    genius.com — a site for crowdsourced annotations of song lyrics.
    Nodes are users, and edges connect users that follow each other on the site.
    About 20% of users in the dataset are marked "gone" on the site, which appears to often include spam users. 
    The task is to predict whether nodes are marked.
    The node features are user usage attributes like the
    Genius assigned expertise score, counts of contributions, and roles held by the user.
    
    `"Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods"
    <https://arxiv.org/pdf/2110.14446>`_ paper.
    

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): Name of the dataset. It can only take the values "Genius" or "Penn94"
        log (bool): Whether to print logs during processing the dataset (default: True)
        
    Stats:
        * - Name
          - #nodes
          - #edges
          - #features
          - #classes
        * - Penn94
          - 42,554
          - 1,362,229
          - 6
          - 2+1
        * - genius
          - 421,961
          - 984,979
          - 12
          - 2
    """
    
    def __init__(
        self,
        root: str,
        name: Literal['Penn94', 'Genius'],
        log: bool = True
    ) -> None:
        urls = {"Penn94": "https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data/facebook100/Penn94.mat", 
                "Genius": "https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data/genius.mat"}
        root = osp.join(root, name.lower())
        
        super().__init__(root, name, urls[name], log)
        
        if osp.exists(osp.join(root, f'{self.filename}.pt')): 
            return
        
        self.download('mat', log)
        
        self.data = self.process(log)
        
        torch.save(self.data, osp.join(root, f'{self.filename}.pt'))
        if log:
            print(f"dataset saved as {osp.join(root, f'{self.filename}.pt')}", file=sys.stderr)
            
    def download(
        self,
        extension: str,
        log: bool = True,
    ) -> None:
        if log:
            print(f'Downloading {self.url}', file=sys.stderr)

        r = requests.get(self.url)
        with open(osp.join(self.raw, f'{self.filename}.{extension}'),'wb') as f:
            f.write(r.content)

    def process(
        self,
        log: bool = True
    ) -> Data:
        if log:
            print(f'Preprocessing ', file=sys.stderr)
            
        raw_data = loadmat(osp.join(self.raw, f'{self.filename}.mat'))

        if log:
            print(f'Processing ', file=sys.stderr)
        
        if self.name == "Penn94":
            metadata = raw_data['local_info'].astype(np.int64)
            edge_index = torch.tensor(np.array(raw_data['A'].nonzero()), dtype=torch.long)
            y = torch.tensor(metadata[:, 1])
            y = torch.where(y == 0, y - 100, y - 1)
            x = np.hstack((np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
            x=torch.tensor(x).float()
            
        elif self.name == "Genius":
            edge_index = torch.tensor(raw_data['edge_index'], dtype=torch.long)
            x = torch.tensor(raw_data['node_feat'], dtype=torch.float)
            y = torch.tensor(raw_data['label'], dtype=torch.long).squeeze()
        
        return Data(x=x, edge_index=edge_index, y=y, num_nodes=x.shape[0])
