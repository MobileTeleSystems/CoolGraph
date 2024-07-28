import sys
import os
import os.path as osp
from zipfile import ZipFile 
import pandas as pd
import numpy as np
import urllib.request
import requests
import torch
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from torch_geometric.data import Data
from cool_graph.data import Dataset


class S_FFSD(Dataset):
    r'''
    S-FFSD is a simulated & small version of finacial fraud semi-supervised dataset.
    
    Args:
        root (str): Root directory where the dataset should be saved.
        log (bool): Whether to print logs during processing the dataset (default: True)
        
    Stats
        * - Name
          - #nodes
          - #edges
          - #features
          - #classes
        * - S-FFSD
          - 77,881
          - 860,968
          - 126
          - 2 + 1
    '''
    
    def __init__(
        self,
        root: str,
        log: bool = True
    ) -> None:
        
        file_id = "1pODQWJFS7-dwUmnwl6YNFYQ17241j26b"
        url = f'https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t'
        
        super().__init__(root, "S-FFSD", url, log)
        
        if osp.exists(osp.join(root, f'{self.filename}.pt')): 
            return
        
        self.download('csv', log)
            
#         self.unzip(log)
        
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
        df = pd.read_csv(osp.join(self.raw, f'{self.filename}.csv'))
        df.replace(np.nan, 0, inplace=True)
        df = df.reset_index(drop=True)
        out = []
        alls = []
        allt = []
        pair = ["Source", "Target", "Location", "Type"]
        for column in pair:
            src, tgt = [], []
            edge_per_trans = 3
            for c_id, c_df in tqdm(df.groupby(column), desc=column):
                c_df = c_df.sort_values(by="Time")
                df_len = len(c_df)
                sorted_idxs = c_df.index

                src.extend([sorted_idxs[i] for i in range(df_len) for j in range(edge_per_trans) if i + j < df_len])
                tgt.extend([sorted_idxs[i+j] for i in range(df_len) for j in range(edge_per_trans) if i + j < df_len])

        alls.extend(src)
        allt.extend(tgt)
        edge_index = torch.tensor([alls, allt], dtype=torch.long)

        cal_list = ["Source", "Target", "Location", "Type"]
        for col in cal_list:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].apply(str).values)

        feat_data = df.drop("Labels", axis=1)

        labels = df["Labels"].replace(2, -100)
        y = torch.from_numpy(labels.to_numpy()).to(torch.long)

        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='normal')
        feat_data = qt.fit_transform(feat_data.to_numpy())

        x = torch.from_numpy(feat_data).to(torch.float32)

        return Data(x=x, y=y, edge_index=edge_index)