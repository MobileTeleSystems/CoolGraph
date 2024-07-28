import sys
import os
import os.path as osp
from zipfile import ZipFile 
import requests
from scipy.io import loadmat
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Literal
from cool_graph.data import Dataset


class AntiFraud(Dataset):
    r"""The YelpChi dataset includes hotel and restaurant reviews filtered (spam) and recommended (legitimate) by Yelp.
    The Amazon dataset includes product reviews under the Musical Instruments category.
    Users with more than 80% helpful votes are labeled as benign entities and users with less
    than 20% helpful votes as fraudulent entities.
    There are three different types of edges in each dataset
    YelpChi:
        1) R-U-R: it connects reviews posted by the same user
        2) R-S-R: it connects reviews under the same product with the same star rating (1-5 stars)
        3) R-T-R: it connects two reviews under the same product posted in the same month.
    Amazon:
        1) U-P-U: it connects users reviewing at least one same product 
        2) U-S-V: it connects users having at least one same star rating within one week
        3) U-V-U: it connects users with top 5% mutual review text similarities (measured by TF-IDF)
    The task is spam review detection.
    Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters
    <https://arxiv.org/pdf/2008.08692>
    
    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): Name of the dataset. It can only take the values "Amazon" or "YelpChi"
        log (bool): Whether to print logs during processing the dataset (default: True)
        
    Stats:
        * - Name
          - #nodes
          - #edges
          - #features
          - #edge_features
          - #classes
        * - Amazon
          - 11,944
          - 8,835,152
          - 25
          - 12
          - 2
        * - YelpChi
          - 45,954
          - 7,693,958
          - 32
          - 12
          - 2
    """
    
    def __init__(
        self,
        root: str,
        name: Literal['YelpChi', 'Amazon'],
        log: bool = True
    ) -> None:
        url = f'https://github.com/finint/antifraud/raw/main/data/{name}.zip'
        root = osp.join(root, name.lower())
        
        cols = {'YelpChi': {'ubl': 'rur', 'rbl': 'rsr', 'sbl': 'rtr', 'rsbl': 'rstr'},
                'Amazon': {'ubl': 'upu', 'rbl': 'usu', 'sbl': 'uvu', 'rsbl': 'usvu'}}
        self.cols = cols[name]

        super().__init__(root, name, url, log)
        
        if osp.exists(osp.join(root, f'{self.filename}.pt')): 
            return
        
        self.download('zip', log)
        
        self.unzip(log)
        
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

    def unzip(
        self,
        log: bool = True
    ) -> None:
        if log:
            print(f'Extracting {osp.join(self.raw, self.filename + ".zip")}', file=sys.stderr)
            
        with ZipFile(osp.join(self.raw, self.filename + ".zip"), 'r') as zObject: 
            zObject.extractall(path=self.raw)
            
    def process(
        self,
        log:bool = True
    ) -> Data:
        if log:
            print(f'Preprocessing ', file=sys.stderr)
            
        matfile = loadmat(osp.join(self.raw, self.name + ".mat"))
        
        if log:
            print(f'Processing ', file=sys.stderr)
            
        ubl = self.cols['ubl']      # User-Based Links
        rbl = self.cols['rbl']      # Rating-Based Links
        sbl = self.cols['sbl']      # Similarity-Based Links
        rsbl = self.cols['rsbl']    # Rating&Simularity-Based Links
        
        net_ubl = matfile['net_' + ubl]
        net_rbl = matfile['net_' + rbl]
        net_sbl = matfile['net_' + sbl]
        matfile_homo = matfile['homo']

        data_file = matfile
        labels = pd.DataFrame(data_file['label'].flatten())[0]
        feat_data = pd.DataFrame(data_file['features'].todense().A)

        adj_ubl = np.vstack(net_ubl.nonzero())
        adj_sbl = np.vstack(net_sbl.nonzero())
        adj_rbl = np.vstack(net_rbl.nonzero())
        adj_homo = np.vstack(matfile_homo.nonzero())

        df_ubl_edges = pd.DataFrame(adj_ubl.T)
        df_sbl_edges = pd.DataFrame(adj_sbl.T)
        df_rbl_edges = pd.DataFrame(adj_rbl.T)
        df_homo_edges = pd.DataFrame(adj_homo.T)

        df_ubl_edges[ubl] = 1
        df_sbl_edges[rbl] = 1
        df_rbl_edges[sbl] = 1

        df_all_edges = df_sbl_edges.merge(
                df_rbl_edges,on=[0,1],how='outer'
            ).merge(df_ubl_edges,on=[0,1],how='outer'
               ).fillna(0)

        df_all_edges = df_all_edges.rename(columns = {0:'index1', 1:'index2'})

        df_all_edges['rsbl'] = df_all_edges[rbl] * df_all_edges[sbl]
        extra_feats = df_all_edges.groupby('index1')[[sbl, rbl, ubl, 'rsbl']].sum().reset_index()\
            .rename(columns={'index1':'index'})

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

        extra_feats[[rbl, ubl, sbl,'rsbl']] = scaler.fit_transform(
            np.log10(extra_feats[[rbl, ubl, sbl,'rsbl']] + 1)
        )

        df_all_edges_extra = df_all_edges\
            .merge(
                    extra_feats.rename(columns={'index':'index1'}), 
                    suffixes=('','_lhs'),
                    on=['index1']
            )\
            .merge(
                    extra_feats.rename(columns={'index':'index2'}), 
                    suffixes=('','_rhs'),
                    on=['index2']
            )

        feat_data['index'] = np.arange(len(feat_data))
        df_feats_all = feat_data.drop(columns=['index'])

        x = torch.FloatTensor(df_feats_all.values)
        edge_index = torch.LongTensor(df_all_edges_extra[['index1','index2']].values.T)

        edge_attr = torch.FloatTensor(df_all_edges_extra.drop(columns=['index1','index2']).values)
        y = torch.LongTensor(labels.sort_index().values)

        return Data(x=x,edge_index=edge_index, edge_attr=edge_attr, y=y)        
