import sys
import os
import os.path as osp
from zipfile import ZipFile 
import pandas as pd
import numpy as np
import urllib.request
import requests
import torch
from torch_geometric.data import Data, HeteroData
from typing import Literal
from cool_graph.data import Dataset


class Multitarget(Dataset):
    
    def __init__(
        self,
        root: str,
        name: Literal['10k','50k'],
        log: bool = True
    ) -> None:
        files = {
                '10k':'1KmxlVj7BhGmvScgT941KKfaX2vT1eBsA',
                '50k':'1OqXt5I-zUgDJuQx36VvxH2dlAGh1331M'}    
        url = f'https://drive.usercontent.google.com/download?id={files[name]}&export=download&confirm=t'
        dataname = name
        root = osp.join(root, name.lower())
        super().__init__(root, dataname, url, log)
            
            
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

        # Reading
        df_edges = pd.read_parquet(osp.join(self.raw, 'df_edges.parquet'))
        df_nodes = pd.read_parquet(osp.join(self.raw, 'df_nodes.parquet'))

        # graph is heterogeneous
        # This field determines which group this node belongs to
        node_group_col = 'node_feature_1'

        del df_nodes['index']

        ### Let's define features for each node group
        a = df_nodes.isnull()#.sum(axis=0)

        b = a[df_nodes[node_group_col]==0].sum(axis=0)
        features_group0 = b[b==0].index.tolist()
        features_group0.remove(node_group_col)

        b = a[df_nodes[node_group_col]==1].sum(axis=0)
        features_group1 = b[b==0].index.tolist()
        features_group1.remove(node_group_col)

        del a

        # Finding indices of those features
        cols = df_nodes.columns.tolist()
        features_idx_group0 = [cols.index(c) for c in features_group0]
        features_idx_group1 = [cols.index(c) for c in features_group1]

        x = torch.FloatTensor(df_nodes.values)
        edge_index = torch.LongTensor(df_edges[['index1','index2']].values).T

        del df_edges['index1']
        del df_edges['index2']

        edge_attr = torch.FloatTensor(df_edges.values)

        from torch_geometric.data import Data

        # Dealing with target
        df_train = pd.read_csv(osp.join(self.raw, 'labels.csv'))

        df_y = df_train.pivot('index','label_type','y')
        df_y_ = pd.DataFrame(np.arange(len(x)) , columns=['index'])
        df_y = df_y_.merge(df_y.reset_index(), on='index',how='left').set_index('index')
        df_y['has_target'] = df_y.notnull().sum(axis=1) > 0
        df_y = df_y.fillna(-100)

        node_type = torch.ShortTensor(df_nodes[node_group_col].values)

        # make Data

        # 4 targets
        label_3 = torch.LongTensor(df_y.label_3.values)
        label_4 = torch.LongTensor(df_y.label_4.values)
        label_5 = torch.LongTensor(df_y.label_5.values)
        label_6 = torch.LongTensor(df_y.label_6.values)

        # 
        has_target = torch.BoolTensor(df_y.has_target.values)

        # Make hetero data
        nodes0 = torch.nonzero(node_type == 0)[:,0]
        nodes1 = torch.nonzero(node_type == 1)[:,0]

        e_id_01 = torch.isin( edge_index[0,:], nodes0) & torch.isin( edge_index[1,:], nodes1) 
        e_id_10 = torch.isin( edge_index[0,:], nodes1) & torch.isin( edge_index[1,:], nodes0) 
        e_id_00 = torch.isin( edge_index[0,:], nodes0) & torch.isin( edge_index[1,:], nodes0) 
        e_id_11 = torch.isin( edge_index[0,:], nodes1) & torch.isin( edge_index[1,:], nodes1) 

        e_id_01 = e_id_01.nonzero()[:,0]
        e_id_10 = e_id_10.nonzero()[:,0]
        e_id_00 = e_id_00.nonzero()[:,0]
        e_id_11 = e_id_11.nonzero()[:,0]

        nodes0 = nodes0.sort().values
        nodes1 = nodes1.sort().values    

        map0 = {c.item():i for i, c in enumerate(nodes0)}
        map1 = {c.item():i for i, c in enumerate(nodes1)}
        map_total = {**map0,**map1}

        df_e = pd.DataFrame(edge_index.T.numpy(),columns=['from','to'])

        edge_index.max(dim=1).values

        df_e['from'] = df_e['from'].map(map_total)
        df_e['to'] = df_e['to'].map(map_total)

        edge_index_mapped = torch.LongTensor(df_e.values.T)

        t = {
            'node_0': {
                        'x':x[nodes0][:,features_idx_group0],
                        'label_3':label_3[nodes0],
                        'label_4':label_4[nodes0],
                        'label_5':label_5[nodes0],
                        'label_6':label_6[nodes0],
                        'y':torch.vstack([label_3[nodes0],label_4[nodes0],label_5[nodes0],label_6[nodes0]]).T,
                        'label_mask':has_target[nodes0],
                        'index':torch.arange(len(x[nodes0])),
                     },
            'node_1': {
                        'x':x[nodes1][:,features_idx_group1],
                        'label_3':label_3[nodes1],
                        'label_4':label_4[nodes1],
                        'label_5':label_5[nodes1],
                        'label_6':label_6[nodes1],
                        'y':torch.vstack([label_3[nodes1],label_4[nodes1],label_5[nodes1],label_6[nodes1]]).T,
                        'label_mask':has_target[nodes1],
                        'index':torch.arange(len(x[nodes1])),        
            },
            # dont use e_id_00
            ('node_0','to','node_1') : {
                                            'edge_index':edge_index_mapped[:,e_id_01],
                                            'edge_attr':edge_attr[e_id_01],
                                       },
            ('node_1','to','node_0') : {
                                            'edge_index':edge_index_mapped[:,e_id_10],
                                            'edge_attr':edge_attr[e_id_10],
                                       },
            ('node_1','to','node_1') : {
                                            'edge_index':edge_index_mapped[:,e_id_11],
                                            'edge_attr':edge_attr[e_id_11],
                                       },

        }

        return HeteroData(t)
