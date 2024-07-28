import os.path as osp
import torch
import sys
import os
import requests


class Dataset:
    def __init__(
        self,
        root: str,
        name: str,
        url: str,
        log: bool = True,
    ) -> None:
        self.root = root
        self.raw = osp.join(root, "raw")
        self.url = url
        self.name = name
        self.filename = name + '_data'
        
        if osp.exists(osp.join(root, f'{self.filename}.pt')): 
            savepath = osp.join(root, f'{self.filename}.pt')
            if log:
                print(f'Using existing file {savepath}', file=sys.stderr)
            self.data = torch.load(savepath)
            return
        
        if not osp.exists(self.raw):
            os.makedirs(self.raw)
