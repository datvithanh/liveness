import os 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import torch
import numpy as np
from utils import load_image

class TrainingDataset(Dataset):
    def __init__(self, file_path):
        self.root = file_path
        self.table = pd.read_csv(os.path.join(file_path))
        
        X = self.table['image'].tolist()
        Y = self.table['label'].tolist()

        self.X = X
        self.Y = Y
    
    def __getitem__(self, index):
        x = [load_image(tmp) for tmp in self.X[index].split(',')[:8]]
        x = np.array(x).transpose(3,0,1,2)

        domains = ['HS', 'HW', 'IP', '5s', 'ZTE']
        if self.Y[index].startswith('G'):
            y = 0
        else: 
            y = 1

        for ind, domain in enumerate(domains):
            if self.Y[index].split('_')[2] == domain:
                z = ind
        return x, y, z

    def __len__(self):
        return len(self.Y)


def LoadDataset(split, data_path, batch_size):
    if split == 'train':
        shuffle = True
        ds = TrainingDataset(os.path.join(data_path, split))
    else: 
        shuffle = False
        ds = TrainingDataset(os.path.join(data_path, split))

    # TODO: load below config as params
    return  DataLoader(ds, batch_size=batch_size, shuffle=True,drop_last=False,num_workers=8)
