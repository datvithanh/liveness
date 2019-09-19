import os 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import torch
import numpy as np
from src.utils import load_image

class TrainingDataset(Dataset):
    def __init__(self, file_path, domains, color_channel):
        self.root = file_path
        self.table = pd.read_csv(os.path.join(file_path))
        
        X = self.table['image'].tolist()
        Y = self.table['label'].tolist()

        self.X = X
        self.Y = Y
        self.domains = domains
    
    def __getitem__(self, index):
        x = [load_image(tmp) for tmp in self.X[index].split(',')[:8]]
        x = np.array(x).transpose(3,0,1,2)

        if self.Y[index].startswith('G'):
            y = 0
        else: 
            y = 1

        for ind, domain in enumerate(self.domains):
            if self.Y[index].split('_')[2] == domain:
                z = ind
        return x, y, z

    def __len__(self):
        return len(self.Y)


def LoadDataset(split, cuda, data_path, color_channel, n_jobs, train_set, batch_size, dev_set, dev_batch_size, domains):
    if split == 'train':
        shuffle = True
        dataset_file = train_set + '.csv'
    else: 
        shuffle = False
        dataset_file = dev_set + '.csv'
        
    ds = TrainingDataset(os.path.join(data_path, dataset_file), domains, color_channel)

    # TODO: load below config as params
    return  DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=n_jobs, pin_memory=cuda)
