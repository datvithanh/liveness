import os
import cv2 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import torch
import numpy as np

class TrainingDataset(Dataset):
    def __init__(self, file_path, dataset, bucket_size):
        self.root = file_path
        self.table = pd.read_csv(os.path.join(file_path, dataset + '.csv'))
        
        X = self.table['image'].tolist()
        Y = self.table['label'].tolist()

        # self.X = []
        # self.Y = []
        
        # for tmp in range(len(X)//bucket_size + 1):
        #     if len(X[tmp*bucket_size:(tmp+1)*bucket_size]) > 0:
        #         self.X.append(X[tmp*bucket_size:(tmp+1)*bucket_size])
        #         self.Y.append(Y[tmp*bucket_size:(tmp+1)*bucket_size])

        self.X = X
        self.Y = Y
    
    def load_image(self, path):
        tmp = cv2.imread(path)
        return cv2.resize(tmp, (int(128), int(128)))
    
    def __getitem__(self, index):
        # x = [[self.load_image(p) for p in tmp.split(',')[:8]] for tmp in self.X[index]]
        x = [self.load_image(tmp) for tmp in self.X[index].split(',')[:8]]
        x = np.array(x).transpose(3,0,1,2)
        if self.Y[index].startswith('G'):
            y = 0
        else: 
            y = 1
        return x, y 

    def __len__(self):
        return len(self.Y)


def LoadDataset(split, data_path, batch_size):
    if split == 'train':
        shuffle = True
        ds = TrainingDataset(data_path, split, batch_size)
    else: 
        shuffle = False
        ds = TrainingDataset(data_path, split, batch_size)

    # TODO: load below config as params
    return  DataLoader(ds, batch_size=8, shuffle=shuffle,drop_last=False,num_workers=8)