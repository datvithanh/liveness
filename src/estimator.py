import torch
import cv2
import numpy as np
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import math
from utils import softmax
import gc
import os
from src.model import CNN3D
from src.dataset import LoadDataset

class Estimator():
    def __init__(self, params, config):
        if params.gpu:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
        else: 
            self.device = torch.device('cpu')

        self.params = params
        self.config = config

        self.step = 0

        self.log = SummaryWriter(os.path.join('log', self.config['experiment']))

        self.set_model()

    def verbose(self,msg):
        print('[INFO]',msg)

    def progress(self, msg):
        print(msg + '                              ', end='\r')

    def load_data(self):
        self.verbose('Load data from: ' + self.config['estimator']['data_path'])
        setattr(self, 'train_set', LoadDataset('train', self.data_path, self.batch_size))
        setattr(self, 'dev_set', LoadDataset('dev', self.data_path, self.batch_size))

    def set_model(self):
        if self.params.model_path:
            if self.device.type == 'cuda':
                self.model = torch.load(self.params.model_path)
            else: 
                self.model = torch.load(self.params.model_path, map_location='cpu')
        else:    
            self.model = CNN3D().to(self.device)

        if self.params.mode == 'finetuning':
            self.model.freeze()
        
        self.opt = torch.optim.Adam(self.model.parameters(), lr = self.config['optimizer']['learning_rate'], betas=self.config['optimizer']['learning_rate'], weight_decay=self.config['optimizer']['weight_decay'])

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduce='mean')

    

    def exec(self):
        if self.params.mode == 'training':
            self.training()

        if self.params.mode == 'finetuning':
            self.finetuning()

        if self.params.mode == 'testing':
            self.testing()
    

    def training(self):
        pass 

    def finetuning(self):
        pass

    def testing(self):
        pass