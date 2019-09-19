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
        if params.cuda:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
        else: 
            self.device = torch.device('cpu')

        self.params = params
        self.config = config

        self.epoch = params.epoch
        self.max_epoch = config['estimator']['']

        self.log = SummaryWriter(os.path.join('log', self.config['experiment']))

        self.load_data()
        self.set_model()

    def verbose(self,msg):
        print('[INFO]',msg)

    def progress(self, msg):
        print(msg + '                              ', end='\r')

    def load_data(self):
        self.verbose('Load data from: ' + self.config['estimator']['data_path'])
        setattr(self, 'train_set', LoadDataset('train', **self.config['estimator']))
        setattr(self, 'dev_set', LoadDataset('dev', **self.config['estimator']))

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
    
    
    def generalization_loss(self, fc, domain_tensor):
        domain_vector = domain_tensor.tolist()

        c = [0] * len(domain_vector)
        cnt = 0 
        for do in range(self.num_domains):
            for i,v in enumerate(domain_vector):
                if do == v:
                    c[i] = cnt 
                    cnt += 1
        Ym = fc[c]

        domain_count = [len([tmp2 for tmp2 in domain_vector if tmp2 == tmp]) for tmp in range(self.num_domains)]

        Q = []
        for i in range(self.num_domains):
            row = []
            for j in range(self.num_domains):
                if domain_count[i] == 0 or domain_count[j] == 0:
                    continue
                if i == j:
                    tmp = np.ones((domain_count[i], domain_count[j]))/(self.num_domains * domain_count[i] * domain_count[j])
                else:
                    tmp = -np.ones((domain_count[i], domain_count[j]))/(self.num_domains * (self.num_domains - 1) * domain_count[i] * domain_count[j])
                if row == []:
                    row = tmp
                else: 
                    row = np.hstack((row, tmp))
            if row != []:
                if Q == []:
                    Q = row
                else:
                    Q = np.vstack((Q, row))

        Q = torch.from_numpy(Q).to(device = self.device, dtype=torch.float)
        K = torch.mm(torch.mm(Ym, Ym.t()), Q)

        return torch.trace(K)

    def exec(self):

        if self.params.mode == 'finetuning':
            self.model.freeze()

        while self.epoch < self.max_epoch: 
            self.verbose(f'{self.params.mode} epoch: ' + str(self.epoch))
            all_pred, all_true, all_loss = [], [], []

            if self.params.mode == 'finetuning':
                all_cross_entropy_loss, all_generalization_loss = [], []
                fc1_tensor_all, fc2_tensor_all, input_tensor_all, domain_tensor_all, y_tensor_all = [], [], [], [], []

            step = 0
            for X_batch, y_batch, domain_batch in self.train_set:
                self.progress(f'Epoch {self.params.mode} step - {step} / {len(self.train_set)}')

                fc1, fc2, input = self.model(X_batch)
                pred = torch.max(input, 1)[1]

                all_pred += pred.tolist()
                all_true += y_batch.tolist()

                if self.params.mode == 'training':
                    loss = self.cross_entropy_loss(input, y_batch)
                    loss.backward()
                    self.opt.step()

                    del loss
                else:
                    fc1_tensor_all.append(fc1)
                    fc2_tensor_all.append(fc2)
                    domain_tensor_all.append(domain_batch) 
                    input_tensor_all.append(input)
                    y_tensor_all.append(y_batch)

                    #TODO: move number of batch to calculate loss to config
                    if len(fc1_tensor_all) == 50:
                        cross_entropy_loss = self.cross_entropy_loss(torch.cat(input_tensor_all, dim = 0), torch.cat(y_tensor_all, dim = 0))
                        generalization_loss_fc1 = self.generalization_loss(torch.cat(fc1_tensor_all, dim = 0), torch.cat(domain_tensor_all, dim = 0))
                        generalization_loss_fc2 = self.generalization_loss(torch.cat(fc2_tensor_all, dim = 0), torch.cat(domain_tensor_all, dim = 0))

                        self.opt.zero_grad()

                        loss = cross_entropy_loss + self.config['estimator']['lambda']*(generalization_loss_fc1 + generalization_loss_fc2)
                        
                        
                        all_cross_entropy_loss.append(cross_entropy_loss.tolist())
                        all_generalization_loss.append(generalization_loss_fc1.tolist() + generalization_loss_fc2.tolist())
                        all_loss.append(loss.tolist())
                        
                        loss.backward()
                        self.opt.step()

                        fc1_tensor_all, fc2_tensor_all, input_tensor_all, domain_tensor_all, y_tensor_all = [], [], [], [], []
                        del cross_entropy_loss, generalization_loss_fc1, generalization_loss_fc2, loss

                step += 1
            
            self.log.add_scalars('acc', {'train': sum([tmp1 == tmp2 for tmp1, tmp2 in zip(all_pred, all_true)])/ len(all_pred)}, self.epoch)
            self.log.add_scalars('loss', {'train': np.mean(all_loss)}, self.epoch)

            self.log.add_scalars('acc-finetune', {'train': sum([tmp1 == tmp2 for tmp1, tmp2 in zip(all_pred, all_true)])/ len(all_pred)}, self.epoch)
            self.log.add_scalars('loss-finetune', {'train-all': np.mean(all_loss)}, self.epoch)
            self.log.add_scalars('loss-finetune', {'train-cel': np.mean(all_cross_entropy_loss)}, self.epoch)
            self.log.add_scalars('loss-finetune', {'train-gen': np.mean(all_generalization_loss)}, self.epoch)

            self.eval()
            self.epoch += 1
                

