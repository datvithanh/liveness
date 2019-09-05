import torch
import cv2
import numpy as np
from model import CNN3D
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from dataset import LoadDataset
import math
import os

GRAD_CLIP = 5

class Solver():
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    def verbose(self,msg):
        print('[INFO]',msg)

    def progress(self, msg):
        print(msg+'                              ', end='\r')


class Trainer(Solver):
    def __init__(self, data_path):
        # TODO: move this to argument or config
        # self.device = torch.device('cpu')
        super(Trainer, self).__init__()

        self.log = SummaryWriter('log')
        self.data_path = data_path
        self.batch_size = 8

        self.epoch = 0
        self.best_val = 1e6

        #move this to some kind of config
        self.max_epoch = 100
        self.max_fine_tune_epoch = 20
    
    def load_data(self):
        self.verbose('Load data from: ' + self.data_path)
        setattr(self, 'train_set', LoadDataset('train', self.data_path, self.batch_size))
        setattr(self, 'dev_set', LoadDataset('dev', self.data_path, self.batch_size))
    
    def set_model(self):
        self.model = CNN3D().to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr = 0.001, betas=[0.9, 0.99], weight_decay=0.00005)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduce='mean')

    def write_log(self, val_name, val_dict):
        self.log.add_text(val_name, val_dict, self.epoch)
    
    def train(self):
        #train
        while self.epoch < self.max_epoch: 
            self.verbose('Training epoch: ' + str(self.epoch))
            all_pred, all_true = [], []
            all_loss = []
            step = 0
            for X_batch, y_batch in self.train_set:
                self.progress('Training step - ' + str(step) + '/' + str(len(self.train_set)))
                X_batch = X_batch.to(device = self.device,dtype=torch.float32)
                y_batch = y_batch.to(device = self.device)
                _, _, input = self.model(X_batch)
                self.opt.zero_grad()
                loss = self.cross_entropy_loss(input, y_batch)
                pred = torch.max(input, 1)[1]

                all_pred += pred.tolist()
                all_true += y_batch.tolist()
                all_loss.append(loss.tolist())
            
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)

                if math.isnan(grad_norm):
                    print('NaN')
                else:
                    self.opt.step()

                step += 1

            # log
            self.log.add_scalars('acc', {'train': sum([tmp1 == tmp2 for tmp1, tmp2 in zip(all_pred, all_true)])/ len(all_pred)}, self.epoch)
            self.log.add_scalars('loss', {'train': np.mean(all_loss)}, self.epoch)

            print(sum([tmp1 == tmp2 for tmp1, tmp2 in zip(all_pred, all_true)]) / len(all_pred))
            print(np.mean(all_loss))
            self.valid()
            self.epoch += 1

    def valid(self):
        self.model.eval()
        # evaluate code here
        all_pred, all_true = [], []
        all_loss = []
        step = 0
        for X_batch, y_batch in self.dev_set:
            self.progress('Valid step - ' + str(step) + '/' + str(len(self.train_set)))
            X_batch = X_batch.to(device = self.device,dtype=torch.float32)
            y_batch = y_batch.to(device = self.device)
            _, _, input = self.model(X_batch)
            self.opt.zero_grad()
            loss = self.cross_entropy_loss(input, y_batch)

            pred = torch.max(input, 1)[1]

            all_pred += pred.tolist()
            all_true += y_batch.tolist()
            all_loss.append(loss.tolist())

            step += 1

        if np.mean(all_loss) < self.best_val:
            self.best_val = np.mean(all_loss)
            if not os.path.exists('result'):
                os.mkdir('result')
            torch.save(self.model, os.path.join('result','model_epoch' + str(self.epoch)))
        
        # log
        self.log.add_scalars('acc', {'dev': sum([tmp1 == tmp2 for tmp1, tmp2 in zip(all_pred, all_true)])/ len(all_pred)}, self.epoch)
        self.log.add_scalars('loss', {'dev': np.mean(all_loss)}, self.epoch)

        self.model.train()

class Finetuner(Solver):
    def __init__(self):
        super(Finetuner, self).__init__()

    def load_model(self):
        pass

    def finetune(self):
        self.model.freeze()
        self.epoch = 0

        while self.epoch < self.max_finetune_epoch: 
            self.verbose('Finetuning epoch: ' + str(self.epoch))
            all_pred, all_true = [], []
            all_loss = []
            step = 0
            for X_batch, (y_batch, domain_batch) in self.train_set:
                self.progress('Finetuning step - ' + str(step) + '/' + str(len(self.train_set)))
                X_batch = X_batch.to(device = self.device,dtype=torch.float32)
                _, _, input = self.model(X_batch)
                self.opt.zero_grad()
                
                cross_entropy_loss = self.cross_entropy_loss(input, y_batch)
                generalization_loss = 0

                pred = torch.max(input, 1)[1]

                all_pred += pred.tolist()
                all_true += y_batch.tolist()
                all_loss.append(loss.tolist())

                lbd = 0.44

                loss = cross_entropy_loss + lbd*generalization_loss

                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)

                if math.isnan(grad_norm):
                    print('NaN')
                else:
                    self.opt.step()

                step += 1

            # log
            self.log.add_scalars('acc', {'train': sum([tmp1 == tmp2 for tmp1, tmp2 in zip(all_pred, all_true)])/ len(all_pred)}, self.epoch)
            self.log.add_scalars('loss', {'train': np.mean(all_loss)}, self.epoch)

            # self.eval()
            self.epoch += 1

