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
            for X_batch, y_batch, _ in self.train_set:
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
        for X_batch, y_batch, _ in self.dev_set:
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
                os.mkdir('result/init')
            torch.save(self.model, os.path.join('result/init','model_epoch' + str(self.epoch)))
        
        # log
        self.log.add_scalars('acc', {'dev': sum([tmp1 == tmp2 for tmp1, tmp2 in zip(all_pred, all_true)])/ len(all_pred)}, self.epoch)
        self.log.add_scalars('loss', {'dev': np.mean(all_loss)}, self.epoch)

        self.model.train()

class Finetuner(Solver):
    def __init__(self, data_path, model_path):
        super(Finetuner, self).__init__()
        self.data_path = data_path
        self.model_path = model_path 

        self.max_epoch = 20
        self.batch_size = 100
        self.domains = ['G', 'Ps', 'Pq', 'Vl', 'Vm', 'Mc', 'Mf', 'Mu', 'Ml']
        self.num_domains = len(self.domains)
        self.lmda = 0.5

    def load_data(self):
        self.verbose('Load data from: ' + self.data_path)
        setattr(self, 'train_set', LoadDataset('train', self.data_path, self.batch_size))
        setattr(self, 'dev_set', LoadDataset('dev', self.data_path, self.batch_size))


    def set_model(self, model_path):
        self.model = torch.load(self.model_path)
        self.opt = torch.optim.Adam(self.model.parameters(), lr = 0.0001, betas=[0.9, 0.99], weight_decay=0.00005)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduce='mean')

    def generalization_loss(self, fc, domain_tensor):
        domain_vector = domain_tensor.tolist()

        c = [0] * 20
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

        Q = torch.from_numpy(Q)
        K = torch.mm(torch.mm(Ym, Ym.t()), Q)

        return torch.trace(K)

    def finetune(self):
        self.model.freeze()
        self.epoch = 0

        while self.epoch < self.max_epoch: 
            self.verbose('Finetuning epoch: ' + str(self.epoch))
            all_pred, all_true = [], []
            all_loss, all_cross_entropy_loss, all_generalization_loss = [], [], []
            step = 0
            for X_batch, y_batch, domain_batch in self.train_set:
                self.progress('Finetuning step - ' + str(step) + '/' + str(len(self.train_set)))
                X_batch = X_batch.to(device = self.device,dtype=torch.float32)
                fc1, fc2, input = self.model(X_batch)
                self.opt.zero_grad()
                
                cross_entropy_loss = self.cross_entropy_loss(input, y_batch)
                generalization_loss_fc1 = self.generalization_loss(fc1, domain_batch)
                generalization_loss_fc2 = self.generalization_loss(fc2, domain_batch)

                pred = torch.max(input, 1)[1]

                loss = cross_entropy_loss + self.lmda*(generalization_loss_fc1 + generalization_loss_fc2)
                
                all_pred += pred.tolist()
                all_true += y_batch.tolist()
                all_cross_entropy_loss.append(cross_entropy_loss.tolist())
                all_generalization_loss.append(generalization_loss_fc1.tolist() + generalization_loss_fc2.tolist())
                all_loss.append(loss.tolist())
                
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)

                if math.isnan(grad_norm):
                    print('NaN')
                else:
                    self.opt.step()

                step += 1

            # log
            self.log.add_scalars('acc-finetune', {'train': sum([tmp1 == tmp2 for tmp1, tmp2 in zip(all_pred, all_true)])/ len(all_pred)}, self.epoch)
            self.log.add_scalars('loss-finetune', {'train-all': np.mean(all_loss)}, self.epoch)
            self.log.add_scalars('loss-finetune', {'train-cel': np.mean(all_cross_entropy_loss)}, self.epoch)
            self.log.add_scalars('loss-finetune', {'train-gen': np.mean(all_generalization_loss)}, self.epoch)



            # self.eval()
            self.epoch += 1

    def valid(self):
        self.model.eval()
        # evaluate code here
        all_pred, all_true = [], []
        all_loss, all_cross_entropy_loss, all_generalization_loss = [], [], [] 
        step = 0
        for X_batch, y_batch, domain_batch in self.dev_set:
            self.progress('Valid step - ' + str(step) + '/' + str(len(self.train_set)))
            X_batch = X_batch.to(device = self.device,dtype=torch.float32)
            fc1, fc2, input = self.model(X_batch)
            self.opt.zero_grad()
            
            cross_entropy_loss = self.cross_entropy_loss(input, y_batch)
            generalization_loss_fc1 = self.generalization_loss(fc1, domain_batch)
            generalization_loss_fc2 = self.generalization_loss(fc2, domain_batch)

            pred = torch.max(input, 1)[1]

            loss = cross_entropy_loss + self.lmda*(generalization_loss_fc1 + generalization_loss_fc2)
            
            all_pred += pred.tolist()
            all_true += y_batch.tolist()
            all_cross_entropy_loss.append(cross_entropy_loss.tolist())
            all_generalization_loss.append(generalization_loss_fc1.tolist() + generalization_loss_fc2.tolist())
            all_loss.append(loss.tolist())

            step += 1

        if np.mean(all_loss) < self.best_val:
            self.best_val = np.mean(all_loss)
            if not os.path.exists('result'):
                os.mkdir('result/init')
            torch.save(self.model, os.path.join('result/init','model_epoch' + str(self.epoch)))
        
        # log
        self.log.add_scalars('acc-finetune', {'dev': sum([tmp1 == tmp2 for tmp1, tmp2 in zip(all_pred, all_true)])/ len(all_pred)}, self.epoch)
        self.log.add_scalars('loss-finetune', {'dev-all': np.mean(all_loss)}, self.epoch)
        self.log.add_scalars('loss-finetune', {'dev-cel': np.mean(all_cross_entropy_loss)}, self.epoch)
        self.log.add_scalars('loss-finetune', {'dev-gen': np.mean(all_generalization_loss)}, self.epoch)

        self.model.train()

