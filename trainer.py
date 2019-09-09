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
        self.log = SummaryWriter('log')

    def verbose(self,msg):
        print('[INFO]',msg)

    def progress(self, msg):
        print(msg+'                              ', end='\r')


class Trainer(Solver):
    def __init__(self, data_path):
        # TODO: move this to argument or config
        # self.device = torch.device('cpu')
        super(Trainer, self).__init__()

        self.data_path = data_path
        self.batch_size = 16

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
    
    def exec(self):
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

        self.best_val = 20.0
        self.max_epoch = 20
        self.batch_size = 30

        self.domains = ['HS', 'HW', 'IP', '5s', 'ZTE']
        self.num_domains = len(self.domains)
        self.lmda = 0.5

    def load_data(self):
        self.verbose('Load data from: ' + self.data_path)
        setattr(self, 'train_set', LoadDataset('train', self.data_path, self.batch_size))
        setattr(self, 'dev_set', LoadDataset('dev', self.data_path, self.batch_size))


    def set_model(self):
        self.model = torch.load(self.model_path)
        self.opt = torch.optim.Adam(self.model.parameters(), lr = 0.0001, betas=[0.9, 0.99], weight_decay=0.00005)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduce='mean')

    def generalization_loss(self, fc, domain_tensor):
        domain_vector = domain_tensor.tolist()

        c = [0] * self.batch_size
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
        self.model.freeze()
        self.epoch = 0

        while self.epoch < self.max_epoch: 
            self.verbose('Finetuning epoch: ' + str(self.epoch))
            all_pred, all_true = [], []
            all_loss, all_cross_entropy_loss, all_generalization_loss = [], [], []
            step = 0

            fc1_tensor_all, fc2_tensor_all, input_tensor_all, domain_tensor_all, y_tensor_all = [], [], [], [], []

            for X_batch, y_batch, domain_batch in self.train_set:
                self.progress('Finetuning step - ' + str(step) + '/' + str(len(self.train_set)))
                X_batch = X_batch.to(device = self.device,dtype=torch.float32)
                y_batch = y_batch.to(device = self.device)
                fc1, fc2, input = self.model(X_batch)
            
                fc1_tensor_all.append(fc1)
                fc2_tensor_all.append(fc2)
                domain_tensor_all.append(domain_batch) 
                input_tensor_all.append(input)
                y_tensor_all.append(y_batch)

                pred = torch.max(input, 1)[1]

                all_pred += pred.tolist()
                all_true += y_batch.tolist()

                if len(fc1_tensor_all) == 50:
                    cross_entropy_loss = self.cross_entropy_loss(torch.cat(input_tensor_all, dim = 0), torch.cat(y_tensor_all, dim = 0))
                    generalization_loss_fc1 = self.generalization_loss(torch.cat(fc1_tensor_all, dim = 0), torch.cat(domain_tensor_all, dim = 0))
                    generalization_loss_fc2 = self.generalization_loss(torch.cat(fc2_tensor_all, dim = 0), torch.cat(domain_tensor_all, dim = 0))

                    self.opt.zero_grad()

                    loss = cross_entropy_loss + self.lmda*(generalization_loss_fc1 + generalization_loss_fc2)
                    
                    
                    all_cross_entropy_loss.append(cross_entropy_loss.tolist())
                    all_generalization_loss.append(generalization_loss_fc1.tolist() + generalization_loss_fc2.tolist())
                    all_loss.append(loss.tolist())
                    
                    loss.backward()

                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)

                    if math.isnan(grad_norm):
                        print('NaN')
                    else:
                        self.opt.step()

                    fc1_tensor_all, fc2_tensor_all, input_tensor_all, domain_tensor_all, y_tensor_all = [], [], [], [], []

                step += 1

            # log
            self.log.add_scalars('acc-finetune', {'train': sum([tmp1 == tmp2 for tmp1, tmp2 in zip(all_pred, all_true)])/ len(all_pred)}, self.epoch)
            self.log.add_scalars('loss-finetune', {'train-all': np.mean(all_loss)}, self.epoch)
            self.log.add_scalars('loss-finetune', {'train-cel': np.mean(all_cross_entropy_loss)}, self.epoch)
            self.log.add_scalars('loss-finetune', {'train-gen': np.mean(all_generalization_loss)}, self.epoch)



            # self.eval()
            self.valid()
            self.epoch += 1

    def valid(self):
        self.model.eval()
        # evaluate code here
        all_pred, all_true = [], []
        all_loss, all_cross_entropy_loss, all_generalization_loss = [], [], []
        fc1_tensor_all, fc2_tensor_all, input_tensor_all, domain_tensor_all, y_tensor_all = [], [], [], [], []
        step = 0

        for X_batch, y_batch, domain_batch in self.dev_set:
            self.progress('Valid step - ' + str(step) + '/' + str(len(self.train_set)))
            X_batch = X_batch.to(device = self.device,dtype=torch.float32)
            y_batch = y_batch.to(device = self.device)
            fc1, fc2, input = self.model(X_batch)
        
            fc1_tensor_all.append(fc1)
            fc2_tensor_all.append(fc2)
            domain_tensor_all.append(domain_batch) 
            input_tensor_all.append(input)
            y_tensor_all.append(y_batch)

            pred = torch.max(input, 1)[1]

            all_pred += pred.tolist()
            all_true += y_batch.tolist()

            if len(fc1_tensor_all) == 50:
                cross_entropy_loss = self.cross_entropy_loss(torch.cat(input_tensor_all, dim = 0), torch.cat(y_tensor_all, dim = 0))
                generalization_loss_fc1 = self.generalization_loss(torch.cat(fc1_tensor_all, dim = 0), torch.cat(domain_tensor_all, dim = 0))
                generalization_loss_fc2 = self.generalization_loss(torch.cat(fc2_tensor_all, dim = 0), torch.cat(domain_tensor_all, dim = 0))

                self.opt.zero_grad()

                loss = cross_entropy_loss + self.lmda*(generalization_loss_fc1 + generalization_loss_fc2)
                
                
                all_cross_entropy_loss.append(cross_entropy_loss.tolist())
                all_generalization_loss.append(generalization_loss_fc1.tolist() + generalization_loss_fc2.tolist())
                all_loss.append(loss.tolist())
                
                fc1_tensor_all, fc2_tensor_all, input_tensor_all, domain_tensor_all, y_tensor_all = [], [], [], [], []

            step += 1

        if np.mean(all_loss) < self.best_val:
            self.best_val = np.mean(all_loss)
            if not os.path.exists('result'):
                os.mkdir('result/final')
            torch.save(self.model, os.path.join('result/final','model_epoch' + str(self.epoch)))
        
        # log
        self.log.add_scalars('acc-finetune', {'dev': sum([tmp1 == tmp2 for tmp1, tmp2 in zip(all_pred, all_true)])/ len(all_pred)}, self.epoch)
        self.log.add_scalars('loss-finetune', {'dev-all': np.mean(all_loss)}, self.epoch)
        self.log.add_scalars('loss-finetune', {'dev-cel': np.mean(all_cross_entropy_loss)}, self.epoch)
        self.log.add_scalars('loss-finetune', {'dev-gen': np.mean(all_generalization_loss)}, self.epoch)

        self.model.train()



class Tester(Solver):
    def __init__(self, data_path, model_path, extract_feature=False):
        super(Tester, self).__init__()
        self.data_path = data_path
        self.model_path = model_path 
        self.extract_feature = extract_feature

        self.batch_size = 10

    def load_data(self):
        self.verbose('Load data from: ' + self.data_path)
        setattr(self, 'test_set', LoadDataset('test', self.data_path, self.batch_size))
        if self.extract_feature:
            setattr(self, 'train_set', LoadDataset('train', self.data_path, self.batch_size))
            setattr(self, 'dev_set', LoadDataset('dev', self.data_path, self.batch_size))

    def set_model(self):
        self.model = torch.load(self.model_path) 
    
    def exec(self):
        self.model.eval()
        # evaluate code here
        all_pred, all_true = [], []
        all_fc1, all_fc2 = [], []

        if self.extract_feature:
            dsets = ['train_set', 'dev_set', 'test_set']
        else:
            dsets = ['test_set']
        
        for dataset in dsets:
            step = 0
            for X_batch, y_batch, _ in getattr(self, dataset):
                self.progress(f'Test {dataset} step - {str(step)}/{str(len(getattr(self,dataset)))}')

                X_batch = X_batch.to(device = self.device,dtype=torch.float32)
                y_batch = y_batch.to(device = self.device)
                fc1, fc2, input = self.model(X_batch)
                
                pred = torch.max(input, 1)[1]
                
                all_fc1 += fc1.tolist()
                all_fc2 += fc2.tolist()
                all_pred += pred.tolist()
                all_true += y_batch.tolist()

                step += 1
        
        # print(sum([tmp1 == tmp2 for tmp1, tmp2 in zip(all_pred, all_true)])/len(all_pred))
        if self.extract_feature:
            npar = np.array([all_fc1, all_fc2, all_true])
            np.save('result/extract.npy', npar)

        else:
            npar = np.array([all_pred, all_true])
            np.save('result/test.npy', npar)
