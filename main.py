import torch
import cv2
import numpy as np
from model import CNN3D
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from dataset import LoadDataset
import math

GRAD_CLIP = 5

# def load_image(path):
#     img_tmp = cv2.imread('1.jpg')
#     return cv2.resize(img_tmp,(int(128),int(128)))

# cnn = CNN3D()
# print(cnn.parameters())
# optimizer = torch.optim.Adam(cnn.parameters(), lr = 0.001, betas=[0.9, 0.99], weight_decay=0.00005)
# tmpten = [load_image('1.jpg'), load_image('2.jpg'), load_image('3.jpg'), load_image('4.jpg'), load_image('1.jpg'), load_image('2.jpg'), load_image('3.jpg'), load_image('4.jpg')]
# tensor = np.array([tmpten, tmpten[::-1]]).transpose(0,4,1,2,3)
# x = torch.from_numpy(tensor)
# x = x.to(device = torch.device('cpu'), dtype=torch.float32)

# for i in range(4):
#     input = cnn(x)

#     print(input)
#     target = np.array([1, 0])
#     target = torch.from_numpy(target)
#     target = target.to(device = torch.device('cpu'), dtype=torch.long)

#     loss_fn = torch.nn.CrossEntropyLoss(reduce='mean')


#     loss = loss_fn(input, target)
#     print(loss)
#     optimizer.zero_grad()

#     # Backward pass: compute gradient of the loss with respect to model
#     # parametersd
#     loss.backward()

#     # Calling the step function on an Optimizer makes an update to its
#     # parameters
#     optimizer.step()

class Trainer():
    def __init__(self, data_path):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.log = SummaryWriter('log')
        self.data_path = data_path
        self.batch_size = 8

        self.epoch = 0
        self.best_val = 1e6

        #move this to some kind of config
        self.max_epoch = 100
        self.max_fine_tune_epoch = 20
    
    def load_data(self):
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
            all_pred, all_true = [], []
            all_loss = []
            for X_batch, y_batch in self.train_set:
                X_batch = X_batch.to(device = self.device,dtype=torch.float32)
                input = self.model(X_batch)

                self.opt.zero_grad()

                loss = self.cross_entropy_loss(input, y_batch)
                all_pred += input
                all_true += y_batch
                all_loss += loss 

                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)

                if math.isnan(grad_norm):
                    print('NaN')
                else:
                    self.opt.step()

            # log
            self.log.add_scalars('acc', {'train': 123}, self.epoch)
            self.log.add_scalars('err', {'train': 123}, self.epoch)

            # self.eval()
            self.epoch += 1

        #fine-tune
        self.epoch = 0
        while self.epoch < self.max_fine_tune_epoch:
            pass

        return None

    def valid(self):
        self.model.eval()
        # evaluate code here

        for X_batch, y_batch in self.dev_set:
            pass

        self.model.train()
        pass

if __name__ == "__main__":
    trainer = Trainer('data')
    trainer.load_data()
    trainer.set_model()
    trainer.train()

