import torch 
import torch.nn as nn
import torch.nn.functional as F 
from tensorboardX import SummaryWriter 

class ConvLayer(nn.Module):
    def __init__(self, in_feat, out_feat, pooling_size):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv3d(in_feat, out_feat, 3, stride=1, padding=1, dilation=1)
        self.bn = nn.BatchNorm3d(out_feat)
        self.max_pool = nn.MaxPool3d(pooling_size, pooling_size, padding=0, dilation=1)

    def forward(self, X):
        return self.max_pool(F.leaky_relu(self.bn(self.conv(X)), 0.1))

class CNN3D(nn.Module):
    def __init__(self):
        super(CNN3D, self).__init__()

        self.conv1 = ConvLayer(3, 128, [1,2,2])
        self.conv2 = ConvLayer(128, 128, [1,2,2])
        self.conv3 = ConvLayer(128, 128, [1,2,2])
        self.conv4 = ConvLayer(128, 128, [2,2,2])
        self.conv5 = ConvLayer(128, 128, [2,2,2])

        self.fc1 = nn.Linear(4096, 1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1024, 2)

    def forward(self, X):
        X = self.conv1(X)

        X = self.conv2(X)

        X = self.conv3(X)

        X = self.conv4(X)

        X = self.conv5(X)

        fc1 = self.fc1(X.view(X.shape[0], -1))
        X = F.relu(self.bn6(fc1))
        X = self.dropout1(X)
        fc2 = self.fc2(X)
        logsoftmax = F.log_softmax(fc2, dim=1)
        return fc1, fc2, logsoftmax
    
    def freeze(self):
        for param in self.conv1.parameters():
            param.requires_grad = False

        for param in self.conv2.parameters():
            param.requires_grad = False
        
        for param in self.conv3.parameters():
            param.requires_grad = False

        for param in self.conv4.parameters():
            param.requires_grad = False






