import torch 
import torch.nn as nn
import torch.nn.functional as F 
from tensorboardX import SummaryWriter 

class CNN3D(nn.Module):
    def __init__(self):
        super(CNN3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 128, 3, stride=1, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm3d(128)
        self.max_pool1 = nn.MaxPool3d([1,2,2], [1,2,2], padding=0, dilation=1)

        self.conv2 = nn.Conv3d(128, 128, 3, stride=1, padding=1, dilation=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.max_pool2 = nn.MaxPool3d([1,2,2], [1,2,2], padding=0, dilation=1)

        self.conv3 = nn.Conv3d(128, 128, 3, stride=1, padding=1, dilation=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.max_pool3 = nn.MaxPool3d([1,2,2], [1,2,2], padding=0, dilation=1)

        self.conv4 = nn.Conv3d(128, 128, 3, stride=1, padding=1, dilation=1)
        self.bn4 = nn.BatchNorm3d(128)
        self.max_pool4 = nn.MaxPool3d([2,2,2], [2,2,2], padding=0, dilation=1)

        self.conv5 = nn.Conv3d(128, 128, 3, stride=1, padding=1, dilation=1)
        self.bn5 = nn.BatchNorm3d(128)
        self.max_pool5 = nn.MaxPool3d([2,2,2], [2,2,2], padding=0, dilation=1)

        self.linear1 = nn.Linear(4096, 1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.5)

        self.linear2 = nn.Linear(1024, 2)

    def forward(self, X):
        X = self.max_pool1(F.leaky_relu(self.bn1(self.conv1(X)), 0.1))

        X = self.max_pool2(F.leaky_relu(self.bn2(self.conv2(X)), 0.1))

        X = self.max_pool3(F.leaky_relu(self.bn3(self.conv3(X)), 0.1))

        X = self.max_pool4(F.leaky_relu(self.bn4(self.conv4(X)), 0.1))

        X = self.max_pool5(F.leaky_relu(self.bn5(self.conv5(X)), 0.1))

        X = self.linear1(X.view(X.shape[0], -1))
        X = self.bn6(X)
        X = self.dropout1(X)
        X = self.linear2(X)
        return X





