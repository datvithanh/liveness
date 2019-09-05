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

        X = self.fc1(X.view(X.shape[0], -1))
        X = F.relu(self.bn6(X))
        X = self.dropout1(X)
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)





