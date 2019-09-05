import torch
from torch.autograd import Variable
import torch.nn.functional as F
a = Variable(torch.randn(5,2))

print(a)

print(F.softmax(a, dim=0))