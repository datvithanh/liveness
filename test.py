import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

a = Variable(torch.randn(20,1024))
b = Variable(torch.randn(20,1024))
c = Variable(torch.randn(20,1024))

tensor_list = [a,b,c]
print(torch.cat(tensor_list, dim = 0).shape)

# a = np.array(range(20))
# a = torch.from_nump(a)


# domains = 9
# b = np.random.randint(0, 9, 20)
# c = [0] * 20

# cnt = 0 
# for do in range(domains):
#     for i,v in enumerate(b):
#         if do == v:
#             c[i] = cnt 
#             cnt += 1
# Ym = a[c]

# print(Ym.shape)

# domain_count = [len([tmp2 for tmp2 in b if tmp2 == tmp]) for tmp in range(domains)]

# Q = []
# for i in range(domains):
#     row = []
#     for j in range(domains):
#         if domain_count[i] == 0 or domain_count[j] == 0:
#             continue
#         if i == j:
#             tmp = np.ones((domain_count[i], domain_count[j]))/(domains * domain_count[i] * domain_count[j])
#         else:
#             tmp = -np.ones((domain_count[i], domain_count[j]))/(domains * (domains - 1) * domain_count[i] * domain_count[j])
#         if row == []:
#             row = tmp
#         else: 
#             row = np.hstack((row, tmp))
#     if row != []:
#         if Q == []:
#             Q = row
#         else:
#             Q = np.vstack((Q, row))

# Q = torch.from_numpy(Q)
# K = torch.mm(torch.mm(Ym, Ym.t()), Q)

# print(K.shape)

# print(c)

