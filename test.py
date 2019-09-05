import torch

loss = torch.nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)

print(input)
print(target)
print(torch.max(input, 1)[1].tolist())
output = loss(input, target)
print(output.tolist())
output.backward()