import torch
from torch import nn

wts = torch.arange(24).reshape(3, 2, 2, 2).to(dtype=torch.float)
inps = torch.arange(18).reshape(1, 2, 3, 3).to(dtype=torch.float)

conv = nn.Conv2d(2, 3, 2, bias=False)

with torch.no_grad():
    conv.weight = nn.Parameter(wts)

output = conv(inps)
output.mean().backward()
print(conv.weight.grad)
