import torch
from torch import nn
import torch.nn.functional as F

in1 = torch.arange(8).reshape(1, 2, 2, 2).to(dtype=torch.float)
# in2 = torch.arange(9).reshape(1, 1, 3, 3).to(dtype=torch.float)

conv1 = nn.Conv2d(2, 5, 2, bias=False)
conv2 = nn.Conv2d(2, 5, 2, bias=False)


output1 = conv1(in1)
output2 = conv2(in1)


# conv = nn.Conv2d(5, 5, 1, bias=False)
# conv.weight = nn.Parameter(output1)
# output3 = conv(output2)

output3 = F.conv2d(output1, output2)

output3.backward()