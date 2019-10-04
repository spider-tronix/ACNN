import torch
# noinspection PyPep8Naming
import torch.nn.functional as F
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_filters, out_filters, downsample=None, stride=1):
        super(BasicBlock, self).__init__()

        self.single_block = nn.Sequential(
            nn.Conv2d(in_filters, out_filters, 3, padding=1, bias=False, stride=stride),
            nn.BatchNorm2d(out_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_filters, out_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_filters)
        )
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.single_block(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return F.relu(out)


class CifarResNet(nn.Module):

    def __init__(self, layers, in_channels=3):
        super(CifarResNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.filters = 16
        layers[0] -= 1

        self.layer2 = self._make_layer(16, layers[0])
        self.layer3 = self._make_layer(32, layers[1])
        self.layer4 = self._make_layer(64, layers[2])

    def _make_layer(self, filters, no_layers):
        downsample = None
        stride = 1

        if self.filters != filters:
            downsample = nn.Sequential(
                nn.Conv2d(self.filters, filters, 1, stride=2, bias=False),
                nn.BatchNorm2d(filters),
            )
            stride = 2

        layers = [BasicBlock(self.filters, filters, downsample, stride=stride)]  # Sub Sampling with Down Sampling
        self.filters = filters
        for _ in range(1, no_layers):
            layers.append(BasicBlock(self.filters, filters))

        return nn.Sequential(*layers)

    # noinspection PyPep8Naming
    def forward(self, X):
        out = self.layer1(X)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


def resnet(n=5, in_channels=3):
    layers = [n + 1, n, n]
    return CifarResNet(layers=layers, in_channels=in_channels)


model = resnet()
# print(model)
y = model(torch.rand((1, 3, 32, 32)))
