import sys
from os.path import dirname, abspath
from torch import nn
import torch
# noinspection PyPep8Naming
import torch.nn.functional as F

sys.path.append(dirname(dirname(abspath(__file__))))
from benchmarks.cifar_resnet import CifarResNet
from utilities.train_helpers import grouped_conv

# noinspection PyPep8Naming
class CifarClippedResNet(CifarResNet):
    """
        Resnet without the final Average pool layer
    """

    def __init__(self, n):
        super(CifarAcnnResNet, self).__init__(n)

    def forward(self, X):
        
        layers = {name:module for name, module in self.model.named_modules()}
        id_dict = {f'identity{i}': None for i in range(3*self.n)}    # network will have 3n skip connections

        out = layers['conv1'](X)
        out = layers['bn1'](out)
        out = layers['relu1'](out)

        k = 0 
        for i in range(2, 6 * self.n + 2):
            id_dict[f'identity{k}'] = out   # residual
            out = layers[f'conv{i}'](out)
            out = layers[f'bn{i}'](out)
            if self.relu_list[i] == 1: 
                out = layers[f'relu{i}'](out)
            if self.relu_list[i] == 0:
                out += id_dict[f'identity{k}'] # residual added
                out = F.relu(out)
                k += 1

        return out


# noinspection PyPep8Naming
class CifarAcnnResNet(nn.Module):
    """
        ACNN variant of ResNet.
        Architecture optimized for training on CIFAR-10 dataset.
    """
    def __init__(self, n1):
        super(CifarAcnnResNet, self).__init__()
        
        self.features_net = CifarClippedResNet(n1)
        
        # gives 64 (3x3) filters
        self.net2 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=3),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64*6*6, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, X):
        out1 = self.features_net(X)
        out2 = self.filters_net(X)

        out = grouped_conv(out1, out2)

        return self.fc(out)


if __name__ == "__main__":
    a = torch.rand((1, 3, 32, 32))
    model = CifarAcnnResNet(5)
    print(model(a))
