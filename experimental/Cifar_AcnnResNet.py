import sys
from os.path import dirname, abspath
from torch import nn
import torch
# noinspection PyPep8Naming
import torch.nn.functional as F

sys.path.append(dirname(dirname(abspath(__file__))))
from experimental.cifar_resnet import CifarResNet
from utilities.train_helpers import grouped_conv


# noinspection PyPep8Naming
class CifarClippedResNet(CifarResNet):
    """
        Resnet without the final Average pool layer
    """

    def __init__(self, n):
        super(CifarClippedResNet, self).__init__(n)

    def forward(self, X):

    
        layers = {name: module for name, module in self.model.named_modules()}
        id_dict = {f'id{i}': None for i in range(3 * self.n)}  

        out = layers['conv1'](X)
        out = layers['bn1'](out)
        out = layers['relu1'](out)

        s = 0  # skip connections
        for i in range(2, 6 * self.n + 2):
            if id_dict[f'id{s}'] is None:
                id_dict[f'id{s}'] = out  # residual
            out = layers[f'conv{i}'](out)
            out = layers[f'bn{i}'](out)

            if self.relu_list[i] == 1:  # normal ReLU 
                out = layers[f'relu{i}'](out)
            else:                       # Skip Connection
                if s % self.n == 0 and s > 0:
                    sub_i = int(s / self.n)  # subSample number
                    id_dict[f'id{s}'] = layers[f'subSample{sub_i}_conv'](id_dict[f'id{s}'])
                    id_dict[f'id{s}'] = layers[f'subSample{sub_i}_bn'](id_dict[f'id{s}'])
                    id_dict[f'id{s}'] = layers[f'subSample{sub_i}_relu'](id_dict[f'id{s}'])

                out += id_dict[f'id{s}']  # residual added
                out = F.relu(out)
                s += 1

        return out


# noinspection PyPep8Naming
class CifarAcnnResNet(nn.Module):
    """
        ACNN variant of ResNet. Does drastic reduction in filters network.
        Architecture optimized for training on CIFAR-10 dataset.
    """

    def __init__(self, n1, n2):
        super(CifarAcnnResNet, self).__init__()

        self.features_net = CifarClippedResNet(n1)

        # gives 64 (3x3) filters
        self.filters_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=3),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 6 * 6, 1024),
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
