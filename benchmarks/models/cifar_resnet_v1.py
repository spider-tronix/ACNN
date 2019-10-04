import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CifarResNet(nn.Module):
    """
        All in one ResNet model for training on CIFAR dataset.
        Implements ResNet-20, ResNet-32, ResNet-44, ResNet-56, ResNet-110.
    """

    def __init__(self, n):
        """
        Init all variables for class
        """
        super(CifarResNet, self).__init__()

        if n not in [3, 5, 7, 9, 18]:
            raise NotImplementedError('Resnet 20/32/44/56/110 only implemented. \
                                            Contact Sharan or Sachin, if u need any other model')
        self.n = n

        self.l = 1  # keeps count of layers added
        self.relu_list = np.zeros(6 * n + 2)

        self.model = nn.Sequential()
        self.model.add_module(f'conv{self.l}', nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False))
        self.model.add_module(f'bn{self.l}', nn.BatchNorm2d(16))
        self.model.add_module(f'relu{self.l}', nn.ReLU())
        self.relu_list[self.l] = 1
        self.l += 1

        self.grouped_resnet_blocks(in_c=16, out_c=16)
        self.grouped_resnet_blocks(in_c=16, out_c=32, sub=1)
        self.grouped_resnet_blocks(in_c=32, out_c=64, sub=2)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Sequential(
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1)
        )

    def grouped_resnet_blocks(self, in_c, out_c, sub=0):

        for i in range(self.n):
            if i == 0 and in_c != out_c:  # condition for downsample and subsample
                self.model.add_module(f'conv{self.l}', nn.Conv2d(in_c, out_c, 1, stride=2, bias=False))  # downsampling
            else:
                self.model.add_module(f'conv{self.l}', nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False))
            self.model.add_module(f'bn{self.l}', nn.BatchNorm2d(out_c))
            self.model.add_module(f'relu{self.l}', nn.ReLU())
            self.relu_list[self.l] = 1
            self.l += 1
            self.model.add_module(f'conv{self.l}', nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False))
            self.model.add_module(f'bn{self.l}', nn.BatchNorm2d(out_c))
            self.l += 1

        if in_c != out_c:  # condition for subsample
            self.model.add_module(f'subSample{sub}_conv', nn.Conv2d(in_c, out_c, 1, stride=2, bias=False))
            self.model.add_module(f'subSample{sub}_bn', nn.BatchNorm2d(out_c))
            self.model.add_module(f'subSample{sub}_relu', nn.ReLU())

    # noinspection PyPep8Naming
    def forward(self, X):
        """
        forward propagation logic
        """

        layers = {name: module for name, module in self.model.named_modules()}
        id_dict = {f'id{i}': None for i in range(3 * self.n)}  # network will have 3n skip connections

        out = layers['conv1'](X)
        out = layers['bn1'](out)
        out = layers['relu1'](out)

        s = 0  # skip connections
        for i in range(2, 6 * self.n + 2):
            if id_dict[f'id{s}'] is None:
                id_dict[f'id{s}'] = out  # residual
                # print('-' * 40, '>')
            out = layers[f'conv{i}'](out)
            out = layers[f'bn{i}'](out)

            # print(out.shape, '\t\t |')

            if self.relu_list[i] == 1:  # normal ReLU 
                out = layers[f'relu{i}'](out)
            else:  # Skip Connection
                if s % self.n == 0 and s > 0:
                    sub_i = int(s / self.n)  # subSample number
                    id_dict[f'id{s}'] = layers[f'subSample{sub_i}_conv'](id_dict[f'id{s}'])
                    id_dict[f'id{s}'] = layers[f'subSample{sub_i}_bn'](id_dict[f'id{s}'])
                    id_dict[f'id{s}'] = layers[f'subSample{sub_i}_relu'](id_dict[f'id{s}'])

                out += id_dict[f'id{s}']  # residual added
                # print('<', '-' * 40)
                out = F.relu(out)
                s += 1

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


if __name__ == "__main__":
    n = 5  # try 5, 7, 9, 18
    model = CifarResNet(n)
    print(model(torch.rand((1, 3, 32, 32))))
