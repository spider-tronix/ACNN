from torch import nn
# noinspection PyPep8Naming
import torch.nn.functional as F

from benchmarks.agents.vanilla_ResNet.resnet10 import ResNet, BasicBlock


class BasicBlock(BasicBlock):

    def __init__(self, *args):
        super().__init__(*args)

    @staticmethod
    def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                         padding=dilation, groups=groups, bias=False, dilation=dilation)


class ResnetUnit(ResNet):
    def __init__(self, architecture=None, num_classes=10):
        super().__init__(architecture, num_classes)
        self.inplanes = 16
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.conv1 = nn.Conv2d(1, 16,
                               kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(16)
        self.layer1 = self._make_layer(16, self.architecture[0])  # Layer 1
        self.layer2 = self._make_layer(32, self.architecture[1], stride=1)  # Layer 2
        self.layer3 = self._make_layer(64, self.architecture[2], stride=2)  # Layer 3
        self.layer4 = self._make_layer(64, self.architecture[3], stride=2)  # Layer 4

    def forward(self, x):
        """
        Forward Prop of Final Network
        :param x: Input image from an iterating dataloader with batch dimension
        :return: Output after propagation
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)

        return x


class ACNNResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = ResnetUnit()
        self.net2 = ResnetUnit()

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, X):
        """
        Forward Prop
        :param X: Input dataset with batch dimension
        :return: Output of model and parameters
        """
        out1 = self.net1(X)
        out2 = self.net2(X)

        batch_size, c1, h1, w1 = out1.shape
        _, c2, h2, w2 = out2.shape

        out2 = out2[:, :, None, :, :]
        out2 = out2.repeat(1, 1, c1, 1, 1)

        out3 = F.conv2d(
            input=out1.view(1, batch_size * c1, h1, w1),
            weight=out2.view(batch_size * c2, c1, h2, w2),
            groups=batch_size
        )

        out3 = out3.reshape(batch_size, -1)
        out3 = self.fc(out3)
        return out3
