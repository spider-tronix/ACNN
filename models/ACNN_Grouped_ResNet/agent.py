from torch import nn
# noinspection PyPep8Naming
import torch.nn.functional as F

from benchmarks.agents.vanilla_ResNet.resnet10 import ResNet


class ResnetUnit(ResNet):
    def __init__(self, architecture=None, num_classes=10):
        super().__init__(architecture, num_classes)

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

        x = self.avgpool(x)


class ACNNResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = ResnetUnit()
        self.net2 = ResnetUnit()

        self.fc = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
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

