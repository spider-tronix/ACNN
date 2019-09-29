from torch import nn
# noinspection PyPep8Naming
import torch.nn.functional as F

from models.ACNN_Grouped_ResNet.modded_resnet import BaseResNet
from utilities.train_helpers import grouped_conv


# noinspection PyPep8Naming
class ACNNResNet(BaseResNet):
    def __init__(self):
        super(ACNNResNet, self).__init__()

    def forward(self, X):
        out1 = self.layer1(X)

        identity1 = out1
        out2 = self.layer2(out1)
        out2 += identity1  # adding residual
        F.relu(out2)

        identity2 = self.downsample_2_3(out2)
        out3 = self.layer3(out2)
        out3 += identity2
        F.relu(out3)

        identity3 = self.downsample_3_4(out3)
        out4 = self.layer4(out3)
        out4 += identity3
        out = F.relu(out4)

        # out = self.avgpool(out4)

        return out


# noinspection PyPep8Naming
class ACNN(nn.Module):
    def __init__(self):
        super(ACNN, self).__init__()
        self.net1 = ACNNResNet()

        self.net2 = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, X):
        out1 = self.net1(X)
        out2 = self.net2(X)

        out = grouped_conv(out1, out2)

        return self.fc(out)
