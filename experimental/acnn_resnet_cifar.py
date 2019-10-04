from torch import nn
import torch.nn.functional as F

from benchmarks.models.cifar_resnet_v2 import resnet
from utilities.train_helpers import grouped_conv


class ACNN(nn.Module):

    def __init__(self, n1=18, n2=3, in_channels=3, no_classes=10):
        super(ACNN, self).__init__()
        self.features_net = resnet(n=n1, in_channels=in_channels)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.filters_net = resnet(n=n2, in_channels=in_channels)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, no_classes),
            nn.LogSoftmax()
        )

    # noinspection PyPep8Naming
    def forward(self, X):
        # out1 = F.relu(self.bn1(self.features_net(X)))
        # out2 = F.relu(self.bn2(self.features_net(X)))
        out1 = self.features_net(X)
        out2 = self.features_net(X)

        out = grouped_conv(out1, out2)

        out = self.fc(out)
        return out
