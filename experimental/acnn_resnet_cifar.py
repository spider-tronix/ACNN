import torch
from torch import nn

from benchmarks.models.cifar_resnet_v2 import resnet
from utilities.train_helpers import grouped_conv


class ACNN(nn.Module):

    def __init__(self, n1=18, n2=3, in_channels=3, no_classes=10):
        super(ACNN, self).__init__()
        self.features_net = resnet(n=n1, in_channels=in_channels)
        self.filters_net = nn.Sequential(
            resnet(n=n2, in_channels=in_channels, channels=[8, 16, 32]),
            nn.Conv2d(32, 64, 3, 1, dilation=2),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        # self.filters_net = nn.Sequential(
        #     nn.Conv2d(3, 16, 3, stride=1),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, 5, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 5, stride=3),
        #     nn.ReLU()
        # )
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

    # noinspection PyPep8Naming
    def forward(self, X):
        # out1 = F.relu(self.bn1(self.features_net(X)))
        # out2 = F.relu(self.bn2(self.features_net(X)))
        out1 = self.features_net(X)
        out2 = self.filters_net(X)

        out = grouped_conv(out1, out2)

        out = self.fc(out)
        return out


if __name__ == "__main__":
    a = torch.rand((1, 3, 32, 32))
    model = ACNN()
    print(model(a))
