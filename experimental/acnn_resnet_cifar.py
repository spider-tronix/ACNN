import torch
from torch import nn

from benchmarks.models.cifar_resnet_v2 import resnet
from utilities.train_helpers import grouped_conv


class ACNN(nn.Module):

    def __init__(self, n1=18, n2=3, in_channels=3, no_classes=10):
        super(ACNN, self).__init__()
        self.features_net = resnet(n=n1, in_channels=in_channels)

        # Intuition: The output of the filters networks are too big to be filters
        # *Some* are in the order of 1e2 while others are not. Normally, trained
        # filters for a Conv2d layer are in the order of 1e-2. So we divide the
        # filters output by 1e4. This works, but some of the learned weights vanish
        # as they were not very big initially. So, the inclusion of BottleNecking layers
        # along with BatchNorm layers prove to solve this problem.
        # Also, since BottleNecking is performed, the ResNet is made to learn filters
        # of larger amount of channels.
        # self.filters_net = nn.Sequential(
        #     resnet(
        #         n=n2, in_channels=in_channels,
        #         channels=[32, 64, 128]  # Learn more channels
        #     ),
        #     nn.Conv2d(128, 64, 1),  # Bottleneck block
        #     nn.ReLU(),
        #     nn.BatchNorm2d(64),  # Normalize it
        #     nn.Conv2d(64, 32, 1),  # Bottleneck it again
        #     nn.ReLU()
        # )
        self.filters_net = nn.Sequential(
            resnet(n=n2, in_channels=in_channels),
            nn.Conv2d(64, 64, 3, 2),  # Bottleneck block
            nn.ReLU(),
            nn.BatchNorm2d(64),  # Normalize it
        )

        self.bn = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
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

    # noinspection PyPep8Naming
    def forward(self, X):
        out1 = self.features_net(X)
        out2 = self.filters_net(X)

        out = grouped_conv(out1, out2)
        out = self.bn(out.view(out2.shape[0], 64, 6, 6)).view(out2.shape[0], -1)
        out = self.fc(out)
        return out


if __name__ == "__main__":
    a = torch.rand((2, 3, 32, 32))
    model = ACNN()
    print(model(a))
