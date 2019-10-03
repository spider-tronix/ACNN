import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F


class BaseResNet(nn.Module):

    def __init__(self):
        """
        Init all variables for class
        """
        super(BaseResNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )

        self.downsample_2_3 = nn.Sequential(
            nn.Conv2d(16, 32, 1, stride=2, bias=False),
            nn.BatchNorm2d(32)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )

        self.downsample_3_4 = nn.Sequential(
            nn.Conv2d(32, 64, 1, stride=2, bias=False),
            nn.BatchNorm2d(64)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Sequential(
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1)
        )

    # noinspection PyPep8Naming
    def forward(self, X):
        """
        forward propagation logic
        """

        out1 = self.layer1(X)

        identity1 = out1
        out2 = self.layer2(out1)
        out2 += identity1  # adding residual
        F.relu(out2, inplace=True)

        identity2 = self.downsample_2_3(out2)
        out3 = self.layer3(out2)
        out3 += identity2
        F.relu(out3, inplace=True)

        identity3 = self.downsample_3_4(out3)
        out4 = self.layer4(out3)
        out4 += identity3
        F.relu(out4, inplace=True)

        out = self.avgpool(out4)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
