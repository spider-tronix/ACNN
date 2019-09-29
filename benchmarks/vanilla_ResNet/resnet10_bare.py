import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F


class BaseResNet(nn.Module):

    def __init__(self, device='cuda:0'):
        """
        Init all variables for class
        """
        super(BaseResNet, self).__init__()
        self.device = device

        self.conv1 = nn.Conv2d(1, 64, 3, stride=1)

        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1)
        )

        self.conv6 = nn.Conv2d(64, 128, 3, stride=1)

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1)
        )

        self.conv9 = nn.Conv2d(128, 256, 3, stride=1)

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(22 * 22 * 256, 5120),
            nn.ReLU(),
            nn.Linear(5120, 2560),
            nn.ReLU(),
            nn.Linear(2560, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.LogSoftmax()
        )

    def forward(self, x):
        """
        forward propagation logic
        """

        x = self.conv1(x)
        residual1 = x  # save input as residual

        x = self.block1(x)
        x += residual1  # add residual to output of block 1
        x = F.relu(x)  # perform relu non-linearity
        residual2 = x  # update residual

        x = self.block2(x)
        x += residual2
        x = F.relu(x)

        x = self.conv6(x)
        residual3 = x  # update residual

        x = self.block3(x)
        x += residual3
        x = F.relu(x)

        x = self.conv9(x)
        residual4 = x

        x = self.block4(x)
        x += residual4
        out1 = F.relu(x)

        out1 = out1.view(out1.shape[0], -1)
        return self.classifier(out1)
