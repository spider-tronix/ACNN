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

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
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
            nn.ReLU(inplace=True),
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
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # TODO: Add Linear Layer dependant upon output

        # self.downsample_4_5 = nn.Sequential(
        #     nn.Conv2d(64, 64, 1, stride=2, bias=False),
        #     nn.BatchNorm2d(64)
        # )
        #
        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        # )

        # self.classifier = nn.Sequential(
        #     nn.Linear(22 * 22 * 256, 5120),
        #     nn.ReLU(),
        #     nn.Linear(5120, 2560),
        #     nn.ReLU(),
        #     nn.Linear(2560, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 10),
        #     nn.LogSoftmax()
        # )

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
