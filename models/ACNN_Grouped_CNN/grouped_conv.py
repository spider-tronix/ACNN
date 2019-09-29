import torch.nn as nn
# noinspection PyPep8Naming
from utilities.train_helpers import grouped_conv


class ACNN(nn.Module):
    """Branches of the Network"""

    def __init__(self,
                 input_channels=1,
                 net1_channels=None,
                 net2_channels=None):
        super(ACNN, self).__init__()

        if net2_channels is None:
            net2_channels = [1, 16, 32, 64]
        if net1_channels is None:
            net1_channels = [1, 16, 32]
        if input_channels != 1:
            net1_channels[0], net2_channels[0] = input_channels, input_channels
        self.net1 = nn.Sequential(
            nn.Conv2d(net1_channels[0], net1_channels[1],
                      kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(net1_channels[1], net1_channels[2],
                      kernel_size=5, stride=2),
            nn.ReLU()
        )

        self.net2 = nn.Sequential(
            nn.Conv2d(net2_channels[0], net2_channels[1],
                      kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(net2_channels[1], net2_channels[2],
                      kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(net2_channels[2], net2_channels[3],
                      kernel_size=5, stride=2),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(5184, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1)
        )

    # noinspection PyPep8Naming
    def forward(self, X):
        """
        Forward Prop
        :param X: Input dataset with batch dimension
        :return: Output of model and parameters
        """
        out1 = self.net1(X)
        out2 = self.net2(X)

        out3 = grouped_conv(out1, out2)

        out3 = self.fc(out3)
        return out3
