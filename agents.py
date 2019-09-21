import torch
import torch.nn as nn
from torchviz import make_dot
import torch.nn.functional as F


class Viba(nn.Module):
    """Branches of the Network"""

    def __init__(self,
                 net1_channels=(1, 16, 32),
                 net2_channels=(1, 16, 32, 64),
                 # kernel_size=3, stride=1, padding=2,
                 junction_channels=30):
        """
        Init all variables for class
        """
        super(Viba, self).__init__()
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


        # self.junction = nn.Conv2d(in_channels=net1_channels[-1],
        #                           out_channels=junction_channels,
        #                           kernel_size=28,
        #                           stride=stride,
        #                           padding=padding,
        #                           bias=False)

    def forward(self, X):
        out1 = self.net1(X)
        out2 = self.net2(X)

        out2 = torch.squeeze(out2)[:, None, :, :]
        out2 = out2.repeat(1, 32, 1, 1)

        out3 = F.conv2d(out1, out2)

        out3 = out3.reshape(out3.size(0), -1)

        return out3
