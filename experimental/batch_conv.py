import torch
import torch.nn as nn
import torch.nn.functional as F
from models.connect_net import ConnectNet


class ACNN(nn.Module):
    """Branches of the Network"""

    def __init__(self,
                 net1_channels=(1, 16, 32),
                 net2_channels=(1, 16, 32, 64),
                 cn_kernel_size=(3, 3), cn_stride=1,
                 device='cuda:0'):
        super(ACNN, self).__init__()
        self.device = device

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

        self.connect_net = ConnectNet(cn_kernel_size,
                                      strides=cn_stride,
                                      device=self.device)

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

    # noinspection PyPep8Naming
    def forward(self, X, return_ff=False):
        """
        Forward Prop
        :param return_ff: Flag for layer visualizations
        :param X: Input dataset with batch dimension
        :return: Output of model and parameters
        """
        params = {}
        out1 = self.net1(X)
        out2 = self.net2(X)
        if return_ff:
            params['features'] = out1
            params['filters'] = out2

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
