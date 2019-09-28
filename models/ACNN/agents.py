import torch
import torch.nn as nn
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

        # Todo : Convert to nn.Parameter to avoid explicit transfer to cuda
        # Todo : Add a ReLU non-linearity to connect_net's output

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

        batch_size, c_in, h, w = out1.shape
        _, c_out, kh, kw = out2.shape
        new_h, new_w = h - kh + 1, h - kw + 1

        out3 = torch.zeros((batch_size, c_out, new_h, new_w),
                           device=self.device)  # need not set requires_grad = True, explicitly

        # TODO : using matrix multiplication with F.conv2d

        for i in range(batch_size):
            i_out2 = torch.squeeze(out2[i])[:, None, :, :]
            i_out2 = i_out2.repeat(1, c_in, 1, 1)  # broadcasting
            out3[i] = self.connect_net(out1[i], i_out2)

        out3 = out3.reshape(batch_size, -1)
        out3 = self.fc(out3)
        return out3
