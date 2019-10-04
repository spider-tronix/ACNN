import sys
from os.path import dirname, abspath

import torch.nn.functional as F
from torch import nn

sys.path.append(dirname(dirname(abspath(__file__))))
from utilities.train_helpers import grouped_conv


class BasicBlock(nn.Module):
    """Performs a single ResNet operation with sub-sampling and down-sampling if necessary"""

    def __init__(self, in_filters, out_filters, downsample=None, stride=1):
        """
        Init all class variables
        :param in_filters: Input Channels
        :param out_filters: Output Channels
        :param downsample: Either None or a nn.Sequential to perform if necessary
        :param stride: For sub-sampling layers
        """
        super(BasicBlock, self).__init__()

        self.single_block = nn.Sequential(
            nn.Conv2d(in_filters, out_filters, 3, padding=1, bias=False, stride=stride),
            nn.BatchNorm2d(out_filters),
            nn.ReLU(),
            nn.Conv2d(out_filters, out_filters, 3, padding=1, bias=False),
        )
        self.bn2 = nn.BatchNorm2d(out_filters)
        self.downsample = downsample

    def forward(self, x):
        identity = x  # Residual
        out = self.single_block(x)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # Residual Connection
        return F.relu(out)


class CifarResNet(nn.Module):
    """
    ResNet designed for CIFAR dataset.
    The first layer is 3×3 convolutions. Then we use a stack of 6n layers
    with 3×3 convolutions on the feature maps of sizes {32; 16; 8}
    respectively,with 2n layers for each feature map size. The numbers of
    filters are {16; 32; 64} respectively. The sub-sampling is performed
    by convolutions with a stride of 2.

    There are totally 6n+1 stacked weighted layers and 3n residual connections.
    """

    def __init__(self, layers, in_channels=3):
        """
        Init all class variables
        :param layers: List that depends on n. Function included for auto decision
        :param in_channels: No. of channels in dataset image. 3 if RGB, 1 if grayscale
        """
        super(CifarResNet, self).__init__()

        # Layer 1
        # Output map size : 32x32
        # No of filters: 16
        # No of Weighted Layers : 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.filters = 16  # Update current no of filters. Tracks the need for down-sampling
        layers[0] -= 1  # One down

        # Layer 2
        # Output map size : 32x32
        # No of filters: 16
        # No of Weighted Layers : 2n
        self.layer2 = self._make_layer(16, layers[0])

        # Layer 3
        # Output map size : 16x16
        # No of filters: 32
        # No of Weighted Layers : 2n
        self.layer3 = self._make_layer(32, layers[1])

        # Layer 2
        # Output map size : 8x8
        # No of filters: 64
        # No of Weighted Layers : 2n
        self.layer4 = self._make_layer(64, layers[2])

        self._weights_initialise()  # Zero initialize the weights for better performance

    def _make_layer(self, filters, no_layers):
        """
        Generates ResNet Sub-Block
        :param filters: Required no of filter in sub-block
        :param no_layers: Required no of BasicBlocks in sub-block
        :return: Sequential Model of the Sub-Block
        """
        downsample = None
        stride = 1

        if self.filters != filters:  # Case for down-sampling
            downsample = nn.Sequential(
                nn.Conv2d(self.filters, filters, 1, stride=2, bias=False),  # Conv1x1 with stride 2
                nn.BatchNorm2d(filters),
            )
            stride = 2  # Sub Sample

        layers = [BasicBlock(self.filters, filters, downsample, stride=stride)]  # Sub Sampling with Down Sampling
        self.filters = filters  # Update current no of filters
        for _ in range(1, no_layers):
            layers.append(BasicBlock(self.filters, filters))

        return nn.Sequential(*layers)

    def _weights_initialise(self):
        """
        Zero-initialize the last BN in each residual branch, so that
        the residual branch starts with zeros, and each residual block
        behaves like an identity.

        This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        :return: None
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, BasicBlock):
                m: BasicBlock
                nn.init.constant_(m.bn2.weight, 0)

    # noinspection PyPep8Naming
    def forward(self, X):
        """Forward Prop"""
        out = self.layer1(X)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


# noinspection PyShadowingNames
def resnet(n=5, in_channels=3):
    layers = [n + 1, n, n]
    # 2n refers to no of weighted layers. Class rather takes basic block as that Makes more sense
    return CifarResNet(layers=layers, in_channels=in_channels)


class ACNN(nn.Module):

    def __init__(self, n1=9, n2=3, in_channels=3, no_classes=10):
        super(ACNN, self).__init__()
        self.features_net = resnet(n=n1, in_channels=in_channels)
        self.filters_net = resnet(n=n2, in_channels=in_channels)

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, no_classes),
            nn.LogSoftmax(dim=1)
        )

    # noinspection PyPep8Naming
    def forward(self, X):
        out1 = self.features_net(X)
        out2 = self.features_net(X)

        out = grouped_conv(out1, out2)

        return self.fc(out)


class BenchmarkResNet(nn.Module):
    def __init__(self, n=9, in_channels=3, out_channels=10):
        super(BenchmarkResNet, self).__init__()
        self.net1 = resnet(n, in_channels)
        self.fc = nn.Linear(64, out_channels)

    def forward(self, x):
        out = self.net1(x)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
