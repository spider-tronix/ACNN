import torch.nn.functional as F
from torch import nn


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    """Handles a single Resnet Block"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        Init all variables for class
        :param inplanes: Input channels
        :param planes: Out put Channels
        :param stride: Stride for Conv layer
        :param downsample: Flag to indicate in channel dimension
        """
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.downsample = downsample

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.stride = stride

    def forward(self, x):
        """
        Forward Prop Implementation
        :param x: Input to single block. Either from prior block or initial input to network
        :return: Output from single Resnet block. To pass into next block or as output of last block
        """
        identity = x  # Remember input

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # Down Sampling to adjust channel dimension

        out += identity  # Magic!!
        out = self.relu(out)

        return out


class ResNetUnit(nn.Module):
    """Handles entire structure"""

    def __init__(self, architecture=None, num_classes=10):
        """
        :param architecture: Decided from required depth of network
        :param num_classes: Possible number of unique labels of classification
        """
        super(ResNetUnit, self).__init__()
        if architecture is None:
            architecture = [1, 1, 1, 1]
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2d(1, 64,
                               kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, architecture[0])  # Layer 1
        self.layer2 = self._make_layer(128, architecture[1], stride=2)  # Layer 2
        self.layer3 = self._make_layer(256, architecture[2], stride=2)  # Layer 3
        self.layer4 = self._make_layer(512, architecture[3], stride=2)  # Layer 4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        self._zero_init()

    def _zero_init(self):
        """
        Zero-initialize the last BN in each residual branch, so that the residual branch starts with zeros,
        and each residual block behaves like an identity.This improves the model by 0.2~0.3% according to
        https://arxiv.org/abs/1706.02677
        :return:  None
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        """
        Concatenates BasicBlocks to create a single unit
        :param planes: Output channels of BasicBlocks
        :param blocks: Number of BasicBlocks to concatenate
        :param stride: Stride for Conv operation
        :return: Single unit of ResNet as a Sequential model
        """
        block = BasicBlock
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            """Generate down sampling based on existence of mismatch of dimensions"""
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward Prop of Final Network
        :param x: Input image from an iterating dataloader with batch dimension
        :return: Output after propagation
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        return x
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        # return F.log_softmax(x, dim=1)  # Thanks to https://github.com/sachin-101 for a great tip


class ACNNResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = ResNetUnit()
        self.net2 = ResNetUnit()
        # self.net2 = nn.Sequential(
        #     nn.Conv2d(1, 16,
        #               kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32,
        #               kernel_size=5, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64,
        #               kernel_size=5, stride=2),
        #     nn.ReLU()
        # )

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, X):
        """
        Forward Prop
        :param X: Input dataset with batch dimension
        :return: Output of model and parameters
        """
        out1 = F.relu(self.net1(X))
        out2 = F.relu(self.net2(X))

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
        return F.log_softmax(out3, dim=1)
