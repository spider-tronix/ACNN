import torch
import torch.nn as nn


class ParallelNets(nn.Module):
    """Branches of the Network"""

    def __init__(self, net1_channels=(1, 10, 20),
                 net2_channels=(1, 10, 20, 30),
                 kernel_size=5, stride=1, padding=2):
        """
        Init all variables for class
        """
        super(ParallelNets, self).__init__()
        self.net1conv1 = nn.Sequential(
            nn.Conv2d(net1_channels[0], net1_channels[1],
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )
        self.net1conv2 = nn.Sequential(
            nn.Conv2d(net1_channels[1], net1_channels[2],
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )
        self.net2conv1 = nn.Sequential(
            nn.Conv2d(net2_channels[0], net2_channels[1],
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )
        self.net2conv2 = nn.Sequential(
            nn.Conv2d(net2_channels[1], net2_channels[2],
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )
        self.net2conv3 = nn.Sequential(
            nn.Conv2d(net2_channels[2], net2_channels[3],
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )


class ConnectNet(torch.nn.Module):
    """Convolution operator of Branches"""

    def __init__(self, input_dim, kernel_size, strides=1, padding='valid', device='cpu'):
        """
        Init all variables for class
        :param input_dim: dimension of the tensor passed in forward function
        :param kernel_size: tuple, dimension of kernel/filter
        :param strides: an integer specifying the stride length
        :param padding: either 'valid' or 'same'. 'valid' refers to no padding.
        :param device: either 'valid' or 'same'. 'valid' refers to no padding.
                      'same' refers to padding the input such that the output has
                      the same length as the original input
        """
        super(ConnectNet, self).__init__()
        self.input_dim = input_dim
        self.ks = kernel_size
        self.s = strides
        self.padding = padding
        self.device = device

    def forward(self, x, kernels):
        """
        Parameters:
            x : image tensor from previous layer of "features" network,
                of dimension (C, H, W)
            kernels : kernels(tensor) to convolve with x, of dimension (C_out, C_in, h_filter, w_filter)
        Returns:
            x_out : x convolved with kernels,tensor of dimension (C_out, H_out, W_out)

            where :
            C - Channels
            H - Height
            W - width
            H_out = floor(H - h_filter + 2*padding) + 1
            W_out = floor(W - w_filter + 2*padding) + 1
        """

        assert (x.shape == self.input_dim)  # assert right input
        C_out, _, hf, wf = kernels.shape
        x = self.pad_tensor(x)  # pad input
        x_col, h_out, w_out = self.im2col(x, hf, wf, self.s)
        x_col = x_col.t()
        x_col = x_col.to(self.device, dtype=torch.float)
        x_col.requires_grad = True

        k_col = kernels.view(C_out, -1)  # converted to 2d tensor
        k_col = k_col.to(device=device, dtype=torch.float)  # to gpu
        k_col.requires_grad = True
        x_out = torch.mm(k_col, x_col).view(C_out, h_out, w_out)  # convolution
        return x_out

    def pad_tensor(self, x):
        """
        Parameters :
            x : tensor to be padded of dimension (C,H,W)
        Returns :
            padded tesnor
        """
        if self.padding == 'same':
            pad_h, pad_w = int((self.ks[0] - 1) / 2), int((self.ks[1] - 1) / 2)
            pad = ConstantPad3d((0, 0, pad_h, pad_h, pad_w, pad_w), value=0)
            return pad(x)
        return x

    def im2col(self, x, hf, wf, stride):

        """
        Parameters:
            x: image tensor to be translated into columns, (C,H,W)
            hf: filter height
            wf: filter width
            stride: stride
        Returns:
            col: tensor of dimension (h_out*w_out,hf*wf*C), where each column is a cube that will convolve with a filter
                h_out = (H-hf) // stride + 1, w_out = (W-wf) // stride + 1
        """

        c, h, w = x.shape
        h_out = (h - hf) // stride + 1
        w_out = (w - wf) // stride + 1
        x_col = torch.zeros(h_out * w_out, c * hf * wf)

        for i in range(h_out):
            for j in range(w_out):
                patch = x[..., i * stride:i * stride + hf, j * stride:j * stride + wf]
                x_col[i * w_out + j, :] = patch.reshape(-1)  # patch.reshape(-1)
        return x_col, h_out, w_out
