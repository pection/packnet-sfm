import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F


class Conv2D(nn.Module):
    """
    2D convolution with GroupNorm and ELU

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int
        Kernel size
    stride : int
        Stride
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
        self.pad = nn.ConstantPad2d([kernel_size // 2] * 4, value=0)
        self.normalize = torch.nn.GroupNorm(16, out_channels)
        self.activ = nn.ELU(inplace=True)

    def forward(self, x):
        """Runs the Conv2D layer."""
        x = self.conv_base(self.pad(x))
        return self.activ(self.normalize(x))


class ResidualConv(nn.Module):
    """2D Convolutional residual block with GroupNorm and ELU"""

    def __init__(self, in_channels, out_channels, stride, dropout=None):
        """
        Initializes a ResidualConv object.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        stride : int
            Stride
        dropout : float
            Dropout value
        """
        super().__init__()
        self.conv1 = Conv2D(in_channels, out_channels, 3, stride)
        self.conv2 = Conv2D(out_channels, out_channels, 3, 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.normalize = torch.nn.GroupNorm(16, out_channels)
        self.activ = nn.ELU(inplace=True)

        if dropout:
            self.conv3 = nn.Sequential(self.conv3, nn.Dropout2d(dropout))

    def forward(self, x):
        """Runs the ResidualConv layer."""
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        shortcut = self.conv3(x)
        return self.activ(self.normalize(x_out + shortcut))
