import torch
import torch.nn as nn


def double_convolution(in_channels, out_channels):
    """
    Double convolution block.
    :param in_channels: int, input channels
    :param out_channels: int, output channels
    :return: nn.Sequential, double convolution block
    """
    conv_op = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
    return conv_op


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_convolution_1 = double_convolution(3, 64)
        self.down_convolution_2 = double_convolution(64, 128)
        self.down_convolution_3 = double_convolution(128, 256)

        self.up_transpose_1 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128,
            kernel_size=2,
            stride=2)
        self.up_convolution_1 = double_convolution(256, 128)
        self.up_transpose_2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64,
            kernel_size=2,
            stride=2)
        self.up_convolution_2 = double_convolution(128, 64)
        self.out = nn.Conv2d(
            in_channels=64, out_channels=1,
            kernel_size=1
        )

    def forward(self, x):
        down_1 = self.down_convolution_1(x)
        down_2 = self.max_pool2d(down_1)
        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool2d(down_3)
        down_5 = self.down_convolution_3(down_4)

        up_1 = self.up_transpose_1(down_5)
        x = self.up_convolution_1(torch.cat([down_3, up_1], 1))
        up_2 = self.up_transpose_2(x)
        x = self.up_convolution_2(torch.cat([down_1, up_2], 1))
        out = self.out(x)
        return out