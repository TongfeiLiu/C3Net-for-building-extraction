import torch.nn as nn
import torch
from model.CBAM import SpatialAttentionModule
from model.High_Frequency_Module import HighFrequencyModule


class Res_EdgeRM(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        # row: (1,4,8,16)
        scale=[1, 4, 8, 16]
        super().__init__()
        self.center_block_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding='same', dilation=(scale[0], scale[0])),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv1x1_1 = nn.Conv2d(out_channels + in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1))

        self.center_block_4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding='same', dilation=(scale[1], scale[1])),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv1x1_2 = nn.Conv2d(out_channels + in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1))

        self.center_block_8 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding='same', dilation=(scale[2], scale[2])),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv1x1_3 = nn.Conv2d(out_channels + in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1))

        self.center_block_16 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding='same', dilation=(scale[3], scale[3])),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv1x1_4 = nn.Conv2d(out_channels + in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1))

        # Edge
        self.sa = SpatialAttentionModule(kernel_size=3)
        self.edge = HighFrequencyModule(input_channel=in_channels, the_filter='Isotropic_Sobel')

        self.conv = nn.Conv2d(out_channels, 1, kernel_size=(3, 3), padding='same')

    def forward(self, x):
        edge = self.edge(self.sa(x))

        cb1 = self.center_block_1(x)
        cb1 = self.conv1x1_1(torch.cat([cb1, edge], dim=1))

        cb2 = self.center_block_4(cb1)
        cb2 = self.conv1x1_1(torch.cat([cb2, edge], dim=1))

        cb3 = self.center_block_8(cb2)
        cb3 = self.conv1x1_1(torch.cat([cb3, edge], dim=1))

        cb4 = self.center_block_16(cb3)
        cb4 = self.conv1x1_1(torch.cat([cb4, edge], dim=1))

        cb = cb1 + cb2 + cb3 + cb4
        out = self.conv(cb)
        return out