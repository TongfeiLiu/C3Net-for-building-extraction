import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) 2次"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, bias=False)
        )
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, bias=False)
        )
        self.DualConv = DoubleConv(in_channels, out_channels)
    def forward(self, x):
        x1 = self.maxpool_conv(x)
        x = self.avgpool_conv(x)
        x = torch.cat([x1, x], dim=1)
        return self.DualConv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, x2):
        x = self.up(x)
        #x2 = self.clpam(x, x2)
        # input is CHW
#        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
#        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

#        x1 = F.pad(x1,[diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)