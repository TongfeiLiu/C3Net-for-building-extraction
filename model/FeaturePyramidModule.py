import torch
import torch.nn as nn

class AtrousPyramidModule(nn.Module):
    def __init__(self, in_channel=1024, out_channel=1024, rate=[1, 2, 4, 8]):
        super(AtrousPyramidModule, self).__init__()

        self.atrous_block1 = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.atrous_block2 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=rate[1], dilation=rate[1])
        self.atrous_block3 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=rate[2], dilation=rate[2])
        self.atrous_block4 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=rate[3], dilation=rate[3])

        self.conv_1x1_output = nn.Conv2d(out_channel * 5, out_channel, 1, 1)

    def forward(self, x):
        atrous_block1 = self.atrous_block1(x)
        atrous_block2 = self.atrous_block2(x)
        atrous_block3 = self.atrous_block3(x)
        atrous_block4 = self.atrous_block4(x)

        aspp = torch.cat([x, atrous_block1, atrous_block2, atrous_block3, atrous_block4], dim=1)

        aspp = self.conv_1x1_output(aspp)
        return aspp