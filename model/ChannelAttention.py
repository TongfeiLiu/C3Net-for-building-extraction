import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()

        self.in_channels = in_channels

        self.linear_1 = nn.Linear(self.in_channels, self.in_channels // 4)
        self.linear_2 = nn.Linear(self.in_channels // 4, self.in_channels)
        self.linear_3 = nn.Linear(self.in_channels * 2, self.in_channels)

    def forward(self, input_):
        n_b, n_c, h, w = input_.size()

        feats1 = F.adaptive_avg_pool2d(input_, (1, 1)).view((n_b, n_c))
        feats1 = F.relu(self.linear_1(feats1))
        feats1 = self.linear_2(feats1)

        feats2 = F.adaptive_max_pool2d(input_, (1, 1)).view((n_b, n_c))
        feats2 = F.relu(self.linear_1(feats2))
        feats2 = self.linear_2(feats2)

        feats = torch.cat([feats1,feats2], dim=1)
        feats = self.linear_3(feats)

        feats = torch.sigmoid(feats)

        feats = feats.view((n_b, n_c, 1, 1))
        outfeats = torch.mul(feats, input_)
        outfeats = input_ + outfeats

        return outfeats
