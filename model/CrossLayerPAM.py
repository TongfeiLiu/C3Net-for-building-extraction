import torch
import torch.nn as nn
import torch.nn.functional as F

class CLAM(nn.Module):
    def __init__(self, hf_channels, lf_channels, out_channels):
        """
        #1: hf_channels denote the number of the high_feats_channels
        #2: lf_channels denote the number of the low_feats_channels
        #3: out_channels denote the number of the out_channels
        """
        super(CLAM, self).__init__()
        self.W_high = nn.Sequential(
            nn.Conv2d(hf_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels)
        )

        self.W_low = nn.Sequential(
            nn.Conv2d(lf_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels)
        )

        self.soft_att = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(lf_channels + hf_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, high_feats, low_feats):
        g1 = self.W_high(high_feats)    # g refers to the first decoder part output
        x1 = self.W_low(low_feats)      #x refers to the second encoder part output
        psi = self.relu(g1 + x1)
        soft_score = self.soft_att(psi)
        low_feats = low_feats * soft_score
        feats = torch.cat([low_feats, high_feats], dim=1)
        out = self.conv1x1(feats)

        return out