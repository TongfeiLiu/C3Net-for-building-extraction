import torch
import torch.nn as nn
from torch.nn import Module, Conv2d
from model.ChannelAttention import ChannelAttention

class ContextAM(Module):
    """ Position attention module (citation: 2019-CVPR-DANet)"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(ContextAM, self).__init__()
        self.chanel_in = in_dim

        self.Query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.Value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.ca = ChannelAttention(in_channels=in_dim)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        Q = self.Query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        K = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(Q, K)

        att_mask = self.sigmoid(energy)

        V = self.Value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(V, att_mask.permute(0, 2, 1))

        out = out.view(m_batchsize, C, height, width)

        out = out + x
        out = self.ca(out)
        return out


class ContentAM(Module):
    """ Position attention module (catation: 2019-CVPR-DANet)"""
    #Ref from SAGAN
    def __init__(self, hf_channels=1024, lf_channels=1024):
        super(ContentAM, self).__init__()
        self.chanel_in = hf_channels

        self.Query_conv = Conv2d(in_channels=hf_channels, out_channels=hf_channels//8, kernel_size=1)
        self.Key_conv = Conv2d(in_channels=hf_channels, out_channels=hf_channels//8, kernel_size=1)
        self.Value_conv = Conv2d(in_channels=lf_channels, out_channels=lf_channels, kernel_size=1)

        self.ca = ChannelAttention(in_channels=hf_channels)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x, x1):
        m_batchsize, C, height, width = x.size()
        Q = self.Query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        K = self.Key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(Q, K)

        att_mask = self.sigmoid(energy)

        V = self.Value_conv(x1).view(m_batchsize, -1, width*height)

        out = torch.bmm(V, att_mask.permute(0, 2, 1))

        out = out.view(m_batchsize, C, height, width)

        out = self.ca(out)
        return out