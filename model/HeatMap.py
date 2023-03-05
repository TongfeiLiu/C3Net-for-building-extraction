import torch
import torch.nn as nn
# from torchvision import models
from model.CPSCNet import *
from model.EdgeRM import Res_EdgeRM
from model.High_Frequency_Module import HighFrequencyModule

class Heatmap(nn.Module):
    """ Full assembly of the parts to form the complete network """
    def __init__(self):
        super(Heatmap, self).__init__()
        self.model = CPSCNet(n_channels=3, n_classes=1)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device=device)
        # print(self.net)
        self.model.load_state_dict(torch.load('Backbone_CLAM_ASPP_C2AM-SFusion_EdgeRM_out1_Dice_out2_BCE_BestmIoU_epoch_59_mIoU_89.783295.pth', map_location=device))
        self.sigmoid = nn.Sigmoid()
        # self.EdgeRM = Res_EdgeRM(64, 64)
        # self.model.EdgeRM.edge = HighFrequencyModule(input_channel=64, the_filter='Isotropic_Sobel')
        self.edge = HighFrequencyModule(input_channel=64, the_filter='Isotropic_Sobel')# Isotropic_Sobel
    def forward(self, x):
        x = self.model.inc(x)
        x1 = x

        x = self.model.down1(x)
        x2 = x

        x = self.model.down2(x)
        x3 = x

        x = self.model.down3(x)
        x4 = x

        x = self.model.down4(x)

        context = self.model.ContextAM(x)
        aspp = self.model.aspp(x)
        content = self.model.ContentAM(x, aspp)

        alpha = self.model.sigmoid(self.model.gamma)
        C2AM = alpha * context + (1 - alpha) * content

        x = self.model.up1(C2AM, x4)

        x = self.model.up2(x, x3)

        x = self.model.up3(x, x2)

        x = self.model.up4(x, x1)
        er4 = x
        x = self.model.outConv1(x)
        out1 = self.model.sigmoid(x)

        # rrm = self.RefineModulel(x)  #RRM-RS paper
        # rrm = self.ERM(erm)

        #rrm = self.model.EdgeRM(er4) # er4
        # edge = self.edge(self.model.EdgeRM.sa(er4))
        sa = self.model.EdgeRM.sa(er4)
        # edge = self.edge(self.sigmoid(sa))
        edge = self.edge(self.model.sigmoid(sa))

        cb1 = self.model.EdgeRM.center_block_1(er4)
        cb1 = self.model.EdgeRM.conv1x1_1(torch.cat([cb1, edge], dim=1))

        cb2 = self.model.EdgeRM.center_block_4(cb1)
        cb2 = self.model.EdgeRM.conv1x1_1(torch.cat([cb2, edge], dim=1))

        cb3 = self.model.EdgeRM.center_block_8(cb2)
        cb3 = self.model.EdgeRM.conv1x1_1(torch.cat([cb3, edge], dim=1))

        cb4 = self.model.EdgeRM.center_block_16(cb3)
        cb4 = self.model.EdgeRM.conv1x1_1(torch.cat([cb4, edge], dim=1))

        cb = cb1 + cb2 + cb3 + cb4
        rrm = self.model.EdgeRM.conv(cb)

        out2 = self.model.sigmoid(rrm)

        x = x + rrm
        out = self.model.sigmoid(x)

        return [aspp, context, content, C2AM]
        # return [edge, er4, sa, self.sigmoid(cb)]

if __name__ == '__main__':
    net = CPSCNet(n_channels=3, n_classes=1)
    print(net)