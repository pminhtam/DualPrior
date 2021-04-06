import torch.nn as nn
# from .DnCNN import DnCNN
from .UNet import UNet
import torch
def weight_init_kaiming(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaimisng_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if not m.bias is None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return net

class VDN(nn.Module):
    def __init__(self, in_channels, wf=64, dep_S=5, dep_U=4, slope=0.2):
        super(VDN, self).__init__()
        self.DNet = UNet(in_channels, in_channels*2, wf=wf, depth=dep_U, slope=slope)

    def forward(self, x):
        phi_Z = self.DNet(x)
        return phi_Z
