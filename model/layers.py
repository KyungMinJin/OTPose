import torch.nn as nn

from torch.nn import functional as F
from thirdparty.deform_conv import ModulatedDeformConv

BN_MOMENTUM = 0.1


def modulated_deform_conv(n_channels, kernel_height, kernel_width, deformable_dilation, deformable_groups):
    conv_offset2d = ModulatedDeformConv(
        n_channels,
        n_channels,
        (kernel_height, kernel_width),
        stride=1,
        padding=int(kernel_height / 2) * deformable_dilation,
        dilation=deformable_dilation,
        deformable_groups=deformable_groups
    )
    return conv_offset2d


class DeformableCONV(nn.Module):
    def __init__(self, num_joints, k, dilation):
        super(DeformableCONV, self).__init__()

        self.deform_conv = modulated_deform_conv(num_joints, k, k, dilation, num_joints).cuda()

    def forward(self, x, offsets, mask):
        return self.deform_conv(x, offsets, mask)
