import torch
import torch.nn as nn

from models.FAM.Dynamic_offset_estimator import Dynamic_offset_estimator
from mmcv.ops.deform_conv import DeformConv2d
from util.util import saveoffset, showpatch

class DeformableConvBlock(nn.Module):
    def __init__(self, input_channels):
        super(DeformableConvBlock, self).__init__()

        self.offset_estimator =  Dynamic_offset_estimator(input_channelsize=input_channels)
        self.offset_conv = nn.Conv2d(in_channels=input_channels, out_channels=1 * 2 * 9, kernel_size=3, padding=1,bias=False)

        self.deformconv = DeformConv2d(in_channels=768, out_channels=768, kernel_size=3,
                                       padding=1, bias=False)

    def forward(self, input_features, reference_features):
        
        input_offset = torch.cat((input_features, reference_features), dim=1)
        estimated_offset = self.offset_estimator(input_offset)
        estimated_offset = self.offset_conv(estimated_offset)
        output = self.deformconv(x=reference_features, offset=estimated_offset)
        

        return output
        # 返回aligned feature