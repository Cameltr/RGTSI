import torch
import torch.nn as nn

from models.FAM.Dynamic_offset_estimator import Dynamic_offset_estimator
from mmcv.ops.deform_conv import DeformConv2d
from util.util import saveoffset, showpatch

class DeformableConvBlock(nn.Module):
    def __init__(self, input_channels):
        super(DeformableConvBlock, self).__init__()

        self.offset_estimator =  Dynamic_offset_estimator(int_channels=input_channels)
        self.offset_conv = nn.Conv2d(in_channels=input_channels, out_channels=1 * 2 * 9, kernel_size=3, padding=1,bias=False)

        self.deformconv = DeformConv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3,
                                       padding=1, bias=False)

    def forward(self, input_features, reference_features):
        #showpatch(input_features, foldername="input_de", modelname="FAM")
        #showpatch(reference_features, foldername="ref_de", modelname="FAM")
        input_offset = torch.cat((input_features, reference_features), dim=1)
        estimated_offset = self.offset_estimator(input_offset)
        #showpatch(estimated_offset, foldername="offsetinput1",modelname = "FAM")
        estimated_offset = self.offset_conv(estimated_offset)
        #showpatch(estimated_offset, foldername="offsetinput2",modelname = "FAM")
        output = self.deformconv(x=reference_features, offset=estimated_offset)
        #showpatch(input_offset, foldername="output",modelname = "FAM")

        return output
        # 返回aligned feature