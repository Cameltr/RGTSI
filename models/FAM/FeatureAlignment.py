import torch.nn as nn
import torch

from models.FAM.DeformableBlock import DeformableConvBlock,DeformableConvBlock_show
from util.util import showpatch

class FAM(nn.Module):
    def __init__(self,in_channels):
        super(FAM, self).__init__()
        self.deformblock = DeformableConvBlock(input_channelsize= in_channels)
        
        #对齐了两次次
    def forward(self,ist_feature, rst_feature):

        st_out = self.deformblock(ist_feature, rst_feature) #输出aligned feature
        out = torch.add(ist_feature,st_out)

        return out 