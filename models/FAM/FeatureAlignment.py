import torch.nn as nn
import torch

from models.FAM.DeformableBlock import DeformableConvBlock,DeformableConvBlock_show
from util.util import showpatch

class FAM(nn.Module):
    def __init__(self,in_channels):
        super(FAM, self).__init__()
        self.deformblock = DeformableConvBlock(input_channels= in_channels)
        #self.deformblock2 = DeformableConvBlock(input_channels= in_channels,mode=mode)
        
        #对齐了两次次
    def forward(self,ist_feature, rst_feature):

        st_out = self.deformblock(ist_feature, rst_feature) #输出aligned feature
        #st_out2 = self.deformblock2(input_feature, st_out1)
        out = torch.add(ist_feature,st_out)

        return out 

#查看每个偏移后输出的图像
class FAM_show(nn.Module):
    def __init__(self,in_channels,mode):
        super(FAM_show, self).__init__()
        self.deformblock1 = DeformableConvBlock_show(input_channels= in_channels,mode = mode)
        self.deformblock2 = DeformableConvBlock_show(input_channels= in_channels, mode = mode)
        self.deformblock3 = DeformableConvBlock_show(input_channels=in_channels, mode = mode)

    def forward(self,ist_feature, rst_feature, modelname, showmode = True):

        st_out1 = self.deformblock1(ist_feature, rst_feature,showmode = showmode,num_block = 1, modelname = modelname)
        st_out2 = self.deformblock2(ist_feature, st_out1, showmode = showmode, num_block = 2, modelname = modelname)
        st_out3 = self.deformblock3(ist_feature, st_out2, showmode = showmode, num_block = 3, modelname = modelname )

        if showmode:
            showpatch(st_out1, foldername="extracted_structure_by_deformconv1", modelname=modelname)
            showpatch(st_out2, foldername="extracted_structure_by_deformconv2", modelname=modelname)
            showpatch(st_out3, foldername="extracted_structure_by_deformconv3", modelname=modelname)

        return st_out3
