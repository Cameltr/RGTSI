import time
import pdb
from options.test_options import TestOptions
from data.dataprocess import DataProcess
from models.model import create_model
import torchvision
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import os
import torch
from PIL import Image
import numpy as np
from glob import glob
from tqdm import tqdm
import torchvision.transforms as transforms
if __name__ == "__main__":

    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    results_dir = r'./result/'
    if not os.path.exists( results_dir):
        os.mkdir(results_dir)

    
    opt = TestOptions().parse()
    writer = SummaryWriter(log_dir=dir, comment=opt.name)
    model = create_model(opt)

    net_EN = torch.load("./checkpoints/RGTSI/70_net_EN.pth")
    net_DE = torch.load("./checkpoints/RGTSI/70_net_DE.pth")
    net_RGTSI = torch.load("./checkpoints/RGTSI/70_net_RGTSI.pth")

    model.netEN.module.load_state_dict(net_EN['net'])
    model.netDE.module.load_state_dict(net_DE['net'])
    model.netRGTSI.module.load_state_dict(net_RGTSI['net'])

    input_mask_paths = glob('{:s}/*'.format("/project/liutaorong/RGTSI/data/DPED10K/test/input_mask/3/"))
    input_mask_paths.sort()
    de_paths = glob('{:s}/*'.format("/project/liutaorong/RGTSI/data/DPED10K/test/images/"))
    de_paths.sort()
    st_path = glob('{:s}/*'.format("/project/liutaorong/RGTSI/data/DPED10K/test/structure/"))
    st_path.sort()
    ref_paths = glob('{:s}/*'.format("/project/liutaorong/RGTSI/data/DPED10K/test/reference/"))
    ref_paths.sort()

    image_len = len(de_paths)

    for i in tqdm(range(image_len)):
        # only use one mask for all image
        path_im = input_mask_paths[11]
        path_de = de_paths[11]
        (filepath,tempfilename) = os.path.split(path_de)
        (filename,extension) = os.path.splitext(tempfilename)
        path_st = st_path[11]
        path_rf = ref_paths[0]
        
        input_mask = Image.open(path_im).convert("RGB")
        detail = Image.open(path_de).convert("RGB")
        structure = Image.open(path_st).convert("RGB")
        reference = Image.open(path_rf).convert("RGB")

        input_mask = mask_transform(input_mask)
        detail = img_transform(detail)
        structure = img_transform(structure)
        reference = img_transform(reference)
        
        input_mask = torch.unsqueeze(input_mask, 0)
        detail = torch.unsqueeze(detail, 0)
        structure = torch.unsqueeze(structure,0)
        reference = torch.unsqueeze(reference,0)

        with torch.no_grad():
            model.set_input(detail,structure,input_mask,reference)
            model.forward()
            fake_out = model.fake_out
            fake_out = fake_out.detach().cpu() * input_mask + detail*(1-input_mask)
            fake_image = (fake_out+1)/2.0
        output = fake_image.detach().numpy()[0].transpose((1, 2, 0))*255
        output = Image.fromarray(output.astype(np.uint8))
        output.save(results_dir+filename+".jpg")
        
        input, reference, output, GT = model.get_current_visuals()
        image_out = torch.cat([input,reference,output,GT], 0)
        grid = torchvision.utils.make_grid(image_out)
        writer.add_image('picture(%d)' % i,grid,i)
