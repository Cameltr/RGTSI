from models.RGTSI import RGTSI
import torch


def create_model(opt):
    model = RGTSI(opt)
    #model = torch.nn.DataParallel(model.to(opt.device), device_ids=opt.gpu_ids, output_device=opt.gpu_ids[0])
    print("model [%s] was created" % (model.name()))
    return model

