import os
import gc
import time
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
import errno

from model import SleepModel
# from model2 import SleepModel
from config import config as cfg

from train_fc_model import DataLoad, load_model


def init_test(test_model):
    all_data = DataLoad()
    test_data = torch.from_numpy(all_data.test_data).float().to(cfg.device)
    # print(test_data.shape)
    model = SleepModel(5, is_training=True)

    model = model.to(cfg.device)
    if cfg.cuda:
        cudnn.benchmark = True

    if cfg.resume:
        load_model(model, test_model)

    return model, test_data


def test(test_model):
    model, test_data = init_test(test_model)
    model.eval()
    output = model(test_data[:, 1:])
    print(output.shape)

if __name__ == "__main__":
    test_model = "./model/sleep1/FC_100.pth"

    test(test_model)

