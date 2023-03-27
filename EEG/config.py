from easydict import EasyDict
import torch
import os

config = EasyDict()

# config.gpu = "0"
# os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

# dataloader jobs number
config.num_workers = 0

# batch_size
config.batch_size = 256

# training epoch number
config.max_epoch = 100

config.start_epoch = 0

# learning rate
config.lr = 1e-3
config.momentum = 0.9
config.weight_decay = 5e-6
config.optim = "Adam"

# using GPU
config.cuda = False
config.resume = True
config.display_freq = 10
config.save_freq = 10
config.print_freq = 10
config.random_seed = 0

config.save_dir = "./model"
config.exp_name = "sleep1"

config.pretrained_model = "./model/FC_best.pth"
# config.pretrained_model = "./model/sleep1/FC_1000.pth"

config.device = torch.device('cuda') if config.cuda else torch.device('cpu')


def print_config(config):
    print('==========Options============')
    for k, v in config.items():
        print('{}: {}'.format(k, v))
    print('=============End=============')

if __name__ == "__main__":
    import pdb; pdb.set_trace()
