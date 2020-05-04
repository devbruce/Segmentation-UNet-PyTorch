import os
import torch
from config import CKPT_DIR


__all__ = ['to_numpy', 'denormalization', 'classify_class', 'save_net', 'load_net']


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy().transpose(0, 2, 3, 1)  # (Batch, H, W, C)

def denormalization(data, mean, std):
    return (data * std) + mean

def classify_class(x):
    return 1.0 * (x > 0.5)

def save_net(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    torch.save(
        {'net': net.state_dict(),'optim': optim.state_dict()},
        os.path.join(ckpt_dir, f'model_epoch{epoch:04}.pth'),
    )
    
def load_net(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        
    ckpt_list = os.listdir(ckpt_dir)
    ckpt_list.sort(key=lambda fname: int(''.join(filter(str.isdigit, fname))))
    
    ckpt_path = os.path.join(CKPT_DIR, ckpt_list[-1])
    model_dict = torch.load(ckpt_path)
    print(f'* Load {ckpt_path}')

    net.load_state_dict(model_dict['net'])
    optim.load_state_dict(model_dict['optim'])
    epoch = int(''.join(filter(str.isdigit, ckpt_list[-1])))
    
    return net, optim, epoch
