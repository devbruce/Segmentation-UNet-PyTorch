import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchvision import transforms
from dataset import *
from torch.utils.data import DataLoader
from model import UNet
from utils import *
from config import *

# Evaluation
# 1. Delete RandomFlip
# 2. shuffle=False
# 3. Tensorboard 사용 X
# 4. Train X (Epoch 존재하지 않음)

cfg = Config()
transform = transforms.Compose([
    GrayscaleNormalization(mean=0.5, std=0.5),
    ToTensor(),
])

RESULTS_DIR = os.path.join(ROOT_DIR, 'test_results')
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

test_dataset = Dataset(imgs_dir=TEST_IMGS_DIR, labels_dir=TEST_LABELS_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

test_data_num = len(test_dataset)
test_batch_num = int(np.ceil(test_data_num / cfg.BATCH_SIZE)) # np.ceil 반올림

# Network
net = UNet().to(device)

# Loss Function
loss_fn = nn.BCEWithLogitsLoss().to(device)

# Optimizer
optim = torch.optim.Adam(params=net.parameters(), lr=cfg.LEARNING_RATE)

start_epoch = 0

# Load Checkpoint File
if os.listdir(CKPT_DIR):
    net, optim, _ = load_net(ckpt_dir=CKPT_DIR, net=net, optim=optim)

# Evaluation
with torch.no_grad():
    net.eval()  # Evaluation Mode
    loss_arr = list()

    for batch_idx, data in enumerate(test_loader, 1):
        # Forward Propagation
        img = data['img'].to(device)
        label = data['label'].to(device)

        output = net(img)

        # Calc Loss Function
        loss = loss_fn(output, label)
        loss_arr.append(loss.item())
        
        print_form = '[Test] | Batch: {:0>4d} / {:0>4d} | Loss: {:.4f}'
        print(print_form.format(batch_idx, test_batch_num, loss_arr[-1]))

        # Tensorboard
        img = to_numpy(denormalization(img, mean=0.5, std=0.5))
        label = to_numpy(label)
        output = to_numpy(classify_class(output))
        
        for j in range(label.shape[0]):
            crt_id = int(test_batch_num * (batch_idx - 1) + j)
            
            plt.imsave(os.path.join(RESULTS_DIR, f'img_{crt_id:04}.png'), img[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(RESULTS_DIR, f'label_{crt_id:04}.png'), label[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(RESULTS_DIR, f'output_{crt_id:04}.png'), output[j].squeeze(), cmap='gray')
            
print_form = '[Result] | Avg Loss: {:0.4f}'
print(print_form.format(np.mean(loss_arr)))
