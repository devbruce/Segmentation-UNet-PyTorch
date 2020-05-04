import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataset import *
from model import UNet
from utils import *
from config import *


cfg = Config()
transform = transforms.Compose([
    GrayscaleNormalization(mean=0.5, std=0.5),
    RandomFlip(),
    ToTensor(),
])

# Set Dataset
train_dataset = Dataset(imgs_dir=TRAIN_IMGS_DIR, labels_dir=TRAIN_LABELS_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
val_dataset = Dataset(imgs_dir=VAL_IMGS_DIR, labels_dir=VAL_LABELS_DIR, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)

train_data_num = len(train_dataset)
val_data_num = len(val_dataset)

train_batch_num = int(np.ceil(train_data_num / cfg.BATCH_SIZE)) # np.ceil 반올림
val_batch_num = int(np.ceil(val_data_num / cfg.BATCH_SIZE))

# Network
net = UNet().to(device)

# Loss Function
loss_fn = nn.BCEWithLogitsLoss().to(device)

# Optimizer
optim = torch.optim.Adam(params=net.parameters(), lr=cfg.LEARNING_RATE)

# Tensorboard
train_writer = SummaryWriter(log_dir=TRAIN_LOG_DIR)
val_writer = SummaryWriter(log_dir=VAL_LOG_DIR)

# Training
start_epoch = 0
# Load Checkpoint File
if os.listdir(CKPT_DIR):
    net, optim, start_epoch = load_net(ckpt_dir=CKPT_DIR, net=net, optim=optim)
else:
    print('* Training from scratch')

num_epochs = cfg.NUM_EPOCHS
for epoch in range(start_epoch+1, num_epochs+1):
    net.train()  # Train Mode
    train_loss_arr = list()
    
    for batch_idx, data in enumerate(train_loader, 1):
        # Forward Propagation
        img = data['img'].to(device)
        label = data['label'].to(device)
        
        output = net(img)
        
        # Backward Propagation
        optim.zero_grad()
        
        loss = loss_fn(output, label)
        loss.backward()
        
        optim.step()
        
        # Calc Loss Function
        train_loss_arr.append(loss.item())
        print_form = '[Train] | Epoch: {:0>4d} / {:0>4d} | Batch: {:0>4d} / {:0>4d} | Loss: {:.4f}'
        print(print_form.format(epoch, num_epochs, batch_idx, train_batch_num, train_loss_arr[-1]))
        
        # Tensorboard
        img = to_numpy(denormalization(img, mean=0.5, std=0.5))
        label = to_numpy(label)
        output = to_numpy(classify_class(output))
        
        global_step = train_batch_num * (epoch-1) + batch_idx
        train_writer.add_image(tag='img', img_tensor=img, global_step=global_step, dataformats='NHWC')
        train_writer.add_image(tag='label', img_tensor=label, global_step=global_step, dataformats='NHWC')
        train_writer.add_image(tag='output', img_tensor=output, global_step=global_step, dataformats='NHWC')
        
    train_loss_avg = np.mean(train_loss_arr)
    train_writer.add_scalar(tag='loss', scalar_value=train_loss_avg, global_step=epoch)
    
    # Validation (No Back Propagation)
    with torch.no_grad():
        net.eval()  # Evaluation Mode
        val_loss_arr = list()
        
        for batch_idx, data in enumerate(val_loader, 1):
            # Forward Propagation
            img = data['img'].to(device)
            label = data['label'].to(device)
            
            output = net(img)
            
            # Calc Loss Function
            loss = loss_fn(output, label)
            val_loss_arr.append(loss.item())
            
            print_form = '[Validation] | Epoch: {:0>4d} / {:0>4d} | Batch: {:0>4d} / {:0>4d} | Loss: {:.4f}'
            print(print_form.format(epoch, num_epochs, batch_idx, val_batch_num, val_loss_arr[-1]))
            
            # Tensorboard
            img = to_numpy(denormalization(img, mean=0.5, std=0.5))
            label = to_numpy(label)
            output = to_numpy(classify_class(output))
            
            global_step = val_batch_num * (epoch-1) + batch_idx
            val_writer.add_image(tag='img', img_tensor=img, global_step=global_step, dataformats='NHWC')
            val_writer.add_image(tag='label', img_tensor=label, global_step=global_step, dataformats='NHWC')
            val_writer.add_image(tag='output', img_tensor=output, global_step=global_step, dataformats='NHWC')
            
    val_loss_avg = np.mean(val_loss_arr)
    val_writer.add_scalar(tag='loss', scalar_value=val_loss_avg, global_step=epoch)
    
    print_form = '[Epoch {:0>4d}] Training Avg Loss: {:.4f} | Validation Avg Loss: {:.4f}'
    print(print_form.format(epoch, train_loss_avg, val_loss_avg))
    
    save_net(ckpt_dir=CKPT_DIR, net=net, optim=optim, epoch=epoch)
    
train_writer.close()
val_writer.close()
