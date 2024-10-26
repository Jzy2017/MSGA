from __future__ import print_function

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.data import DataLoader
from dataset import Image_One
from model import myIFCNN
import argparse
import loss
import model
import utils
import time

parser = argparse.ArgumentParser("GQPR")
parser.add_argument('--is_visualize', type = bool, default = True, help = 'whether visualize or not')
parser.add_argument('--dis_batch', type = int, default = 100, help = 'frequency for visualization')
parser.add_argument('--data_ir_root', type = str, default = 'E:/ZZX/zzx-dataset/RoadScene-rgb/VIS')
parser.add_argument('--data_vis_root', type = str, default = 'E:/ZZX/zzx-dataset/RoadScene-rgb/IR')
parser.add_argument('--vis_batch', type = int, default = 50, help = 'frequency for visualization')
parser.add_argument('--device', type = str, default = 'cuda:0', help = 'device for training')
parser.add_argument('--num_epochs', type = int, default = 200, help = 'num of epochs')
parser.add_argument('--batch_size', type = int, default = 4, help = 'batch size')
parser.add_argument('--learning_rate', type = float, default = 0.0001, help = 'learning rate')
parser.add_argument('--netG_path', type = str, default = None, help = 'path for snapshot of netR')
args = parser.parse_args()

use_cuda = False

if args.device != 'cpu':
    use_cuda=True
else:
    use_cuda=False
nz = 200
data = Image_One(1, [args.data_ir_root, args.data_vis_root])
dataloader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,shuffle = True,num_workers = 0)

ssim_val = loss.SSIM()
mse_loss = nn.MSELoss()
if use_cuda:
    ssim_val = ssim_val.cuda()
    mse_loss = mse_loss.cuda()

netG=myIFCNN()
if args.netG_path is not None:
    netG.load_state_dict(torch.load(args.netG_path, map_location='cpu'))
if use_cuda:
    netG = netG.cuda()

optimizer = torch.optim.Adam(netG.parameters(), lr=args.learning_rate, betas=(0.5, 0.999),weight_decay=0.0001)

# Training Loop

# Lists to keep track of progress
img_list = []
# G_losses = []
iters = 0
aa = 1
i = 0
print("Starting Training Loop...")
# For each epoch
for epoch in range(args.num_epochs):
    for i,(src,gt) in enumerate(dataloader):
        optimizer.state_dict()['param_groups'][0]['lr']=optimizer.state_dict()['param_groups'][0]['lr']/2.
        netG.zero_grad()
        if use_cuda:
            src=src.cuda()
            gt=gt.cuda()
        gt=gt.float()
        src=src.float()
        out=netG(src)
        mse = mse_loss(gt, out)
        ssim = 1-ssim_val(gt, out)
        loss_all = mse + 100 * ssim
        # Calculate gradients for G
        loss_all.backward()
        optimizer.step()
        # Output training stats
        if i % 100 == 0:
            print('[%d/%d][%d/%d]\tlr_rate: %0.4f \tLoss_total: %.4f\tLoss_ssim: %.4f\tLoss_mse: %.4f\t' %
                  (epoch, args.num_epochs, i, len(dataloader),optimizer.state_dict()['param_groups'][0]['lr'],loss_all.mean().item(), ssim.mean().item(),mse.mean().item()))
            # print(optimizer.state_dict()['param_groups'][0]['lr'])
    if epoch % 5 == 0:
        torch.save(netG.state_dict(), 'snapshot/' + str(epoch) + ".pkl")

