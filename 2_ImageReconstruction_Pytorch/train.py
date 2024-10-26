from __future__ import print_function
#%matplotlib inline
import os
import torch
from tqdm import tqdm
import torch.nn as nn
# import torch.nn.parallel
# import torch.utils.data
# import torchvision.datasets as dset
# import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.data import DataLoader
import torchvision
from dataset import Image
from visdom import Visdom
# from model import Reconstruction
import torch.nn.functional as F
#from generator import Generator
#sys.path.append(r"src/model")
import loss
import PIL
import model
# import utils
import time
import argparse

parser = argparse.ArgumentParser("GQPR")
parser.add_argument('--is_visualize', type = bool, default = True, help = 'whether visualize or not')
parser.add_argument('--dis_batch', type = int, default = 100, help = 'frequency for visualization')
parser.add_argument('--data_root', type = str, default = 'D:/TIP2021_fakeIJCAI/training_recon')
parser.add_argument('--vis_batch', type = int, default = 50, help = 'frequency for visualization')
parser.add_argument('--device', type = str, default = 'cuda:0', help = 'device for training')
parser.add_argument('--num_epochs', type = int, default = 200, help = 'num of epochs')
parser.add_argument('--batch_size', type = int, default = 4, help = 'batch size')
parser.add_argument('--learning_rate', type = float, default = 0.0001, help = 'learning rate')
parser.add_argument('--netG_path', type = str, default = None, help = 'path for snapshot of netR')
args = parser.parse_args()

train_augmentation = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224),interpolation=PIL.Image.NEAREST)])
if args.device != 'cpu':
    use_cuda=True
else:
    use_cuda=False

def main():
    data = Image(0, os.path.join(args.data_root,'warp1'), os.path.join(args.data_root,'warp2'),\
                    os.path.join(args.data_root,'mask1'),os.path.join(args.data_root,'mask2'), mode='train')
    dataloader = torch.utils.data.DataLoader(data, batch_size= args.batch_size,shuffle=True,num_workers=0)
    print(len(dataloader))
    per_loss = loss.PerceptualLoss()
    l1_loss = nn.L1Loss()
    ssim_loss=loss.SSIM()
    if use_cuda:
        l1_loss = l1_loss.cuda()
        per_loss = per_loss.cuda()
        ssim_loss = ssim_loss.cuda()
    netG = model.UNet(6,3)
    if args.netG_path is not None:
        netG.load_state_dict(torch.load(args.netG_path, map_location='cpu'))
    if use_cuda:
        netG = netG.to(args.device)
    optimizer = torch.optim.Adam(netG.parameters(), lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=0.0001)
    i = 0
    print("Starting Training Loop...")
    for epoch in range(1, args.num_epochs):
        for i,(warp1, warp2, mask1, mask2) in enumerate(dataloader):
            optimizer.state_dict()['param_groups'][0]['lr'] = optimizer.state_dict()['param_groups'][0]['lr']/2.
            netG.zero_grad()
            if use_cuda:
                warp1 = warp1.to(args.device)
                warp2 = warp2.to(args.device)
                mask1 = mask1.to(args.device)
                mask2 = mask2.to(args.device)
            warp1 = warp1.float()
            warp2 = warp2.float()
            mask1 = mask1.float()
            mask2 = mask2.float()

            inputs = torch.cat((warp1, warp2),1)
            outputs = netG(inputs)

            seam_mask1 = mask1*model.seammask_extraction(mask2,use_cuda)
            seam_mask2 = mask2*model.seammask_extraction(mask1,use_cuda)
            seam_loss1 = l1_loss(outputs*seam_mask1, warp1*seam_mask1)
            seam_loss2 = l1_loss(outputs*seam_mask2, warp2*seam_mask2)

            train_stitched1 = train_augmentation(outputs*mask1)  
            train_stitched2 = train_augmentation(outputs*mask2)  
            train_warp1 = train_augmentation(warp1*mask1)
            train_warp2 = train_augmentation(warp2*mask2)     

            content_loss1 = per_loss(train_stitched1,train_warp1 )
            content_loss2  = per_loss(train_stitched2,train_warp2 )
            
            seam_loss = (seam_loss1 + seam_loss2)*2
            content_loss = (content_loss1 + content_loss2)
            loss_all =  0.001 * content_loss + 1 * seam_loss + 100
            
            loss_all.backward()
            optimizer.step()
            if args.is_visualize and i % args.dis_batch == 0:
                visualize = outputs[0].permute(1,2,0).detach().cpu().numpy()*255
                cv2.imwrite('visual.png', cv2.cvtColor(visualize,cv2.COLOR_RGB2BGR))
            if i % args.dis_batch == 0:
                print('[%d/%d][%d/%d]\tlr_rate: %0.4f \tLoss_total: %.4f\tLoss_seam: %.4f\tLoss_content: %0.4f\t' %
                    (epoch, args.num_epochs, i, len(dataloader),optimizer.state_dict()['param_groups'][0]['lr'],\
                    loss_all.mean().item(), seam_loss.mean().item(), 0.001* content_loss.mean().item(), ))
        if epoch % 3 == 0:
            torch.save(netG.state_dict(), 'snapshot/' + str(epoch) + "_G.pkl")

if __name__ == '__main__':
    main()