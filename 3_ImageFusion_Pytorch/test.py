from __future__ import print_function
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import Image_fus
from model_fusion import myIFCNN
import cv2
import time

root = '../output/example'
os.makedirs(os.path.join(root, 'fusion'),exist_ok=True)
data = Image_fus(1, os.path.join(root,'recon_ir/'), os.path.join(root,'recon_vis/'))
dataloader = torch.utils.data.DataLoader(data, batch_size = 1,shuffle = False, num_workers = 0)

# Decide which device we want to run on
device='cuda:0'
if device!='cpu':
    use_cuda=True
else:
    use_cuda=False


netG=myIFCNN()
netG.load_state_dict(torch.load('snapshot/fusion.pkl', map_location='cpu'))
netG.eval()
if use_cuda:
    netG = netG.to(device)
# Lists to keep track of progress
img_list = []
# G_losses = []
iters = 0
aa = 1
i = 0
print("Starting Training Loop...")

for i, (ir, vis, name) in enumerate(dataloader):
    if use_cuda:
        vis = vis.to(device)
        ir = ir.to(device)
    vis = vis.float()
    ir = ir.float()
    with torch.no_grad():
        out = netG(vis,ir)
    if use_cuda:
        out = out.cpu()
    out = out[0].permute(1,2,0).numpy()*255
    out= cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    vis = vis[0].cpu().permute(1,2,0).numpy()*255
    vis_ycrbb = cv2.cvtColor(vis, cv2.COLOR_RGB2YCrCb)
    vis_ycrbb[...,0] = out
    vis_bgr = cv2.cvtColor(vis_ycrbb, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(os.path.join(root, 'fusion',name[0]), vis_bgr)
    # print(os.path.join(root, 'fusion/',name[0]))



