from __future__ import print_function
#%matplotlib inline
import os
import torch
import cv2
from dataset import Image_test
import model
import cv2
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_cuda=True
test_folder = '../output/example'
save_folder = test_folder
os.makedirs(os.path.join(test_folder, 'recon_ir'),exist_ok=True)
os.makedirs(os.path.join(test_folder, 'recon_vis'),exist_ok=True)
data_ir = Image_test(0, os.path.join(test_folder,'ir_warp1'), os.path.join(test_folder,'ir_warp2'))
data_vis = Image_test(0,  os.path.join(test_folder,'vis_warp1'), os.path.join(test_folder,'vis_warp2'))
dataloader_ir = torch.utils.data.DataLoader(data_ir, batch_size = 1,shuffle = False, num_workers = 0)
dataloader_vis = torch.utils.data.DataLoader(data_vis, batch_size = 1,shuffle = False, num_workers = 0)
print(len(dataloader_ir)+len(dataloader_vis))


netG=model.UNet(6,3)
netG.load_state_dict(torch.load('snapshot/reconstruction.pkl', map_location='cpu'))
if use_cuda:
    netG = netG.cuda()

img_list = []
iters = 0
aa = 1
i = 0
print("Starting Training Loop...")
# For each epoch
    # For each batch in the dataloader
for i, (warp1, warp2, name) in enumerate(dataloader_ir):
    if use_cuda:
        warp1 = warp1.cuda()
        warp2 = warp2.cuda()
    
    warp1=warp1.float()
    warp2=warp2.float()

    with torch.no_grad():
        inputs=torch.cat((warp1,warp2),1)
        outputs=netG(inputs)
    outputs=(outputs[0]).permute(1,2,0).detach().cpu().numpy()
    outputs=(outputs)*255.#/np.max(outputs)
    outputs=cv2.cvtColor(outputs, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(save_folder,'recon_ir/', name[0]), outputs)
for i, (warp1, warp2,name) in enumerate(dataloader_vis):
    if use_cuda:
        warp1 = warp1.cuda()
        warp2 = warp2.cuda()
    
    warp1=warp1.float()
    warp2=warp2.float()

    with torch.no_grad():
        inputs=torch.cat((warp1,warp2),1)
        outputs=netG(inputs)
    outputs=(outputs[0]).permute(1,2,0).detach().cpu().numpy()
    outputs=(outputs)*255.#/np.max(outputs)
    outputs=cv2.cvtColor(outputs, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(save_folder,'recon_vis/', name[0]), outputs)
