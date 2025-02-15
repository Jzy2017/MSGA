import os
from H_model import H_estimator,H_joint_out
import torch.nn as nn
import numpy as np
import torch
import cv2
from dataset import Image_stitch_test
import time
import argparse
parser = argparse.ArgumentParser("GQPR")
parser.add_argument('--device', type = str, default = 'cuda:0', help = 'device for training')
parser.add_argument('--batch_size', type = int, default = 1, help = 'batch size')
parser.add_argument('--data_root', type = str, default = '../example', help = 'path for test dataset')
parser.add_argument('--netR_path', type = str, default = None, help = 'path for snapshot of netR')
parser.add_argument('--netH_path', type = str, default = None, help = 'path for snapshot of netH')
args = parser.parse_args()

if args.device!='cpu':
    use_cuda=True
else:
    use_cuda=False

out_folder = os.path.join('../output/',args.data_root.split('/')[-1])
data = Image_stitch_test(ir1_path=os.path.join(args.data_root,'ir_input1'),\
                  ir2_path=os.path.join(args.data_root,'ir_input2'),\
                  vis1_path=os.path.join(args.data_root,'vis_input1'),\
                  vis2_path=os.path.join(args.data_root,'vis_input2'))
dataloader = torch.utils.data.DataLoader(data, batch_size= args.batch_size,shuffle=False,num_workers=0,pin_memory=True)

netR = H_estimator(batch_size=args.batch_size,device=args.device,is_training=False)
netH = H_joint_out(batch_size=args.batch_size,device=args.device,is_training=False)
if args.netR_path is not None:
    netR.load_state_dict(torch.load(args.netR_path,map_location='cpu'))
if args.netH_path is not None:
    netH.load_state_dict(torch.load(args.netH_path,map_location='cpu'))
if use_cuda:
    netR = netR.to(args.device)
    netH = netH.to(args.device)

# define dataset
if not os.path.exists(os.path.join(out_folder,'ir_warp1')):
    os.makedirs(os.path.join(out_folder,'ir_warp1'))
if not os.path.exists(os.path.join(out_folder,'ir_warp2')):
    os.makedirs(os.path.join(out_folder,'ir_warp2'))
if not os.path.exists(os.path.join(out_folder,'vis_warp1')):
    os.makedirs(os.path.join(out_folder,'vis_warp1'))
if not os.path.exists(os.path.join(out_folder,'vis_warp2')):
    os.makedirs(os.path.join(out_folder,'vis_warp2'))

netR.eval()
netH.eval()


for i,(ir_input1,ir_input2,vis_input1,vis_input2,size,name) in enumerate(dataloader):
    if use_cuda:
        ir_input1=ir_input1.to(args.device)
        ir_input2=ir_input2.to(args.device)
        vis_input1=vis_input1.to(args.device)
        vis_input2=vis_input2.to(args.device)
        size=size.to(args.device)
    train_ir_inputs=torch.cat((ir_input1,ir_input2), 3)
    train_vis_inputs=torch.cat((vis_input1,vis_input2), 3)
    start=time.time()
    with torch.no_grad():
        ir_off, vis_off = netR(None,train_ir_inputs,None,train_vis_inputs,gt=None,is_test=True)
        warps_ir, warps_vis = netH(ir_off,vis_off, size,train_ir_inputs,train_vis_inputs)
       
    end=time.time()  
    ir_warp1_H = warps_ir[0][0:3].permute(1,2,0).detach().cpu().numpy()*255
    ir_warp2_H = warps_ir[0][3:6].permute(1,2,0).detach().cpu().numpy()*255
    vis_warp1_H = warps_vis[0][0:3].permute(1,2,0).detach().cpu().numpy()*255
    vis_warp2_H = warps_vis[0][3:6].permute(1,2,0).detach().cpu().numpy()*255
    cv2.imwrite(os.path.join(out_folder,'ir_warp1',name[0]),cv2.cvtColor(ir_warp1_H,cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_folder,'ir_warp2',name[0]),cv2.cvtColor(ir_warp2_H,cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_folder,'vis_warp1',name[0]),cv2.cvtColor(vis_warp1_H,cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_folder,'vis_warp2',name[0]),cv2.cvtColor(vis_warp2_H,cv2.COLOR_RGB2BGR))

