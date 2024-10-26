import os
from models import disjoint_augment_image_pair#,H_estimator
# from H_model_mini import H_estimator
from H_model import H_estimator,H_joint
# from H_model_detone import H_estimator
import torch.nn as nn
import numpy as np
import torch
import cv2
from dataset import Image_stitch
import time
import argparse

parser = argparse.ArgumentParser("GQPR")
parser.add_argument('--is_visualize', type = bool, default = True, help = 'whether visualize or not')
parser.add_argument('--dis_batch', type = int, default = 50, help = 'frequency for dslplay results')
parser.add_argument('--device', type = str, default = 'cuda:0', help = 'device for training')
parser.add_argument('--mode', type = str, default = 'supervise', help ='optinal in [supervise,unsupervise]')
parser.add_argument('--num_epochs', type = int, default = 100, help = 'num of epochs')
parser.add_argument('--batch_size', type = int, default = 4, help = 'batch size')
parser.add_argument('--learning_rate', type = float, default = 0.0001, help = 'learning rate')
parser.add_argument('--data_root', type = str, default = 'D:/training_roadandmsisandcoco_mix', help = 'path for training dataset')
parser.add_argument('--netR_path', type = str, default = None, help = 'path for snapshot of netR')
parser.add_argument('--netH_path', type = str, default = None, help = 'path for snapshot of netH')
args = parser.parse_args()

if args.device != 'cpu':
    use_cuda=True
else:
    use_cuda=False

def main():
    data=Image_stitch(ir1_path=os.path.join(args.data_root, 'ir_input1'),\
                    ir2_path=os.path.join(args.data_root, 'ir_input2'),\
                    vis1_path=os.path.join(args.data_root, 'vis_input1'),\
                    vis2_path=os.path.join(args.data_root, 'vis_input2'),\
                    gt_path=os.path.join(args.data_root, 'y_shift'))

    dataloader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,shuffle=True,num_workers=0,pin_memory=True)
    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()
    netR = H_estimator(batch_size = args.batch_size, device = args.device, is_training = True)
    netH = H_joint(batch_size = args.batch_size, device = args.device, is_training = True)
    if args.netR_path is not None:
        netR.load_state_dict(torch.load(args.netR_path, map_location='cpu'))
    if args.netH_path is not None:
        netH.load_state_dict(torch.load(args.netH_path, map_location='cpu'))
    if use_cuda:
        l1loss = l1loss.to(args.device)
        l2loss = l2loss.to(args.device)
        netR = netR.to(args.device)
        netH = netH.to(args.device)
    optimizerR = torch.optim.Adam(netR.parameters(), lr = args.learning_rate, betas=(0.5, 0.999), weight_decay=0.0001)
    optimizerH = torch.optim.Adam(netH.parameters(), lr = args.learning_rate, betas=(0.5, 0.999), weight_decay=0.0001)
    save_folder = 'snapshot'
    # define dataset
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    loss_all_batch = 0
    if args.mode == 'unsupervise':
        l1_1_batch = 0
        l1_2_batch = 0
        l1_3_batch = 0
        l1_out_batch = 0
    elif args.mode == 'supervise':
        l2_gt_1_batch = 0
        l2_gt_2_batch = 0
        l2_gt_3_batch = 0
        l2_gt_out_batch = 0
    netR.train()
    netH.train()
    for epoch in range(0, args.num_epochs+1):
        for i,(ir_input1,ir_input2,vis_input1,vis_input2,gt) in enumerate(dataloader):
            train_ir_inputs_aug = disjoint_augment_image_pair(ir_input1,ir_input2)
            train_vis_inputs_aug = disjoint_augment_image_pair(vis_input1,vis_input2)
            if use_cuda:
                ir_input1 = ir_input1.to(args.device)
                train_ir_inputs_aug = train_ir_inputs_aug.to(args.device)
                train_vis_inputs_aug = train_vis_inputs_aug.to(args.device)
                ir_input2 = ir_input2.to(args.device)
                vis_input1=vis_input1.to(args.device)
                vis_input2 = vis_input2.to(args.device)
                gt = gt.to(args.device)
            
            train_ir_inputs = torch.cat((ir_input1,ir_input2), 3)
            train_vis_inputs = torch.cat((vis_input1,vis_input2), 3)

            ir_off1, ir_off2, ir_off3, ir_warp1, ir_warp2,  vis_off1, vis_off2, vis_off3, vis_warp1, vis_warp2= netR(train_ir_inputs_aug,train_ir_inputs, train_vis_inputs_aug,train_vis_inputs,gt=gt)
            ir_shift = ir_off1 + ir_off2 + ir_off3
            vis_shift = vis_off1 + vis_off2 + vis_off3
            offset_out, one_warp_out, ir_warp2_out, vis_warp2_out = netH(ir_shift, vis_shift, train_ir_inputs, train_vis_inputs)
            ir_warp1_out = train_ir_inputs[...,0:3].permute(0,3,1,2)*one_warp_out
            vis_warp1_out = train_vis_inputs[...,0:3].permute(0,3,1,2)*one_warp_out
            ##### unsupervise training  (optional)##################################################################################
            if args.mode == 'unsupervise':
                ir_l1_1 = 16 * l1loss(ir_warp1[:,0:3,...],  ir_warp2[:,0:3,...])
                ir_l1_2 = 4 * l1loss(ir_warp1[:,3:6,...],  ir_warp2[:,3:6,...])
                ir_l1_3 = 1 * l1loss(ir_warp1[:,6:9,...],  ir_warp2[:,6:9,...])
                ir_l1_4 = 1 * l1loss(ir_warp1_out, ir_warp2_out)
                vis_l1_1 = 16 * l1loss(vis_warp1[:,0:3,...],  vis_warp2[:,0:3,...])
                vis_l1_2 = 4 * l1loss(vis_warp1[:,3:6,...],  vis_warp2[:,3:6,...])
                vis_l1_3 = 1 * l1loss(vis_warp1[:,6:9,...],  vis_warp2[:,6:9,...])
                vis_l1_4 = 1 * l1loss(vis_warp1_out, vis_warp2_out)
                loss_unsupervise = ir_l1_1 + ir_l1_2 + ir_l1_3 + ir_l1_4 + vis_l1_1 + vis_l1_2 + vis_l1_3 + vis_l1_4
                loss_all = loss_unsupervise
                l1_1_batch  += (ir_l1_1.item() + vis_l1_1.item())
                l1_2_batch  += (ir_l1_2.item() + vis_l1_2.item())
                l1_3_batch  += (ir_l1_3.item() + vis_l1_3.item())
                l1_out_batch  += (ir_l1_4.item() + vis_l1_4.item())
            ##### supervise training (optional)##################################################################################
            elif args.mode == 'supervise':
                ir_l2_gt1 = 0.02  * l2loss(gt,ir_off1)
                ir_l2_gt2 = 0.01  * l2loss(gt,ir_off1+ir_off2)
                ir_l2_gt3 = 0.005 * l2loss(gt,ir_off1+ir_off2+ir_off3)
                vis_l2_gt1 = 0.02 * l2loss(gt,vis_off1)
                vis_l2_gt2 = 0.01 * l2loss(gt,vis_off1+vis_off2)
                vis_l2_gt3 = 0.005* l2loss(gt,vis_off1+vis_off2+vis_off3)
                out_l2_gt = 0.02  * l2loss(gt,offset_out)
                loss_supervise = ir_l2_gt1 + ir_l2_gt2 + ir_l2_gt3 + vis_l2_gt1 + vis_l2_gt2 + vis_l2_gt3 + out_l2_gt
                loss_all = loss_supervise
                l2_gt_1_batch += (ir_l2_gt1.item() + vis_l2_gt1.item())
                l2_gt_2_batch += (ir_l2_gt2.item() + vis_l2_gt2.item())
                l2_gt_3_batch += (ir_l2_gt3.item() + vis_l2_gt3.item())
                l2_gt_out_batch += out_l2_gt.item()
            loss_all_batch += loss_all.item()
            
            if i % args.dis_batch == 0 and i!=0:
                if args.mode=='unsupervise':
                    print('[%d/%d][%d/%d] lr_rate: %0.4f  Loss_total: %.3f  Loss_1: %.3f  Loss_2: %.3f  Loss_3: %.3f  Loss_H: %.3f\t' % 
                        (epoch, args.num_epochs, i, len(dataloader),optimizerH.state_dict()['param_groups'][0]['lr'],loss_all_batch/args.dis_batch, \
                            l1_1_batch/args.dis_batch, l1_2_batch/args.dis_batch, l1_3_batch/args.dis_batch, l1_out_batch/args.dis_batch))
                    l1_1_batch = 0
                    l1_2_batch = 0
                    l1_3_batch = 0
                    l2_gt_batch = 0
                elif args.mode=='supervise':
                    print('[%d/%d][%d/%d] lr_rate: %0.4f  Loss_total: %.3f  Loss_1: %.3f  Loss_2: %.3f  Loss_3: %.3f  Loss_H: %.3f\t' % 
                        (epoch, args.num_epochs, i, len(dataloader),optimizerH.state_dict()['param_groups'][0]['lr'],loss_all_batch/args.dis_batch, \
                            l2_gt_1_batch/args.dis_batch, l2_gt_2_batch/args.dis_batch, l2_gt_3_batch/args.dis_batch, l2_gt_out_batch/args.dis_batch))
                    l2_gt_1_batch = 0
                    l2_gt_2_batch = 0
                    l2_gt_3_batch = 0
                    l2_gt_out_batch = 0
                loss_all_batch=0
                
            if args.is_visualize and i % args.dis_batch == 0:
                ir_warp1s = torch.cat((train_ir_inputs[0][...,0:3].permute(2,0,1), ir_warp1[0,0:3], ir_warp1[0,3:6], ir_warp1[0,6:9], ir_warp1_out[0], ir_warp1[0,9:]),1)
                ir_warp2s = torch.cat((train_ir_inputs[0][...,3:6].permute(2,0,1), ir_warp2[0,0:3], ir_warp2[0,3:6], ir_warp2[0,6:9], ir_warp2_out[0], ir_warp2[0,9:]),1)
                vis_warp1s = torch.cat((train_vis_inputs[0][...,0:3].permute(2,0,1), vis_warp1[0,0:3], vis_warp1[0,3:6], vis_warp1[0,6:9], vis_warp1_out[0], vis_warp1[0,9:]),1)
                vis_warp2s = torch.cat((train_vis_inputs[0][...,3:6].permute(2,0,1), vis_warp2[0,0:3], vis_warp2[0,3:6], vis_warp2[0,6:9], vis_warp2_out[0], vis_warp2[0,9:]),1)
                visualize = torch.cat((ir_warp1s,ir_warp2s,vis_warp1s,vis_warp2s),2).permute(1,2,0).detach().cpu().numpy()*255
                cv2.imwrite('visual.png', visualize)
            loss_all.backward()
            optimizerR.step()
            optimizerH.step()
            optimizerR.zero_grad()
            optimizerH.zero_grad()
        if epoch % 4 == 0:
            torch.save(netR.state_dict(), save_folder+ '/' + str(epoch) + "_R.pkl")
            torch.save(netH.state_dict(), save_folder+ '/' + str(epoch) + "_H.pkl")


if __name__ == '__main__':
    main()