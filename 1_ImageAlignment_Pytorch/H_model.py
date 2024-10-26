import numpy as np
from tensorDLT_function import solve_DLT
from spatial_transform import Transform
from output_spatial_transform import Transform_output
import torch
from output_tensorDLT import output_solve_DLT
import torch.nn as nn
from attention import Attention1D
from position import PositionEmbeddingSine

class feature_extractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.batch_norm4 = nn.BatchNorm2d(64)
        self.conv1=torch.nn.Sequential( nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(True),
                                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(True)
                                        )
        self.maxpool1 = torch.nn.MaxPool2d((2,2), stride=2)
        self.conv2=torch.nn.Sequential( nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(True),
                                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(True))
        self.maxpool2 = torch.nn.MaxPool2d((2,2), stride=2)
        self.conv3=torch.nn.Sequential( nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(True),
                                        nn.BatchNorm2d(128),
                                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                        nn.ReLU(True))
        self.maxpool3 = torch.nn.MaxPool2d((2,2), stride=2)
        self.conv4 = torch.nn.Sequential( nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(True),
                                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(True))
    def forward(self, input):
        conv1 = self.conv1(input)
        conv1m = self.maxpool1(conv1)
        conv2 = self.conv2(conv1m)
        conv2m = self.maxpool1(conv2)
        conv3 = self.conv3(conv2m)
        conv3m = self.maxpool1(conv3)
        conv4 = self.conv4(conv3m)
        return conv1,conv2,conv3,conv4


def vertical_cost_volume(c1, warp, search_range,pad,lrelu):
    """Build vertical cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: Level of the feature pyramid of Image1
        warp: Warped level of the feature pyramid of image22
        search_range: Search range (maximum displacement)
    """
    padded_lvl = pad(warp)
    _,c ,h, w = c1.shape
    max_offset = search_range * 2 + 1
    cost_vol = []
    for y in range(0, max_offset):
        slice=padded_lvl[:, :,y:y+h, :]
        cost = torch.mean(c1 * slice, axis=1, keepdims=True)
        cost_vol.append(cost)
    cost_vol = torch.cat(cost_vol, axis=1)
    cost_vol = lrelu(cost_vol)
    return cost_vol
def horizontal_cost_volume(c1, warp, search_range,pad,lrelu):
    """Build horizontal cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: Level of the feature pyramid of Image1
        warp: Warped level of the feature pyramid of image22
        search_range: Search range (maximum displacement)
    """

    padded_lvl = pad(warp)
    _,c ,h, w = c1.shape
    max_offset = search_range * 2 + 1
    cost_vol = []
    for x in range(0, max_offset):
        slice=padded_lvl[:, :,:, x:x+w]
        cost = torch.mean(c1 * slice, axis=1, keepdims=True)
        cost_vol.append(cost)
    cost_vol = torch.cat(cost_vol, axis=1)
    cost_vol = lrelu(cost_vol)
    return cost_vol
		
class RegressionNet(torch.nn.Module):
    def __init__(self,is_training, search_range, warped=False):
        super().__init__()
        self.search_range=search_range
        self.warped=warped
        self.keep_prob = 0.5 if is_training==True else 1.0
        self.feature = []
        if self.search_range==16:
            self.in_channel=128
            self.stride=[1,1,1]
            self.out=[128,128,128]
            self.fully=1024
        elif self.search_range==8:
            self.in_channel=128
            self.stride=[1,1,2]
            self.out=[128,128,128]
            self.fully=512
        elif self.search_range==4:
            self.in_channel=64
            self.stride=[1,2,2]
            self.out=[128,128,128]
            self.fully=256
        
        self.vertical_pad=nn.ConstantPad2d(value = 0,padding = [0, 0, search_range, search_range])
        self.horizontal_pad=nn.ConstantPad2d(value = 0,padding = [search_range, search_range, 0, 0])
        self.lrelu=nn.LeakyReLU(inplace = True)

        self.conv1 =torch.nn.Sequential( nn.Conv2d(in_channels = ((2*self.search_range)+1)*2, out_channels=self.out[0], kernel_size=3, stride=self.stride[0],padding=1),
                                        nn.BatchNorm2d(self.out[0]),
                                        nn.ReLU(True))
        self.conv2 = torch.nn.Sequential( nn.Conv2d(in_channels = self.out[0], out_channels=self.out[1], kernel_size=3, stride=self.stride[1],padding=1),
                                        nn.BatchNorm2d(self.out[1]),
                                        nn.ReLU(True))
        self.conv3 = torch.nn.Sequential( nn.Conv2d(in_channels = self.out[1], out_channels=self.out[2], kernel_size=3, stride=self.stride[2],padding=1),
                                        nn.BatchNorm2d(self.out[2]),
                                        nn.ReLU(True))
        self.getoffset = torch.nn.Sequential(torch.nn.Linear(in_features = 256*self.out[2], out_features = self.fully),
                                            nn.ReLU(True),
                                            nn.Dropout(p = self.keep_prob),
                                            torch.nn.Linear(in_features = self.fully, out_features = 8))
        self.horizontal_attention = Attention1D(self.in_channel,
                                  y_attention = False,
                                  double_cross_attn = True,
                                  )
        self.vertical_attention = Attention1D(self.in_channel,
                                  y_attention=True,
                                  double_cross_attn=True,
                                  )

    def forward(self, feature1, feature2):
        # print(feature1.shape)
        pos_channels = feature1.shape[1] // 2
        pos_enc = PositionEmbeddingSine(pos_channels)
        position = pos_enc(feature1)  # [B, C, H, W]
        if not self.warped:
            verti_attn,attn_y = self.vertical_attention(nn.functional.normalize(feature1, dim=1, p=2), nn.functional.normalize(feature2,dim=1, p=2),position)
            hori_attn,attn_x = self.horizontal_attention(nn.functional.normalize(feature1, dim=1, p=2), nn.functional.normalize(feature2,dim=1, p=2),position)
        else:
            verti_attn,attn_y = self.vertical_attention(nn.functional.normalize(feature1, dim=1, p=2), feature2, position)
            hori_attn,attn_x = self.horizontal_attention(nn.functional.normalize(feature1, dim=1, p=2), feature2, position)
        verti_correlation = vertical_cost_volume(nn.functional.normalize(feature1, dim=1, p=2), hori_attn, self.search_range,self.vertical_pad,self.lrelu) 
        hori_correlation = horizontal_cost_volume(nn.functional.normalize(feature1, dim=1, p=2), verti_attn, self.search_range,self.horizontal_pad,self.lrelu)  
        correlation=torch.cat((hori_correlation,verti_correlation),1)
        conv1 = self.conv1(correlation)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv3_flatten = conv3.contiguous().view(conv3.shape[0],-1)
        offset=self.getoffset(conv3_flatten)
        return offset

class H_estimator(torch.nn.Module):
    def __init__(self, batch_size, device, is_training=1):
        super().__init__()
        self.device=device
        self.feature_ir =  feature_extractor()#.cuda()
        self.feature_vis = feature_extractor()#.cuda()
        self.Rnet1 = RegressionNet(is_training, 16, warped=False)#.cuda()
        self.Rnet2 = RegressionNet(is_training, 8, warped=True)#.cuda()
        self.Rnet3 = RegressionNet(is_training, 4, warped=True)#.cuda()
        for m in self.Rnet1.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        for m in self.Rnet2.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        for m in self.Rnet3.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        self.M_tile_inv_128, self.M_tile_128 = self.to_transform_H(128, batch_size)
        self.M_tile_inv_32, self.M_tile_32 = self.to_transform_H(32, batch_size)
        self.M_tile_inv_64, self.M_tile_64 = self.to_transform_H(64, batch_size)
        self.transform32  = Transform(32, 32,self.device,batch_size).to(self.device)
        self.transform64  = Transform(64, 64,self.device,batch_size).to(self.device)
        self.transform128 = Transform(128,128,self.device,batch_size).to(self.device)

    def to_transform_H(self, patch_size, batch_size):            
        M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
                    [0., patch_size / 2.0, patch_size / 2.0],
                    [0., 0., 1.]]).astype(np.float32)
        M_tensor = torch.from_numpy(M)
        M_tile = torch.unsqueeze(M_tensor, 0).repeat( [batch_size, 1, 1])
        M_inv = np.linalg.inv(M)
        M_tensor_inv = torch.from_numpy(M_inv)
        M_tile_inv = torch.unsqueeze(M_tensor_inv, 0).repeat([batch_size, 1, 1])
        M_tile_inv=M_tile_inv.to(self.device)
        M_tile=M_tile.to(self.device)
        return M_tile_inv, M_tile

    def forward(self, inputs_ir_aug, inputs_ir, inputs_vis_aug, inputs_vis, gt=None,is_test=False,patch_size=128.):
        
        batch_size = inputs_ir.shape[0]
        if is_test == True:
            ir_input1 = inputs_ir[...,0:3].permute(0,3,1,2)
            ir_input2 = inputs_ir[...,3:6].permute(0,3,1,2)
            ir_input1 = torch.nn.functional.interpolate(ir_input1, [128,128])
            ir_input2 = torch.nn.functional.interpolate(ir_input2, [128,128])   
            
            vis_input1 = inputs_vis[...,0:3].permute(0,3,1,2)
            vis_input2 = inputs_vis[...,3:6].permute(0,3,1,2)
            vis_input1 = torch.nn.functional.interpolate(vis_input1, [128,128])
            vis_input2 = torch.nn.functional.interpolate(vis_input2, [128,128])      
        else:
            ir_input1 = inputs_ir_aug[...,0:3].permute(0,3,1,2)
            ir_input2 = inputs_ir_aug[...,3:6].permute(0,3,1,2)
            ir_input1 = torch.nn.functional.interpolate(ir_input1, [128,128])
            ir_input2 = torch.nn.functional.interpolate(ir_input2, [128,128])   
            
            vis_input1 = inputs_vis_aug[...,0:3].permute(0,3,1,2)
            vis_input2 = inputs_vis_aug[...,3:6].permute(0,3,1,2)
            vis_input1 = torch.nn.functional.interpolate(vis_input1, [128,128])
            vis_input2 = torch.nn.functional.interpolate(vis_input2, [128,128])
        ##############################  feature_extractor ##############################       
        ir_feature1 = self.feature_ir(ir_input1.to(torch.float32))
        ir_feature2 = self.feature_ir(ir_input2.to(torch.float32))
        vis_feature1 = self.feature_vis(vis_input1.to(torch.float32))
        vis_feature2 = self.feature_vis(vis_input2.to(torch.float32))
        #########################################  ir  ###########################################
        ##############################  Regression Net 1 ##############################
        ir_net1_f = self.Rnet1(ir_feature1[-1], ir_feature2[-1])
        ir_net1_f = torch.unsqueeze(ir_net1_f, 2)#*128
        # H1 = self.DLT(net1_f/4., 32.)
        ir_H1 = solve_DLT(ir_net1_f/4., 32.)
        ir_H1 = torch.matmul(torch.matmul(self.M_tile_inv_32, ir_H1), self.M_tile_32)
        ir_feature2_warp = self.transform32(nn.functional.normalize(ir_feature2[-2], dim=1, p=2), ir_H1)
        # ##############################  Regression Net 2 ##############################
        ir_net2_f = self.Rnet2(ir_feature1[-2], ir_feature2_warp)
        ir_net2_f = torch.unsqueeze(ir_net2_f, 2)#*128
        # H2 = self.DLT((net1_f+net2_f)/2., 64.)
        ir_H2 = solve_DLT((ir_net1_f + ir_net2_f)/2., 64.)
        ir_H2 = torch.matmul(torch.matmul(self.M_tile_inv_64, ir_H2), self.M_tile_64)
        ir_feature3_warp = self.transform64(nn.functional.normalize(ir_feature2[-3], dim=1, p=2), ir_H2)
        # ##############################  Regression Net 3 ##############################
        ir_net3_f = self.Rnet3(ir_feature1[-3], ir_feature3_warp)
        ir_net3_f = torch.unsqueeze(ir_net3_f, 2)#*128
        
        ################################ vis #####################################################
        ##############################  Regression Net 1 ##############################
        vis_net1_f = self.Rnet1(vis_feature1[-1], vis_feature2[-1])
        vis_net1_f = torch.unsqueeze(vis_net1_f, 2)#*128
        # H1 = self.DLT(net1_f/4., 32.)
        vis_H1 = solve_DLT(vis_net1_f/4., 32.)
        vis_H1 = torch.matmul(torch.matmul(self.M_tile_inv_32, vis_H1), self.M_tile_32)
        vis_feature2_warp = self.transform32(nn.functional.normalize(vis_feature2[-2], dim=1, p=2), vis_H1)
        # ##############################  Regression Net 2 ##############################
        vis_net2_f = self.Rnet2(vis_feature1[-2], vis_feature2_warp)
        vis_net2_f = torch.unsqueeze(vis_net2_f, 2)#*128
        # H2 = self.DLT((net1_f+net2_f)/2., 64.)
        vis_H2 = solve_DLT((vis_net1_f + vis_net2_f)/2., 64.)
        vis_H2 = torch.matmul(torch.matmul(self.M_tile_inv_64, vis_H2), self.M_tile_64)
        vis_feature3_warp = self.transform64(nn.functional.normalize(vis_feature2[-3], dim=1, p=2), vis_H2)
        # ##############################  Regression Net 3 ##############################
        vis_net3_f = self.Rnet3(vis_feature1[-3], vis_feature3_warp)
        vis_net3_f = torch.unsqueeze(vis_net3_f, 2)#*128
        if is_test:
            return ir_net1_f + ir_net2_f + ir_net3_f,  vis_net1_f + vis_net2_f + vis_net3_f
        ###############################################################################
        H_gt = solve_DLT(gt, 128)
        Hgt_mat = torch.matmul(torch.matmul(self.M_tile_inv_128, H_gt), self.M_tile_128)

        ir_H1 = solve_DLT(ir_net1_f , 128)
        ir_H2 = solve_DLT(ir_net1_f + ir_net2_f, 128)
        ir_H3 = solve_DLT(ir_net1_f + ir_net2_f + ir_net3_f, 128)
        ir_H1_mat = torch.matmul(torch.matmul(self.M_tile_inv_128, ir_H1), self.M_tile_128)
        ir_H2_mat = torch.matmul(torch.matmul(self.M_tile_inv_128, ir_H2), self.M_tile_128)
        ir_H3_mat = torch.matmul(torch.matmul(self.M_tile_inv_128, ir_H3), self.M_tile_128)
        ir_image2_tensor = inputs_ir[..., 3:6].permute(0,3,1,2)
        ir_image1_tensor = inputs_ir[..., 0:3].permute(0,3,1,2)
        ir_warp2_H1 = self.transform128(ir_image2_tensor, ir_H1_mat)
        ir_warp2_H2 = self.transform128(ir_image2_tensor, ir_H2_mat)
        ir_warp2_H3 = self.transform128(ir_image2_tensor, ir_H3_mat)
        ir_warp2_gt = self.transform128(ir_image2_tensor, Hgt_mat)
        one = torch.ones_like(ir_image2_tensor)
        ir_warp1_H1 = self.transform128(one, ir_H1_mat) * ir_image1_tensor
        ir_warp1_H2 = self.transform128(one, ir_H2_mat) * ir_image1_tensor
        ir_warp1_H3 = self.transform128(one, ir_H3_mat) * ir_image1_tensor
        ir_warp1_gt = self.transform128(one, Hgt_mat) * ir_image1_tensor

        vis_H1 = solve_DLT(vis_net1_f , 128)
        vis_H2 = solve_DLT(vis_net1_f + vis_net2_f, 128)
        vis_H3 = solve_DLT(vis_net1_f + vis_net2_f + vis_net3_f, 128)
        vis_H1_mat = torch.matmul(torch.matmul(self.M_tile_inv_128, vis_H1), self.M_tile_128)
        vis_H2_mat = torch.matmul(torch.matmul(self.M_tile_inv_128, vis_H2), self.M_tile_128)
        vis_H3_mat = torch.matmul(torch.matmul(self.M_tile_inv_128, vis_H3), self.M_tile_128)
        vis_image2_tensor = inputs_vis[..., 3:6].permute(0,3,1,2)
        vis_image1_tensor = inputs_vis[..., 0:3].permute(0,3,1,2)
        vis_warp2_H1 = self.transform128(vis_image2_tensor, vis_H1_mat)
        vis_warp2_H2 = self.transform128(vis_image2_tensor, vis_H2_mat)
        vis_warp2_H3 = self.transform128(vis_image2_tensor, vis_H3_mat)
        vis_warp2_gt = self.transform128(vis_image2_tensor, Hgt_mat)
        one = torch.ones_like(vis_image2_tensor)
        vis_warp1_H1 = self.transform128(one, vis_H1_mat)*vis_image1_tensor
        vis_warp1_H2 = self.transform128(one, vis_H2_mat)*vis_image1_tensor
        vis_warp1_H3 = self.transform128(one, vis_H3_mat)*vis_image1_tensor
        vis_warp1_gt = self.transform128(one, Hgt_mat)*vis_image1_tensor
        ir_warp2=torch.cat((ir_warp2_H1,ir_warp2_H2,ir_warp2_H3,ir_warp2_gt),1)
        ir_warp1=torch.cat((ir_warp1_H1,ir_warp1_H2,ir_warp1_H3,ir_warp1_gt),1)
        vis_warp2=torch.cat((vis_warp2_H1,vis_warp2_H2,vis_warp2_H3,vis_warp2_gt),1)
        vis_warp1=torch.cat((vis_warp1_H1,vis_warp1_H2,vis_warp1_H3,vis_warp1_gt),1)
        return ir_net1_f, ir_net2_f, ir_net3_f, ir_warp1, ir_warp2, vis_net1_f, vis_net2_f, vis_net3_f, vis_warp1, vis_warp2
        # return net1_f, net1_f, net1_f, warp2_H1, warp2_H1, warp2_H1, one_warp_H1, one_warp_H1, one_warp_H1,warp2_gt, one_warp_gt

class H_joint(torch.nn.Module):
    def __init__(self, batch_size, device, is_training=1):
        super().__init__()
        self.device=device
        self.keep_prob = 1.0
        self.getoffset = torch.nn.Sequential(torch.nn.Linear(in_features = 16, out_features = 64),
                                            # nn.ReLU(True),
                                            torch.nn.Linear(in_features = 64, out_features = 8))
        self.M_tile_inv_128, self.M_tile_128 = self.to_transform_H(128, batch_size)
        self.transform128 = Transform(128,128,self.device,batch_size).to(self.device)


    def to_transform_H(self, patch_size, batch_size):            
        M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
                    [0., patch_size / 2.0, patch_size / 2.0],
                    [0., 0., 1.]]).astype(np.float32)
        M_tensor = torch.from_numpy(M)
        M_tile = torch.unsqueeze(M_tensor, 0).repeat( [batch_size, 1, 1])
        M_inv = np.linalg.inv(M)
        M_tensor_inv = torch.from_numpy(M_inv)
        M_tile_inv = torch.unsqueeze(M_tensor_inv, 0).repeat([batch_size, 1, 1])
        M_tile_inv=M_tile_inv.to(self.device)
        M_tile=M_tile.to(self.device)
        return M_tile_inv, M_tile
        
    def forward(self, offset1, offset2,irs,viss):

        # fusion=torch.cat([offset1,offset2],2)
        fusion=torch.cat([offset1,offset2],1)
        fusion = fusion.contiguous().view(fusion.shape[0],-1)
        offset_out = self.getoffset(fusion)
        offset_out = torch.unsqueeze(offset_out, 2)#*128
        H = solve_DLT(offset_out, 128)
        H_mat = torch.matmul(torch.matmul(self.M_tile_inv_128, H), self.M_tile_128)
        ir2 = irs[..., 3:6].permute(0,3,1,2)
        vis2 = viss[..., 3:6].permute(0,3,1,2)
        ir_warp2 = self.transform128(ir2, H_mat)
        vis_warp2 = self.transform128(vis2, H_mat)
        one = torch.ones_like(vis2)
        one_warp = self.transform128(one, H_mat)
        return offset_out,one_warp,ir_warp2,vis_warp2

class H_joint_out(torch.nn.Module):
    def __init__(self, batch_size, device, is_training=1):
        super().__init__()
        self.device = device
        self.keep_prob = 1.0
        self.getoffset = torch.nn.Sequential(torch.nn.Linear(in_features = 16, out_features = 64),
                                            # nn.ReLU(True),
                                            torch.nn.Linear(in_features = 64, out_features = 8))
        self.transform_output=Transform_output()

        
    def forward(self, offset1, offset2,size, irs, viss):
        fusion=torch.cat([offset1,offset2],1)
        fusion = fusion.contiguous().view(fusion.shape[0],-1)
        offset_out=self.getoffset(fusion)
        offset_out = torch.unsqueeze(offset_out, 2)#*128

        size_tmp = torch.cat([size,size,size,size],axis=1)/128.
        resized_shift = torch.mul(offset_out, size_tmp)
        H_mat = output_solve_DLT(resized_shift, size)  

        warps_ir = self.transform_output(irs.permute(0,3,1,2), H_mat,size,resized_shift)
        warps_vis = self.transform_output(viss.permute(0,3,1,2), H_mat,size,resized_shift)
        return warps_ir, warps_vis


