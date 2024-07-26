# import tensorflow.compat.v1 as tf
import numpy as np
import torch
import time
#######################################################
# Auxiliary matrices used to solve DLT
Aux_M1  = np.array([
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float64)


Aux_M2  = np.array([
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ]], dtype=np.float64)



Aux_M3  = np.array([
          [0],
          [1],
          [0],
          [1],
          [0],
          [1],
          [0],
          [1]], dtype=np.float64)



Aux_M4  = np.array([
          [-1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 ,-1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  ,-1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 ,-1 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ]], dtype=np.float64)


Aux_M5  = np.array([
          [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 ,-1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ]], dtype=np.float64)



Aux_M6  = np.array([
          [-1 ],
          [ 0 ],
          [-1 ],
          [ 0 ],
          [-1 ],
          [ 0 ],
          [-1 ],
          [ 0 ]], dtype=np.float64)


Aux_M71 = np.array([
          [0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float64)


Aux_M72 = np.array([
          [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [-1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 ,-1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  ,-1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 ,-1 , 0 ]], dtype=np.float64)



Aux_M8  = np.array([
          [0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 ,-1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ]], dtype=np.float64)


Aux_Mb  = np.array([
          [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , -1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float64)
########################################################

def output_solve_DLT(pre_4pt_shift, size):
    pts_1_tile = torch.tile(size, (1, 4, 1))
    tmp = torch.unsqueeze(torch.unsqueeze(torch.tensor([0., 0., 1., 0., 0., 1., 1., 1.], dtype=torch.float32, device=pre_4pt_shift.device), 0),-1)
    pts_1_tile = pts_1_tile*tmp
    batch_size = pre_4pt_shift.shape[0]
    pred_pts_2_tile = pre_4pt_shift + pts_1_tile
    orig_pt4 = pred_pts_2_tile
    pred_pt4 = pts_1_tile

    # pts_1_tile = torch.unsqueeze(pts_1_tile, 0).repeat([batch_size, 1, 1])
    # pts_1_tile=pts_1_tile.to(pre_4pt_shift.device)
    # pred_pts_2_tile = pre_4pt_shift+ pts_1_tile

    M1 = torch.tensor(Aux_M1)
    M1=M1.float()

    M1_tensor = torch.unsqueeze(M1, 0)
    M1_tile = M1_tensor.repeat([batch_size ,1 ,1])

    M2 = torch.tensor(Aux_M2)
    M2=M2.float()
    M2_tensor =torch.unsqueeze(M2, 0)
    M2_tile = M2_tensor.repeat([batch_size ,1 ,1])

    M3 = torch.tensor(Aux_M3)
    M3=M3.float()
    M3_tensor = torch.unsqueeze(M3, 0)
    M3_tile = M3_tensor.repeat([batch_size ,1 ,1])

    M4 = torch.tensor(Aux_M4)
    M4=M4.float()
    M4_tensor = torch.unsqueeze(M4, 0)
    M4_tile = M4_tensor.repeat([batch_size ,1 ,1])

    M5 = torch.tensor(Aux_M5)
    M5=M5.float()
    M5_tensor = torch.unsqueeze(M5, 0)
    M5_tile = M5_tensor.repeat([batch_size ,1 ,1])

    M6 = torch.tensor(Aux_M6)
    M6=M6.float()
    M6_tensor = torch.unsqueeze(M6, 0)
    M6_tile = M6_tensor.repeat([batch_size ,1 ,1])


    M71 = torch.tensor(Aux_M71)
    M71=M71.float()
    M71_tensor = torch.unsqueeze(M71, 0)
    M71_tile = M71_tensor.repeat([batch_size ,1 ,1])

    M72 = torch.tensor(Aux_M72)
    M72=M72.float()
    M72_tensor = torch.unsqueeze(M72, 0)
    M72_tile = M72_tensor.repeat([batch_size ,1 ,1])

    M8 = torch.tensor(Aux_M8)
    M8=M8.float()
    M8_tensor = torch.unsqueeze(M8, 0)
    M8_tile = M8_tensor.repeat([batch_size ,1 ,1])

    Mb = torch.tensor(Aux_Mb)
    Mb=Mb.float()
    Mb_tensor = torch.unsqueeze(Mb, 0)
    Mb_tile = Mb_tensor.repeat([batch_size ,1 ,1])

    # Form the equations Ax = b to compute H
    # Form A matrix
    M1_tile=M1_tile.to(pre_4pt_shift.device)
    M2_tile=M2_tile.to(pre_4pt_shift.device)
    M3_tile=M3_tile.to(pre_4pt_shift.device)
    M4_tile=M4_tile.to(pre_4pt_shift.device)
    M5_tile=M5_tile.to(pre_4pt_shift.device)
    M6_tile=M6_tile.to(pre_4pt_shift.device)
    M71_tile=M71_tile.to(pre_4pt_shift.device)
    M72_tile=M72_tile.to(pre_4pt_shift.device)
    M8_tile=M8_tile.to(pre_4pt_shift.device)
    A1 = torch.matmul(M1_tile, orig_pt4) # Column 1
    # print(orig_pt4)
    # time.sleep(100)
    A2 = torch.matmul(M2_tile, orig_pt4) # Column 2
    A3 = M3_tile                   # Column 3
    A4 = torch.matmul(M4_tile, orig_pt4) # Column 4
    A5 = torch.matmul(M5_tile, orig_pt4) # Column 5
    A6 = M6_tile                   # Column 6
    A7 = torch.matmul(M71_tile, pred_pt4) *  torch.matmul(M72_tile, orig_pt4  )# Column 7
    A8 = torch.matmul(M71_tile, pred_pt4) *  torch.matmul(M8_tile, orig_pt4  )# Column 8

    a=torch.stack([torch.reshape(A1 ,[-1 ,8]) ,torch.reshape(A2 ,[-1 ,8]), \
                                   torch.reshape(A3 ,[-1 ,8]) ,torch.reshape(A4 ,[-1 ,8]), \
                                   torch.reshape(A5 ,[-1 ,8]) ,torch.reshape(A6 ,[-1 ,8]), \
                                   torch.reshape(A7 ,[-1 ,8]) ,torch.reshape(A8 ,[-1 ,8])] ,axis=1)
    A_mat = a.permute(0 ,2 ,1) # BATCH_SIZE x 8 (A_i) x 8
    Mb_tile=Mb_tile.to(pre_4pt_shift.device)
    b_mat = torch.matmul(Mb_tile, pred_pt4)
    Mb_tile=Mb_tile.to('cpu')

    # H_8el,Lu=torch.solve(b_mat, A_mat)
    H_8el=torch.linalg.solve(A_mat,b_mat)
    # Add ones to the last cols to reconstruct H for computing reprojection error
    h_ones = torch.ones([batch_size, 1, 1])
    h_ones=h_ones.to(pre_4pt_shift.device)
    H_9el = torch.cat([H_8el ,h_ones] ,1)
    H_flat = torch.reshape(H_9el, [-1 ,9])
    H_mat = torch.reshape(H_flat ,[-1 ,3 ,3])   # BATCH_SIZE x 3 x 3
    return H_mat











