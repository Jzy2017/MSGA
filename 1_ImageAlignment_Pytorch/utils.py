import torch
import numpy as np
import cv2
import time

def DLT_solve(src_p, off_set):
    # src_p: shape=(bs, n, 4, 2)
    # off_set: shape=(bs, n, 4, 2)
    # can be used to compute mesh points (multi-H)
    bs, _ = src_p.shape
    divide = int(np.sqrt(len(src_p[0])/2)-1)# divide=1 
    row_num = (divide+1)*2# row_num = 4
    for i in range(divide):
        for j in range(divide):
            h4p = src_p[:,[ 2*j + row_num*i, 2*j + row_num*i + 1, 
                    2*(j+1) + row_num*i, 2*(j+1) + row_num*i + 1, 
                    2*(j+1) + row_num*i + row_num, 2*(j+1) + row_num*i + row_num + 1,
                    2*j + row_num*i + row_num, 2*j + row_num*i + row_num+1]].reshape(bs, 1, 4, 2)              
            pred_h4p = off_set[:,[2*j+row_num*i, 2*j+row_num*i+1, 
                    2*(j+1)+row_num*i, 2*(j+1)+row_num*i+1, 
                    2*(j+1)+row_num*i+row_num, 2*(j+1)+row_num*i+row_num+1,
                    2*j+row_num*i+row_num, 2*j+row_num*i+row_num+1]].reshape(bs, 1, 4, 2)

            if i+j==0:
                src_ps = h4p
                off_sets = pred_h4p
            else:
                src_ps = torch.cat((src_ps, h4p), axis = 1)    
                off_sets = torch.cat((off_sets, pred_h4p), axis = 1)

    bs, n, h, w = src_ps.shape

    N = bs*n #1*1=1

    src_ps = src_ps.reshape(N, h, w)#(1,4,2)
    off_sets = off_sets.reshape(N, h, w)#(1,4,2)

    dst_p = src_ps + off_sets#
    # print(dst_p)
    ones = torch.ones(N, 4, 1) #(1,4,1)
    if torch.cuda.is_available():
        ones = ones.cuda()
    xy1 = torch.cat((src_ps, ones), 2)#(1,4,3) 
    # print(xy1.shape)
    zeros = torch.zeros_like(xy1)#(1,4,3)
    if torch.cuda.is_available():
        zeros = zeros.cuda()

    xyu, xyd = torch.cat((xy1, zeros), 2), torch.cat((zeros, xy1), 2)#(1,4,6)
    M1 = torch.cat((xyu, xyd), 2).reshape(N, -1, 6)
    M2 = torch.matmul(
        dst_p.reshape(-1, 2, 1), 
        src_ps.reshape(-1, 1, 2),
    ).reshape(N, -1, 2)

    A = torch.cat((M1, -M2), 2)
    b = dst_p.reshape(N, -1, 1)

    Ainv = torch.inverse(A)
    h8 = torch.matmul(Ainv, b).reshape(N, 8)
 
    H = torch.cat((h8, ones[:,0,:]), 1).reshape(N, 3, 3)
    H = H.reshape(bs, n, 3, 3)
    return H