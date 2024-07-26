import cv2
import numpy as np
from PIL import Image
import torch
import time

class Transform_output(torch.nn.Module):
    """Spatial Transformer Layer

    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and edited by David Dao for Tensorflow.

    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    theta: float
        The output of the
        localisation network should be [num_batch, 6].
    out_size: tuple of two ints
        The size of the output of the network (height, width)
    """
    def __init__(self):
        super().__init__()
        # self.grid = torch.reshape(torch.reshape(torch.unsqueeze(self._meshgrid(height, width),0), [-1]).repeat(batch_size),[batch_size, 3, -1]).to(device)
        # base = _repeat(tf.range(num_batch) * dim1, out_height * out_width)
        # self.base = self._repeat(torch.arange(batch_size) * (width * height), height * width).to(device)

    def _repeat(self, x, n_repeats):

        rep = torch.unsqueeze(torch.ones(int(n_repeats),), 1).permute([1, 0])
        
        rep = torch.FloatTensor(rep)
        x=x.float()
        x = torch.matmul(torch.reshape(x, (-1, 1)), rep)
        return torch.reshape(x, [-1])

    def _interpolate(self,im, x, y, out_size):
        # constants
        batch_size = im.shape[0]
        height = im.shape[1]
        width = im.shape[2]
        channels = im.shape[3]
        x = x.float()
        y = y.float()
        height_f = height
        width_f = width
        out_height = out_size[0]
        out_width = out_size[1]
        zero=0
        max_y = im.shape[1] - 1
        max_x = im.shape[2] - 1
        #scale indices from [-1, 1] to [0, width/height]
        # x = (x + 1.0) * (width_f) / 2.0
        # y = (y + 1.0) * (height_f) / 2.0
        # do sampling
        x0 = torch.floor(x)
        x1 = x0 + 1
        y0 = torch.floor(y)
        y1 = y0 + 1
        x0 = torch.clamp(x0, zero, max_x)      #0-127 zuobiao    
        x1 = torch.clamp(x1, zero, max_x)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        dim2 = width
        dim1 = width * height
        base = self._repeat(torch.arange(batch_size) * dim1, out_height * out_width).to(im.device)
        # print(out_height*out_width)

        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim

        im_flat = torch.reshape(im, (-1, channels))
        im_flat = im_flat.float()
        Ia=im_flat[idx_a.type(torch.long)]
        Ib=im_flat[idx_b.type(torch.long)]
        Ic=im_flat[idx_c.type(torch.long)]
        Id=im_flat[idx_d.type(torch.long)]

        # and finally calculate interpolated values
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()
        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)

        output = sum([wa * Ia, wb * Ib, wc * Ic, wd * Id])

        return output

    def _transform(self,im, H, width_max, width_min, height_max, height_min):
        num_batch = im.shape[0]
        num_height = im.shape[2]
        num_width = im.shape[3]
        num_channels = im.shape[1]
        out_width = width_max - width_min
        out_height = height_max - height_min
        grid = self._meshgrid(width_max, width_min, height_max, height_min)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.view(-1)
        grid = grid.repeat(num_batch)
        # print(grid.shape)
        # time.sleep(1000)
        grid = grid.view(num_batch, 3, -1).to(im.device)
        H = H.float()

        T_g = torch.matmul(H, grid)
        x_s = T_g[:, 0:1, :]
        y_s = T_g[:, 1:2, :]
        t_s = T_g[:, 2:3, :]
        t_s_flat = torch.reshape(t_s, [-1])
        one = torch.tensor(1, dtype=torch.float32)
        small = torch.tensor(1e-7, dtype=torch.float32)
        smallers = 1e-6 * (one - torch.gt(torch.abs(t_s_flat), small).float())
        t_s_flat = t_s_flat + smallers
        x_s_flat = torch.reshape(x_s, [-1]) / t_s_flat
        y_s_flat = torch.reshape(y_s, [-1]) / t_s_flat
        input_transformed = self._interpolate(im.permute(0,2,3,1), x_s_flat, y_s_flat, (out_height,out_width))
        output = torch.reshape(input_transformed, (num_batch, out_height, out_width, num_channels)).permute(0,3,1,2)
        # print(output.shape)
        return output

        return output
    # def _meshgrid(self,height, width):
    def _meshgrid(self,width_max, width_min, height_max, height_min):
        width = width_max - width_min
        height = height_max - height_min
        # torch.linspace(width_min,  width_max, width)
        # torch.ones(shape=torch.stack([height, 1]))
        # print(torch.ones(int(height), 1).shape)
        # print("ohoho")
        # print(torch.unsqueeze(torch.linspace(width_min, width_max, width), 1).permute([1, 0]).shape)
        # x_t = torch.matmul(torch.ones(int(height), 1),
        #                     torch.unsqueeze(torch.linspace(int(width_min, width_max, width), 1).permute([1, 0]))
        x_t = torch.matmul(torch.ones(int(height), 1),
                            torch.unsqueeze(torch.linspace(float(width_min), float(width_max), int(width)), 1).permute([1, 0]))
        # print("hahah")
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(float(height_min), float(height_max), int(height)), 1),
                            torch.ones(1, int(width)))
        # print(height)
        # print(width)
        # print(height*width)
        
        # print(x_t.shape)
        # print(y_t.shape)
        x_t_flat = torch.reshape(x_t, (1, -1))
        y_t_flat = torch.reshape(y_t, (1, -1))
        # print(x_t_flat.shape)
        # print(y_t_flat.shape)
        # time.sleep(1000)
        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], 0)
        return grid

    def forward(self, inputs, H,size,resized_shift):
        # pts_1_tile = torch.repeat(size, [1, 4, 1])
        pts_1_tile = torch.tile(size, (1, 4, 1))
        tmp = torch.unsqueeze(torch.unsqueeze(torch.tensor([0., 0., 1., 0., 0., 1., 1., 1.], dtype=torch.float32, device=inputs.device), 0),-1)

        # tmp = torch.unsqueeze(torch.tensor([0., 0., 1., 0., 0., 1., 1., 1.], shape=(8,1), dtype = tf.float32), [0])
        pts_1 = pts_1_tile*tmp
        pts_2 = resized_shift + pts_1
        # pts1_list = torch.split(pts_1, 8, dim=1)
        pts1_list = torch.split(pts_1, 1, dim=1)
        pts2_list = torch.split(pts_2, 1, dim=1)
        pts_list = pts1_list + pts2_list
        width_list = [pts_list[i] for i in range(0, 16, 2)]
        height_list = [pts_list[i] for i in range(1, 16, 2)]
        width_list_tf = torch.cat(width_list, axis=1)
        height_list_tf = torch.cat(height_list, axis=1)
        width_max = int(torch.max(width_list_tf))
        width_min = int(torch.min(width_list_tf))
        height_max = int(torch.max(height_list_tf))
        height_min = int(torch.min(height_list_tf))
        out_width = int(width_max - width_min)
        out_height = int(height_max - height_min)
  
        batch_size=inputs.shape[0]
        H_one = torch.eye(3)
        H_one = torch.tile(torch.unsqueeze(H_one, 0), [batch_size, 1, 1]).to(inputs.device)
        # pts_2 = tf.add(resized_shift, pts_1)
        img1 = inputs[:, 0:3,...]
        img1 = self._transform(img1, H_one, width_max, width_min, height_max, height_min)
        warp = inputs[:, 3:6,...]
        warp = self._transform(warp, H, width_max, width_min, height_max, height_min)
        one = torch.ones_like(inputs[: , 0:3,...]).float()
        mask1 = self._transform(one, H_one, width_max, width_min, height_max, height_min)
        mask2 = self._transform(one, H, width_max, width_min, height_max, height_min)
        # resized_height = out_height - out_height%8
        # resized_width = out_width - out_width%8
        # img1 = torch.nn.functional.interpolate(img1, [resized_height, resized_width], method=0)
        # warp = torch.nn.functional.interpolate(warp, [resized_height, resized_width], method=0)
        # mask1 = torch.nn.functional.interpolate(mask1, [resized_height, resized_width], method=0)
        # mask2 = torch.nn.functional.interpolate(mask2, [resized_height, resized_width], method=0)
        output = torch.cat([img1, warp, mask1, mask2], axis=1)
        return output
