import cv2
import numpy as np
from PIL import Image
import torch
import time

class Transform(torch.nn.Module):
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
    def __init__(self, height, width, device,batch_size):
        super().__init__()
        self.grid = torch.reshape(torch.reshape(torch.unsqueeze(self._meshgrid(height, width),0), [-1]).repeat(batch_size),[batch_size, 3, -1]).to(device)
        # base = _repeat(tf.range(num_batch) * dim1, out_height * out_width)
        self.base = self._repeat(torch.arange(batch_size) * (width * height), height * width).to(device)

    def _repeat(self, x, n_repeats):
        # Process
        # dim2 = width
        # dim1 = width*height
        # v = tf.range(num_batch)*dim1
        # print 'old v:', v # num_batch
        # print 'new v:', tf.reshape(v, (-1, 1)) # widthx1
        # n_repeats = 20
        # rep = tf.transpose(tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0]) # 1 x out_width*out_height
        # print rep
        # rep = tf.cast(rep, 'int32')
        # v = tf.matmul(tf.reshape(v, (-1, 1)), rep) # v: num_batch x (out_width*out_height)
        # print '--final v:\n', v.eval()
        # # v is the base. For parallel computing.
        #with tf.variable_scope('_repeat'):
        rep = torch.unsqueeze(torch.ones(n_repeats,), 1).permute([1, 0])
        
        rep = torch.FloatTensor(rep)
        x=x.float()
        x = torch.matmul(torch.reshape(x, (-1, 1)), rep)
        return torch.reshape(x, [-1])

    def _interpolate(self,im, x, y, out_size):
        # constants
        height = im.shape[1]
        width = im.shape[2]
        channels = im.shape[3]
        x = x.float()
        y = y.float()
        height_f = height
        width_f = width
        zero=0
        max_y = im.shape[1] - 1
        max_x = im.shape[2] - 1
        #scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0) * (width_f) / 2.0
        y = (y + 1.0) * (height_f) / 2.0
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
        base_y0 = self.base + y0 * dim2
        base_y1 = self.base + y1 * dim2
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


    def _meshgrid(self,height, width):
        x_t = torch.matmul(torch.ones((height, 1)),
                            torch.unsqueeze(torch.linspace(-1.0, 1.0, width), 1).permute([1, 0]))
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(-1.0, 1.0, height), 1),
                            torch.ones(1, width))
        x_t_flat = torch.reshape(x_t, (1, -1))
        y_t_flat = torch.reshape(y_t, (1, -1))

        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], 0)
        return grid

    def forward(self, image2_tensor, H_tf):

        num_batch = image2_tensor.shape[0]

        height = image2_tensor.shape[2]
        width = image2_tensor.shape[3]
        num_channels =image2_tensor.shape[1]
        #  Changed
        H_tf = torch.reshape(H_tf, (-1, 3, 3))
        H_tf = H_tf.float()
        #  Added: add two matrices M and B defined as follows in
        # order to perform the equation: H x M x [xs...;ys...;1s...] + H x [width/2...;height/2...;0...]
        H_tf_shape = H_tf.shape
        # initial
        # # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
        T_g = torch.matmul(H_tf, self.grid)
        # x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
        # # Ty changed
        # # y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
        # y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
        # # Ty added
        # t_s = tf.slice(T_g, [0, 2, 0], [-1, 1, -1])
        x_s = T_g[:, 0:1, :]
        y_s = T_g[:, 1:2, :]
        t_s = T_g[:, 2:3, :]
        # The problem may be here as a general homo does not preserve the parallelism
        # while an affine transformation preserves it.
        t_s_flat = torch.reshape(t_s, [-1])

        # # Avoid zero division
        # zero = tf.constant(0, dtype=tf.float32)
        # one = tf.constant(1, dtype=tf.float32)
        #
        # # smaller
        # small = tf.constant(1e-7, dtype=tf.float32)
        # smallers = 1e-6 * (one - tf.cast(tf.greater_equal(tf.abs(t_s_flat), small), tf.float32))
        #
        # t_s_flat = t_s_flat + smallers
        # condition = tf.reduce_sum(tf.cast(tf.greater(tf.abs(t_s_flat), small), tf.float32))

        #  batchsize * width * height
        x_s_flat = torch.reshape(x_s, [-1]) / t_s_flat
        y_s_flat = torch.reshape(y_s, [-1]) / t_s_flat
        input_transformed = self._interpolate(image2_tensor.permute(0,2,3,1), x_s_flat, y_s_flat, (height,width))
        output = torch.reshape(input_transformed, (num_batch, height, width, num_channels)).permute(0,3,1,2)
        # print(output.shape)
        return output


    # output = _transform(image2_tensor, H_tf)
    # return output






