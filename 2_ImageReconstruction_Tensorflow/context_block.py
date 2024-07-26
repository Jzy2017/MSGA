import tensorflow.compat.v1 as tf
import numpy as np
import os
import time
import tf_slim as slim
os.environ['CUDA_VISIBLE_DEVICES'] = 'cpu'



def spatial_pool(x):
    batch, height, width,channel = x.shape
    input_x = x
    input_x = tf.reshape(input_x, [batch, -1,channel])    # [N, H * W, C]\
    input_x = tf.expand_dims(input_x,1) # [N, 1, H * W,   C]
    context_mask = slim.conv2d(inputs=x, num_outputs=1, kernel_size=1, activation_fn=None,normalizer_fn=None)  # [N, H, W, 1]
    context_mask = tf.reshape(context_mask,[batch,-1, 1])    # [N,  H * W,   1   ]
    context_mask = tf.nn.softmax(context_mask,axis=1)    # [N, H * W,  1  ]
    context_mask =  tf.expand_dims(context_mask,-1)    # [N, H * W,   1  , 1]
    input_x=tf.transpose(input_x,(0,3,1,2))    # [N, 1, C, 1]
    context_mask=tf.transpose(context_mask,(0,3,1,2))
    context = tf.matmul(input_x, context_mask)
    context=tf.transpose(context,(0,2,3,1))    #  [N, 1, 1, C]
    return context
# return context
def channel_add_conv(x,inplane,channel):# = nn.Sequential(
    encoder_conv1_1 = slim.conv2d(inputs=x, num_outputs=inplane, kernel_size=1,weights_initializer = tf.constant_initializer(0), activation_fn=tf.nn.relu,normalizer_fn=slim.layer_norm) # 6->64
    encoder_conv1_1 = slim.conv2d(inputs=encoder_conv1_1, num_outputs=channel, kernel_size=1, weights_initializer = tf.constant_initializer(0), activation_fn=None,normalizer_fn=None) # 6->64
    return encoder_conv1_1
def context_block(x,ratios):
    context=spatial_pool(x)
    out=x
    channel=x.shape[3]
    channel_add_term = channel_add_conv(context,channel*ratios,channel)
    out = out + channel_add_term
    return out
