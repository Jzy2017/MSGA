import tensorflow.compat.v1 as tf
import tf_slim as slim
from context_block import context_block

import time

    

def ReconstructionNetAttention(inputs):

    # batch_size = tf.shape(inputs)[0]
    batch_size=inputs.shape[0]
    height = inputs.shape[1]
    width = inputs.shape[2]
    inputs.set_shape([batch_size, height, width, 6])

    LR_inputs = tf.concat([inputs[...,0:3], inputs[...,3:6]], axis=3)
    encoder_conv1_1 = slim.conv2d(inputs=LR_inputs, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu) # 6->64
    encoder_conv1_2 = slim.conv2d(inputs=encoder_conv1_1, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)  # 64->64
    encoder_pooling1 = slim.max_pool2d(inputs=encoder_conv1_2, kernel_size=2, padding='SAME') # 降采样
    encoder_conv2_1 = slim.conv2d(inputs= encoder_pooling1, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu) # 64->128
    encoder_conv2_2 = slim.conv2d(inputs= encoder_conv2_1, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu) # 128->128
    encoder_pooling2 = slim.max_pool2d(inputs= encoder_conv2_2, kernel_size=2, padding='SAME') # 降采样
    encoder_conv3_1 = slim.conv2d(inputs= encoder_pooling2, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu) # 128->256
    encoder_conv3_2 = slim.conv2d(inputs= encoder_conv3_1, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu) # 256->256
    encoder_pooling3 = slim.max_pool2d(inputs= encoder_conv3_2, kernel_size=2, padding='SAME') # 降采样
    encoder_conv4_1 = slim.conv2d(inputs= encoder_pooling3, num_outputs=512, kernel_size=3, activation_fn=tf.nn.relu) # 256->512
    encoder_conv4_2 = slim.conv2d(inputs= encoder_conv4_1, num_outputs=512, kernel_size=3, activation_fn=tf.nn.relu) # 512->512
    ############################################################################ decoder ############################################################################
    decoder_up1 = slim.conv2d_transpose(inputs= encoder_conv4_2, num_outputs=256, kernel_size=2, stride=2) # 512->256
    ##################################################################################################################################
    decoder_concat1 = tf.concat([context_block(encoder_conv3_2,2), decoder_up1], axis=3) #[256,256]->512
    decoder_conv1_1 = slim.conv2d(inputs=decoder_concat1, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu) # 512->256
    # decoder_conv1_2 = slim.conv2d(inputs=decoder_conv1_1, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu) # 256->256
    decoder_up2 = slim.conv2d_transpose(inputs=decoder_conv1_1, num_outputs=128, kernel_size=2, stride=2) # 反卷积(长宽翻倍) 256->128
  
    decoder_concat2 = tf.concat([context_block(encoder_conv2_2,2), decoder_up2], axis=3) #[128,128]->256
    decoder_conv2_1 = slim.conv2d(inputs=decoder_concat2, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu) # 256->128
    # decoder_conv2_2 = slim.conv2d(inputs=decoder_conv2_1, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu) # 128->128
    decoder_up3 = slim.conv2d_transpose(inputs=decoder_conv2_1, num_outputs=64, kernel_size=2, stride=2) # 反卷积(长宽翻倍) 128->64
    decoder_concat3 = tf.concat([context_block(encoder_conv1_2,2), decoder_up3], axis=3) #[64,64]->128
    decoder_conv3_1 = slim.conv2d(inputs=decoder_concat3, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu) # 128->64
    # decoder_conv3_2 = slim.conv2d(inputs=decoder_conv3_1, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu) # 64->64
    # the output of low-resolution branch
    LR_output = slim.conv2d(inputs=decoder_conv3_1, num_outputs=3, kernel_size=3, activation_fn=None)  # 64->3
    LR_output = tf.tanh(LR_output)

    return LR_output




   
