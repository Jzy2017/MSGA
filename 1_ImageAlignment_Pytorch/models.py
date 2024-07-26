
import H_model
# from convblock import ConvBlock
import torch
# import output_tensorDLT
# import output_tf_spatial_transform
import time

# import vgg19
import random
def disjoint_augment_image_pair(img1,img2, min_val=0, max_val=1):
    # img1 = train_inputs[...,0:3]
    # img2 = train_inputs[...,3:6]
    
    
    # Randomly shift brightness
    # random_brightness = tf.random_uniform([], 0.7, 1.3)
    random_brightness = random.uniform( 0.7, 1.3)
    img1_aug = img1 * random_brightness
    # random_brightness=random.uniform( 0.7, 1.3)
    # random_brightness = tf.random_uniform([], 0.7, 1.3)
    random_brightness = random.uniform( 0.7, 1.3)
    img2_aug = img2 * random_brightness
    
    # Randomly shift color
    random_colors=torch.zeros((3))
    for i in range(3):
        random_colors[i]=random.uniform( 0.7, 1.3)
    # random_colors = tf.random_uniform([3], 0.7, 1.3)
    white = torch.ones(img1.shape[0], img1.shape[1], img1.shape[2])
    color_image = torch.stack([white * random_colors[i] for i in range(3)], axis=3)
    img1_aug  *= color_image

    for i in range(3):
        random_colors[i]=random.uniform( 0.7, 1.3)
    color_image = torch.stack([white * random_colors[i] for i in range(3)], axis=3)
    img2_aug  *= color_image
    
    # Saturate
    img1_aug  = torch.clamp(img1_aug,  min_val, max_val)
    img2_aug  = torch.clamp(img2_aug, min_val, max_val)
    train_inputs = torch.cat([img1_aug, img2_aug], axis = 3)
    return train_inputs

# def disjoint_augment_image_pair(ir1,ir2,vis1,vis2, min_val=0, max_val=1):
#     random_brightness = random.uniform( 0.7, 1.3)
#     ir1_aug = ir1 * random_brightness
#     random_brightness = random.uniform( 0.7, 1.3)
#     ir2_aug = ir2 * random_brightness
#     # Randomly shift color
#     random_colors=torch.zeros((3))
#     for i in range(3):
#         random_colors[i]=random.uniform( 0.7, 1.3)
#     # random_colors = tf.random_uniform([3], 0.7, 1.3)
#     white = torch.ones(ir1.shape[0], ir1.shape[1], ir1.shape[2])
#     color_image = torch.stack([white * random_colors[i] for i in range(3)], axis=3)
#     ir1_aug  *= color_image
#     for i in range(3):
#         random_colors[i]=random.uniform( 0.7, 1.3)
#     color_image = torch.stack([white * random_colors[i] for i in range(3)], axis=3)
#     ir2_aug  *= color_image
#     # Saturate
#     ir1_aug  = torch.clamp(ir1_aug,  min_val, max_val)
#     ir2_aug  = torch.clamp(ir2_aug, min_val, max_val)
#     vis1_aug  = torch.clamp(vis1_aug,  min_val, max_val)
#     vis2_aug  = torch.clamp(vis2_aug, min_val, max_val)
#     train_inputs = torch.cat([ir1_aug, ir2_aug, vis1_aug, vis2_aug], axis = 3)
#     return train_inputs
# def Vgg19_simple_api(rgb, reuse):
#     return vgg19.Vgg19_simple(rgb, reuse)
    