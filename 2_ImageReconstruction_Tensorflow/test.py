
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.disable_eager_execution()
import numpy as np
import cv2
import os

from models import reconstruction
from utils_testing import DataLoader, load, save
import constant
import time


os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
test_folder = '../output/example'
save_folder=test_folder
snapshot_dir =   'snapshot/model.ckpt-200000'
batch_size = constant.TEST_BATCH_SIZE
os.makedirs(os.path.join(test_folder,'recon_ir'),exist_ok=True)
os.makedirs(os.path.join(test_folder,'recon_vis'),exist_ok=True)

# define dataset
with tf.name_scope('dataset'):
    ##########testing###############
    test_inputs = tf.placeholder(shape=[batch_size, None, None, 3 * 2], dtype=tf.float32)
    print('test inputs = {}'.format(test_inputs))


# define testing generator function
with tf.variable_scope('Reconstruction', reuse=None):
    print('testing = {}'.format(tf.get_variable_scope().name))
    hr_test_stitched = reconstruction(test_inputs)



config = tf.ConfigProto()
config.gpu_options.allow_growth = True      
with tf.Session(config=config) as sess:
    # dataset
    data_loader = DataLoader(test_folder)

    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init global successfully!')

    # tf saver
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)

    restore_var = [v for v in tf.global_variables()]
    
    loader = tf.train.Saver(var_list=restore_var)
    length = data_loader.images['ir_warp1']['length'] 
    def inference_func(ckpt):
        print("============")
        print(ckpt)
        load(loader, sess, ckpt)
        print("============")
        for i in range(0, length):
            # start=time.time()
            input_clip_ir = np.expand_dims(data_loader.get_image_clips_ir(i), axis=0)
            input_clip_vis = np.expand_dims(data_loader.get_image_clips_vis(i), axis=0)
            name=data_loader.np_load_name(i)
            print(name)
            start=time.time()
            stitch_result_ir = sess.run(hr_test_stitched, feed_dict={test_inputs: input_clip_ir})
            stitch_result_vis = sess.run(hr_test_stitched, feed_dict={test_inputs: input_clip_vis})
            print(str(i+1)+' '+ str(time.time()-start))

            stitch_result_ir = (stitch_result_ir+1) * 127.5     
            stitch_result_ir = stitch_result_ir[0]
                
            path1 = os.path.join(test_folder,'recon_ir',name)
            cv2.imwrite(path1, stitch_result_ir)

            stitch_result_vis = (stitch_result_vis+1) * 127.5     
            stitch_result_vis = stitch_result_vis[0]
            path2 = os.path.join(test_folder,'recon_vis',name)
            cv2.imwrite(path2, stitch_result_vis)
            print(i+1)

            
        print("===================DONE!==================")  

    inference_func(snapshot_dir)

    

