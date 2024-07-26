import tensorflow.compat.v1 as tf
import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import time

rng = np.random.RandomState(2020)

class DataLoader(object):
    def __init__(self, image_folder):
        self.dir = image_folder
        self.images = OrderedDict()
        self.setup(image_folder)

    
    def __call__(self, batch_size):
        image_info_list = list(self.images.values())
        length = image_info_list[0]['length']

        def image_clip_generator():
            while True:

                image_clip = []
                frame_id = rng.randint(0, length-1)
                image_clip.append(np_load_input(image_info_list[2]['frame'][frame_id]))
                image_clip.append(np_load_input(image_info_list[3]['frame'][frame_id]))
                image_clip.append(np_load_input(image_info_list[0]['frame'][frame_id]))
                image_clip.append(np_load_input(image_info_list[1]['frame'][frame_id]))
                image_clip = np.concatenate(image_clip, axis=2)
                yield image_clip

        dataset = tf.data.Dataset.from_generator(generator=image_clip_generator, output_types=tf.float32, output_shapes=[None, None, 12])
        print('generator dataset, {}'.format(dataset))
        dataset = dataset.prefetch(buffer_size=16)
        dataset = dataset.shuffle(buffer_size=16).batch(batch_size)
        print('epoch dataset, {}'.format(dataset))

        return dataset

    def __getitem__(self, image_name):
        assert image_name in self.images.keys(), 'image = {} is not in {}!'.format(image_name, self.images.keys())
        return self.images[image_name]

    def setup(self,image_folder):

        images = glob.glob(os.path.join(self.dir, '*'))

        for image in sorted(images):
            image_name = image.split('\\')[-1].split('/')[-1]

            if image_name == 'ir_warp1' or image_name == 'ir_warp2' or image_name == 'vis_warp1' or image_name == 'vis_warp2':# or image_name == 'mask1_ir' or image_name == 'mask2_ir' or image_name == 'mask1_vis' or image_name == 'mask2_vis':

                self.images[image_name] = {}
                self.images[image_name]['path'] = image
                self.images[image_name]['frame'] = glob.glob(os.path.join(image, '*g'))
                self.images[image_name]['frame'].sort()
                self.images[image_name]['length'] = len(self.images[image_name]['frame'])



    def get_image_clips_ir(self, index):
        batch = []
        image_info_list = list(self.images.values())
        batch.append(np_load_input(image_info_list[0]['frame'][index]))
        batch.append(np_load_input(image_info_list[1]['frame'][index]))
        return np.concatenate(batch, axis=2)
    def get_image_clips_vis(self, index):
        batch = []
        image_info_list = list(self.images.values())
        batch.append(np_load_input(image_info_list[2]['frame'][index]))
        batch.append(np_load_input(image_info_list[3]['frame'][index]))
        return np.concatenate(batch, axis=2)


    def np_load_name(self,index):
        image_info_list = list(self.images.values())
        return image_info_list[0]['frame'][index].split('\\')[-1]

def np_load_input(filename):

    image_decoded = cv2.imread(filename)
    height = image_decoded.shape[0] 
    width = image_decoded.shape[1]  

    if width%16!=0:
        width-=(width%16)
    if height%16!=0:
        height-=(height%16)
    image_resized = cv2.resize(image_decoded, (width, height))

    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = image_resized / 127.5 - 1.
    return image_resized



def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')