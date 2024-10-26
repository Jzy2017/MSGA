import os
import sys
import cv2
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import numpy as np
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

import time
def resize2tensor(x: str, reshape: bool = False, size: tuple = (), channel: int = 1):
    x = Image.open(x)
    if channel == 3:
        if reshape:
            x = x.resize(size).convert('RGB')
        tensor = transforms.Compose([transforms.ToTensor()])
    else:
        if reshape:
            x = x.resize(size)
        tensor = transforms.Compose([transforms.ToTensor()])
    x = tensor(x)
    return x


def is_image(path):
    return any(path.endswith(t) for t in IMG_EXTENSIONS)


class Image(data.Dataset):
    "数据集"
    def __init__(self,
                 use_cuda,
                 warp1_path: list,
                 warp2_path: list,
                 mask1_path: list,
                 mask2_path: list,
                 mode: str='train',
                 srcchannel: int = 3,
                 reshape: bool = False,
                 size: tuple = (128, 128)):
        super(Image, self).__init__()

        self.use_cuda = use_cuda
        self.reshape = reshape
        self.size = size
        self.srcchannel = srcchannel
        self.mode=mode
        self.warp1_path=warp1_path
        self.warp2_path=warp2_path
        self.mask1_path=mask1_path
        self.mask2_path=mask2_path
        self.warp1s = sorted(os.listdir(warp1_path))
        self.warp2s = sorted(os.listdir(warp2_path))
        self.mask1s = sorted(os.listdir(mask1_path))
        self.mask2s = sorted(os.listdir(mask2_path))
        assert len(self.warp1s)==len(self.warp2s) and len(self.warp1s)==len(self.mask1s) and len(self.warp1s)==len(self.mask2s)


    def __getitem__(self, index):
        
        warp1 = cv2.imread(self.warp1_path+'/'+self.warp1s[index])
        warp2 = cv2.imread(self.warp2_path+'/'+self.warp2s[index])
        mask1 = cv2.imread(self.mask1_path+'/'+self.mask1s[index])
        mask2 = cv2.imread(self.mask2_path+'/'+self.mask2s[index])
        if self.mode=='train':
            warp1 = cv2.resize(warp1, (384,256))
            warp2 = cv2.resize(warp2, (384,256))
            mask1 = cv2.resize(mask1, (384,256),interpolation=cv2.INTER_NEAREST)
            mask2 = cv2.resize(mask2, (384,256),interpolation=cv2.INTER_NEAREST)

        warp1= np.transpose(cv2.cvtColor(warp1, cv2.COLOR_BGR2RGB)/255.,(2,0,1))
        warp2= np.transpose(cv2.cvtColor(warp2, cv2.COLOR_BGR2RGB)/255.,(2,0,1))
        mask1= np.transpose(cv2.cvtColor(mask1, cv2.COLOR_BGR2RGB)/255.,(2,0,1))
        mask2= np.transpose(cv2.cvtColor(mask2, cv2.COLOR_BGR2RGB)/255.,(2,0,1))
        
        return warp1,warp2,mask1,mask2
       

    def __len__(self):
        return len(self.warp1s)

class Image_test(data.Dataset):
    "数据集"
    def __init__(self,
                 use_cuda,
                 warp1_path: list,
                 warp2_path: list):
        super(Image_test, self).__init__()

        self.use_cuda = use_cuda
        self.warp1_path=warp1_path
        self.warp2_path=warp2_path
        self.warp1s =  sorted(os.listdir(warp1_path))
        self.warp2s =  sorted(os.listdir(warp2_path))
        assert len(self.warp1s)==len(self.warp2s)


    def __getitem__(self, index):
        name = self.warp1s[index]
        warp1 = cv2.imread(self.warp1_path + '/' + self.warp1s[index])
        warp2 = cv2.imread(self.warp2_path + '/' + self.warp1s[index])
        h,w,c = warp1.shape
        h_reisze = h - h%8
        w_reisze = w - w%8
        warp1 = cv2.resize(warp1, (w_reisze, h_reisze))
        warp2 = cv2.resize(warp2, (w_reisze, h_reisze))
        warp1 = np.transpose(cv2.cvtColor(warp1, cv2.COLOR_BGR2RGB)/255.,(2,0,1))
        warp2 = np.transpose(cv2.cvtColor(warp2, cv2.COLOR_BGR2RGB)/255.,(2,0,1))
        
        return warp1,warp2,name
       

    def __len__(self):
        return len(self.warp1s)
