import os
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
        self.warp1s =  os.listdir(warp1_path)
        self.warp2s =  os.listdir(warp2_path)
        self.warp1s.sort()
        self.warp2s.sort()
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
