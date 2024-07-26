import os
import sys
import cv2
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import numpy as np
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
NPY_EXTENSIONS = ['.npy']

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
def is_npy(path):
    return any(path.endswith(t) for t in NPY_EXTENSIONS)





class Image_stitch_test(data.Dataset):
    #"数据集"

    def __init__(self, ir1_path: str, ir2_path: str, vis1_path: str, vis2_path: str):
        super(Image_stitch_test, self).__init__()

        self.ir1_path, self.ir2_path ,self.vis1_path, self.vis2_path = ir1_path, ir2_path, vis1_path, vis2_path
        self.ir1s = [x for x in os.listdir(ir1_path) if is_image(x)]
        self.ir2s = [x for x in os.listdir(ir2_path) if is_image(x)]
        self.vis1s = [x for x in os.listdir(vis1_path) if is_image(x)]
        self.vis2s = [x for x in os.listdir(vis2_path) if is_image(x)]
        self.ir1s.sort()
        self.ir2s.sort()
        self.vis1s.sort()
        self.vis2s.sort()
        # 检查图片匹配
        try:
            if len(self.ir1s) != len(self.ir2s):
                sys.exit(0)
            for i in range(len(self.ir1s)):
                if self.ir1s[i] != self.ir2s[i]:
                    sys.exit(0)
        except:
            print("[Src Image] and [Sal Image] don't match.")

    def __getitem__(self, index):
        name=self.vis1s[index]
        ir1 = cv2.imread(os.path.join(self.ir1_path, name))
        ir2 = cv2.imread(os.path.join(self.ir2_path, name))
        vis1 = cv2.imread(os.path.join(self.vis1_path, name))
        vis2 = cv2.imread(os.path.join(self.vis2_path, name))
        height = ir1.shape[0] 
        width = ir1.shape[1]  
        size = np.array([width, height], dtype=np.float32)
        size=np.expand_dims(size, 1)
        ir1= (cv2.cvtColor(ir1, cv2.COLOR_BGR2RGB)/255.)
        ir2= (cv2.cvtColor(ir2, cv2.COLOR_BGR2RGB)/255.)
        vis1= (cv2.cvtColor(vis1, cv2.COLOR_BGR2RGB)/255.)
        vis2= (cv2.cvtColor(vis2, cv2.COLOR_BGR2RGB)/255.)

        
        return ir1, ir2,vis1,vis2,size, name.split('/')[-1]


    def __len__(self):
        return len(self.ir1s)


