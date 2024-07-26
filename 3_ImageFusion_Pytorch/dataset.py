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


class Image_One(data.Dataset):
    "数据集"

    def __init__(self,
                 use_cuda,
                 src_path: list,
                 mode: str='train',
                 srcchannel: int = 3,
                 reshape: bool = False,
                 size: tuple = (128, 128)):
        super(Image_One, self).__init__()

        self.use_cuda = use_cuda
        self.reshape = reshape
        self.size = size
        self.srcchannel = srcchannel
        self.src_path = src_path
        self.mode=mode
        print(len(src_path))
        self.srcs = [x for x in os.listdir(src_path[0]) if is_image(x)]
        self.all_path=[]
        for i in range(len(src_path)):
            for j in range(len(self.srcs)):
                self.all_path.append(os.path.join(self.src_path[i], self.srcs[j]))
        self.all_path.sort()

        print(len(self.all_path))

    def __getitem__(self, index):
        src = cv2.imread(self.all_path[index])
        if self.mode=='train':
            src = cv2.resize(src, (256,128))
 
        # print(src.shape)
        # src=np.expand_dims(src,0)
        # src= cv2.cvtColor(src, cv2.COLOR_BGR2YCR_CB)/255.
        src= cv2.cvtColor(src, cv2.COLOR_BGR2RGB)/255.
        src = np.transpose(src,(2,0,1))
        if self.mode=='train':
            return src,src.copy()
        else:
            return src,self.all_path[index].split('\\')[-1]

    def __len__(self):
        return len(self.all_path)


class Image_fus(data.Dataset):
    #"数据集"

    def __init__(self, use_cuda, src_path: str, src2_path: str):
        super(Image_fus, self).__init__()

        self.use_cuda = use_cuda

        self.src_path, self.src2_path = src_path, src2_path,
        self.srcs = [x for x in os.listdir(src_path) if is_image(x)]
        self.srcs2 = [x for x in os.listdir(src2_path) if is_image(x)]
        self.srcs.sort()
        self.srcs2.sort()
        print(len(self.srcs))
        print(len(self.srcs2))

        # 检查图片匹配
        try:
            if len(self.srcs) != len(self.srcs2):
                sys.exit(0)
            for i in range(len(self.srcs)):
                if self.srcs[i] != self.srcs2[i]:
                    sys.exit(0)
        except:
            print("[Src Image] and [Sal Image] don't match.")

    def __getitem__(self, index):
        name=self.srcs[index]
        src = cv2.imread(os.path.join(self.src_path, name))
        # print(src)
        # time.sleep(100)
        # src = cv2.resize(src, (256,128))
        # src= cv2.cvtColor(src, cv2.COLOR_BGR2YCR_CB)/255.
        src= cv2.cvtColor(src, cv2.COLOR_BGR2RGB)/255.
        src = np.transpose(src,(2,0,1))

        src2 = cv2.imread(os.path.join(self.src2_path,name))
        # src2 = cv2.resize(src2, (256,128))
        # src2= cv2.cvtColor(src2, cv2.COLOR_BGR2YCR_CB)/255.
        src2= cv2.cvtColor(src2, cv2.COLOR_BGR2RGB)/255.
        src2 = np.transpose(src2,(2,0,1))


        return src, src2, name

    def __len__(self):
        return len(self.srcs)


