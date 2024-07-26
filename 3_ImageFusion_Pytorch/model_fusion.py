'''---------------------------------------------------------------------------
IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network
----------------------------------------------------------------------------'''
import torch
import time
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from context_block import ContextBlock
from torchstat import stat
# My Convolution Block
class ConvBlock(nn.Module):
    def __init__(self, inplane, outplane):
        super(ConvBlock, self).__init__()
        self.padding = (1, 1, 1, 1)
        self.conv = nn.Conv2d(inplane, outplane, kernel_size=3, padding=0, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(outplane)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = F.pad(x, self.padding, 'replicate')
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

    
class IFCNN(nn.Module):
    def __init__(self, resnet, fuse_scheme=0):
        super(IFCNN, self).__init__()
        self.fuse_scheme = fuse_scheme # MAX, MEAN, SUM
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 64)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=1, padding=0, stride=1, bias=True)
        self.CB=ContextBlock(inplanes=64,ratio=4)
        # self.CB=ContextBlock(inplanes=3,ratio=8)
        # Initialize parameters for other parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        # Initialize conv1 with the pretrained resnet101 and freeze its parameters
        for p in resnet.parameters():
            p.requires_grad = False
        self.conv1 = resnet.conv1
        self.conv1.stride = 1
        self.conv1.padding = (0, 0)
    def L1_norm(self,narry_a, narry_b):
        # caculate L1-norm
        temp_abs_a = torch.abs(narry_a)
        temp_abs_b = torch.abs(narry_b)
        l1_a = torch.sum(temp_abs_a,1)
        l1_b = torch.sum(temp_abs_b,1)
        mask_value = l1_a + l1_b
        array_MASK_a = torch.unsqueeze(l1_a/mask_value,1)
        array_MASK_b =  torch.unsqueeze(l1_b/mask_value,1)
        resule_torch = array_MASK_a*narry_a + array_MASK_b * narry_b
        return resule_torch
    def tensor_max(self, tensors):
        max_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                max_tensor = tensor
            else:
                max_tensor = torch.max(max_tensor, tensor)
        return max_tensor

    def tensor_sum(self, tensors):
        sum_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                sum_tensor = tensor
            else:
                sum_tensor = sum_tensor + tensor
        return sum_tensor

    def tensor_mean(self, tensors):
        sum_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                sum_tensor = tensor
            else:
                sum_tensor = sum_tensor + tensor
        mean_tensor = sum_tensor / len(tensors)
        return mean_tensor

    def operate(self, operator, tensors):
        out_tensors = []
        for tensor in tensors:
            out_tensor = operator(tensor)
            out_tensors.append(out_tensor)
        return out_tensors

    def tensor_padding(self, tensors, padding=(1, 1, 1, 1), mode='constant', value=0):
        out_tensors = []
        for tensor in tensors:
            out_tensor = F.pad(tensor, padding, mode=mode, value=value)
            out_tensors.append(out_tensor)
        return out_tensors
    def tensor_CB(self, tensors):
        out_tensors = []
        for tensor in tensors:
            
            out_tensor = self.CB(tensor)
            out_tensors.append(out_tensor)
        return out_tensors

    def forward(self, *tensors):
        # Feature extraction
        # tensors = self.tensor_CB(tensors)
        outs = self.tensor_padding(tensors=tensors, padding=(3, 3, 3, 3), mode='replicate')
        outs = self.operate(self.conv1, outs)# 3 -> 64  maybe1.1
        outs = self.operate(self.conv2, outs)# 64-> 64 607.13mflops*2
        out = self.tensor_max(outs)
        out = self.CB(out)# 1.1Mflops
        
        # Feature reconstruction
        out = self.conv3(out)# 64 -> 64  607.13mflops(128*128)
        
        
        
        out = self.conv4(out)# 64 -> 64  3.19Mflops
        return out


def myIFCNN(fuse_scheme=0):
    # pretrained resnet101
    resnet = models.resnet101(pretrained=True)
    # our model
    model = IFCNN(resnet, fuse_scheme=fuse_scheme)
    return model