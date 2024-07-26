import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from context_block import ContextBlock
def edge_extraction(gen_frames,use_cuda):
    
    # calculate the loss for each scale
    # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.

    channels = gen_frames.shape[1]
    pos = torch.from_numpy(np.identity(channels))     # 3 x 3
    neg = -1 * pos
    filter_x = torch.cat([neg, pos], 0).unsqueeze(0)
    filter_y = torch.cat([neg, pos], 1).unsqueeze(0)
    # print(neg.shape)

    strides = [1, 1, 1, 1]  # stride of (1, 1)
    padding = 'SAME'
    conv_x = nn.Conv2d(1,1,kernel_size=(2,1),stride=1,bias=False,padding=0)
    # conv_x = nn.Conv2d(1,1,3,bias=False)
    conv_y = nn.Conv2d(1,1,kernel_size=(1,2),stride=1,bias=False,padding=0)
    conv_x.requires_grad = False
    conv_y.requires_grad = False
    x_pad=nn.ZeroPad2d((0,0,1,0))
    y_pad=nn.ZeroPad2d((1,0,0,0))
    if use_cuda:
        filter_x=filter_x.cuda()
        filter_y=filter_y.cuda()
        conv_x=conv_x.cuda()
        conv_y=conv_y.cuda()
        x_pad=x_pad.cuda()
        y_pad=y_pad.cuda()
    conv_x.weight.data = filter_x.unsqueeze(0).float()
    conv_y.weight.data = filter_y.unsqueeze(0).float()

    # a=conv_x(gen_frames)
    gen_dx = torch.abs(conv_x(x_pad(gen_frames)))
    gen_dy = torch.abs(conv_y(y_pad(gen_frames)))

    edge = gen_dx ** 1 + gen_dy ** 1
    edge_clip  = torch.clamp(edge, 0, 1)
    # condense into one tensor and avg
    return edge_clip
def seammask_extraction(mask,use_cuda):

    seam_mask = edge_extraction(torch.unsqueeze(torch.mean(mask, axis=1),1),use_cuda)
    filters = torch.from_numpy(np.array([[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]])).unsqueeze(0).unsqueeze(0).float()
    conv_1 = nn.Conv2d(1,1,kernel_size=3,stride=1,bias=False,padding=1)
    conv_2 = nn.Conv2d(1,1,kernel_size=3,stride=1,bias=False,padding=1)
    conv_3 = nn.Conv2d(1,1,kernel_size=3,stride=1,bias=False,padding=1)
    conv_2.requires_grad = False
    conv_2.requires_grad = False
    conv_3.requires_grad = False

    if use_cuda:
        conv_1=conv_1.cuda()
        conv_2=conv_2.cuda()
        conv_3=conv_3.cuda()
        filters=filters.cuda()
    conv_1.weight.data = filters
    conv_2.weight.data = filters
    conv_3.weight.data = filters
    test_conv1 =conv_1(seam_mask)
    test_conv1 = torch.clamp(test_conv1, 0, 1)
    test_conv2 =conv_2(test_conv1)
    test_conv2 = torch.clamp(test_conv2, 0, 1)
    test_conv3 =conv_3(test_conv2)
    test_conv3 = torch.clamp(test_conv3, 0, 1)
    # condense into one tensor and avg
    return test_conv3

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        # if bilinear:
        #     self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #     self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        # else:
        # self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # print(x1.shape)
        x1 = self.up(x1)
        # print(x1.shape)
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
    def forward(self, x):
        return self.tanh(self.conv(x))


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        self.down3 = Down(256, 512)
        # factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)
        # self.up1 = Up(1024, 512 // factor, bilinear)
        self.deconv = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        # self.up1 = Up(512, 256 // factor, bilinear)
        # self.up2 = Up(256, 128 // factor, bilinear)
        self.up1 = Up(512, 256, bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.CB512=ContextBlock(inplanes=512,ratio=1)
        self.CB256=ContextBlock(inplanes=256,ratio=1)
        self.CB128=ContextBlock(inplanes=128,ratio=2)
        self.CB64=ContextBlock(inplanes=64,ratio=3)

    def forward(self, x):
        x1 = self.inc(x)#1, 64, 256, 384
        x2 = self.down1(x1)#1, 128, 128, 192
        x3 = self.down2(x2)#1, 256, 64, 96
        x4 = self.down3(x3)#1, 512, 32, 48
        x = self.up1(x4,  self.CB256(x3))#1, 128, 64, 96
        x = self.up2(x,  self.CB128(x2))#1, 64, 128, 192
        x = self.up3(x,  self.CB64(x1))#1, 64, 256, 384
        logits = self.outc(x)
        return logits
class LocNet(torch.nn.Module):
    def __init__(self):
        super(LocNet, self).__init__()

        ch = 9**2 *3 + 7**2 *3
        self.layer1 = nn.Linear(ch, ch*2)
        self.bn1 = nn.BatchNorm1d(ch*2)
        self.layer2 = nn.Linear(ch*2, ch*2)
        self.bn2 = nn.BatchNorm1d(ch*2)
        self.layer3 = nn.Linear(ch*2, ch)
        self.bn3 = nn.BatchNorm1d(ch)
        self.layer4 = nn.Linear(ch, 6)

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                # print(classname)
                nn.init.kaiming_normal(m.weight)
                nn.init.constant(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.layer2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.layer3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.layer4(x)
        return x