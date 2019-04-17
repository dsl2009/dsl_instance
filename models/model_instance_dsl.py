from torch import nn, optim
import torch
from torch.nn import functional as F
from data import data_gen
import numpy as np
import os
from matplotlib import pyplot as plt
import glob
from loss.loss_utils import *
from layer.coord_conv import CoordConv, CoordConvTranspose
from loss import loss
from layer import renet
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super(Down, self).__init__()
        self.cnn = nn.Sequential(
            CoordConv(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(num_features=out_c),
            nn.ReLU()
        )
    def forward(self, inputs):
        x = self.cnn(inputs)
        return x

class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super(Up, self).__init__()
        self.deconv = nn.Sequential(
            CoordConvTranspose(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_c),
            nn.ReLU()
        )
    def forward(self, x1, x2):
        x = torch.cat([x1, x2],1)
        x = self.deconv(x)
        return x

class Generater(nn.Module):
    def __init__(self):
        super(Generater, self).__init__()
        self.down1 = Down(3, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 512)
        self.down6 = Down(512, 512)
        self.down7 = Down(512, 512)
        self.down8 = Down(512, 512)
        self.decov = nn.Sequential(
            CoordConvTranspose(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU()
        )
        self.up1 = Up(1024,512)
        self.up2 = Up(1024, 512)
        self.up3 = Up(1024, 512)
        self.up4 = Up(1024, 256)
        self.up5 = Up(512, 128)
        self.up6 = Up(256, 64)
        self.mask = CoordConvTranspose(in_channels=128, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.edge = CoordConvTranspose(in_channels=128, out_channels=1, kernel_size=4, stride=2, padding=1)

    def forward(self, inputs):
        dw1 = self.down1(inputs)
        dw2 = self.down2(dw1)
        dw3 = self.down3(dw2)
        dw4 = self.down4(dw3)
        dw5 = self.down5(dw4)
        dw6 = self.down6(dw5)
        #dw6 = F.dropout2d(dw6, p=0.3, inplace=True)
        dw7 = self.down7(dw6)
        decov1 = self.decov(dw7)
        up = self.up2(dw6, decov1)
        up = self.up3(dw5, up)
        up = self.up4(dw4, up)
        up = self.up5(dw3, up)
        up = self.up6(dw2, up)
        up = torch.cat([dw1, up],1)
        mask_logits = self.mask(up)
        edge_logits = self.edge(up)
        return mask_logits, edge_logits






