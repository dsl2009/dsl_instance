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
            CoordConv(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1),
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
            CoordConvTranspose(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=out_c),
            nn.ReLU()
        )
    def forward(self, x1, x2):
        x = torch.cat([x1, x2],1)
        x = self.deconv(x)
        return x

class InstanceModel(nn.Module):
    def __init__(self):
        super(InstanceModel, self).__init__()
        self.n_classes = 1
        self.down1 = Down(3, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 512)
        self.down6 = Down(512, 512)
        self.down7 = Down(512, 512)
        self.down8 = Down(512, 512)
        self.decov = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU()
        )
        self.up1 = Up(1024,512)
        self.up2 = Up(1024, 512)
        self.up3 = Up(1024, 512)
        self.up4 = Up(1024, 256)
        self.up5 = Up(512, 256)
        self.up6 = Up(384, 128)
        self.up7 = Up(192, 128)

        self.renet1 = renet.ReNet(256, 128)
        self.renet2 = renet.ReNet(256, 128)
        self.sem_seg_output = nn.Conv2d(128,self.n_classes, kernel_size=(1, 1),stride=(1, 1))


        self.ins_seg_output = nn.Conv2d(128, 32, kernel_size=(1, 1),stride=(1, 1))

    def forward(self, inputs):
        dw1 = self.down1(inputs)
        dw2 = self.down2(dw1)
        dw3 = self.down3(dw2)
        dw4 = self.down4(dw3)
        dw5 = self.down5(dw4)
        dw6 = self.down6(dw5)
        dw7 = self.down7(dw6)

        decov1 = self.decov(dw7)
        up = self.up2(dw6, decov1)
        up = self.up3(dw5, up)
        up = self.up4(dw4, up)
        xenc = self.up5(dw3, up)
        xenc = self.renet1(xenc)
        xenc = self.renet2(xenc)
        up = self.up6(dw2,xenc)
        up = self.up7(dw1, up)
        sem_seg_out = self.sem_seg_output(up)
        ins_seg_out = self.ins_seg_output(up)

        return sem_seg_out, ins_seg_out








def train():
    md = InstanceModel()
    md.cuda()
    a = torch.randn(1,3,256,256).cuda()
    md(a)

if __name__ == '__main__':
    train()

