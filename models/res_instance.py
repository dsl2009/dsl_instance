from torch import nn, optim
import torch
from torch.nn import functional as F
from data import data_gen
import numpy as np
import os
from matplotlib import pyplot as plt
import glob
from loss.loss_utils import *
from layer.coord_conv import CoordConv
from loss import loss
from layer import renet
from models import resnet
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, use_coord=1):
        super(Bottleneck, self).__init__()
        if use_coord:
            ConvModel = CoordConv
        else:
            ConvModel = nn.Conv2d

        self.conv1 = ConvModel(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = ConvModel(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = ConvModel(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)


        out += residual
        out = self.relu(out)

        return out


class InstanceModel(nn.Module):
    def __init__(self, use_coord=1):
        super(InstanceModel, self).__init__()
        self.n_classes = 1
        if use_coord:
            ConvModel = CoordConv
        else:
            ConvModel = nn.Conv2d

        self.down1 = nn.Sequential(
            ConvModel(3, 64, kernel_size=3, padding=1, stride=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ConvModel(64, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )


        self.down2 = nn.Sequential(
            ConvModel(64, 256, kernel_size=3, padding=1, stride=2,bias=False),
            nn.BatchNorm2d(256),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),

        )

        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2,bias=False),
            nn.BatchNorm2d(512),
            Bottleneck(512, 128),
            Bottleneck(512, 128),
            Bottleneck(512, 128),
            Bottleneck(512, 128),
        )



        self.renet1 = renet.ReNet(512, 128, use_coordinates=True)
        self.renet2 = renet.ReNet(256, 128, use_coordinates=True)

        self.upsampling1 = nn.ConvTranspose2d(256, 100,
                                              kernel_size=(2, 2),
                                              stride=(2, 2))
        self.relu1 = nn.ReLU()
        self.upsampling2 = nn.ConvTranspose2d(356,
                                              100, kernel_size=(2, 2),
                                              stride=(2, 2))

        self.sem_seg_output = ConvModel(164,self.n_classes, kernel_size=(1, 1),stride=(1, 1))

        self.ins_seg_output = ConvModel(164, 32, kernel_size=(1, 1),stride=(1, 1))

    def forward(self, inputs):
        first_skip = self.down1(inputs)
        second_skip = self.down2(first_skip)
        x_enc = self.down3(second_skip)



        x_enc = self.renet1(x_enc)

        x_enc = self.renet2(x_enc)



        x_dec = F.relu(self.upsampling1(x_enc))

        x_dec = torch.cat((x_dec, second_skip), dim=1)


        x_dec = F.relu(self.upsampling2(x_dec))
        x_dec = torch.cat((x_dec, first_skip), dim=1)


        sem_seg_out = self.sem_seg_output(x_dec)
        ins_seg_out = self.ins_seg_output(x_dec)
        return sem_seg_out ,ins_seg_out










def train():
    md = InstanceModel()
    md.cuda()

    a = torch.randn(1,3,256,256).cuda()
    md(a)

if __name__ == '__main__':
    train()

