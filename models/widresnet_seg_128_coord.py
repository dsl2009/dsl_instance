from models import resnet_coord as resnet
import torch
from torch import nn
from torch.nn import functional as F
from layer import renet
from layer.coord_conv import CoordConv

class SegModel(nn.Module):
    def __init__(self):
        super(SegModel, self).__init__()
        self.cnn = resnet.resnet101(pretrained=False)
        self.cov1 = nn.Sequential(
            CoordConv(2048, 512, kernel_size=1, stride=1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

        )

        self.cov2 = nn.Sequential(
            CoordConv(1024, 512, kernel_size=3,padding=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.cov3 = nn.Sequential(
            CoordConv(768, 256, kernel_size=3,padding=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.cov4 = nn.Sequential(
            CoordConv(320, 256, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.renet1 = renet.ReNet(512, 256, use_coordinates=True)
        self.renet2 = renet.ReNet(512, 256, use_coordinates=True)

        self.final_seg = CoordConv(256, 1, kernel_size=3,padding=1, stride=1, bias=False)

        self.final_instance = CoordConv(256, 16, kernel_size=3, padding=1, stride=1, bias=False)

    def forward(self, img):
        x1, x2, x3, _, x4 = self.cnn(img)
        x4 = self.cov1(x4)

        x4_up = F.interpolate(x4, scale_factor=2)
        x3 = torch.cat([x4_up, x3], dim=1)
        x3 = self.cov2(x3)

        x3 = self.renet1(x3)
        x3 = self.renet2(x3)

        x3_up = F.interpolate(x3,scale_factor=2)
        x2 = torch.cat([x3_up, x2],dim =1)
        x2 = self.cov3(x2)
        x2_up = F.interpolate(x2,scale_factor=2)
        x1 = torch.cat([x2_up, x1],dim =1)
        x1 = self.cov4(x1)
        seg = self.final_seg(x1)
        ins = self.final_instance(x1)
        return seg, ins






if __name__ == '__main__':
    x = torch.randn(2,3,256,256).cuda()
    md = SegModel()
    md.cuda()
    md(x)
