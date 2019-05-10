from models import resnet
import torch
from torch import nn
from torch.nn import functional as F
from layer import renet
class SegModel(nn.Module):
    def __init__(self):
        super(SegModel, self).__init__()
        self.cnn = resnet.resnet50(pretrained=False)
        self.cov1 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, stride=1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

        )

        self.cov2 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3,padding=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.cov3 = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3,padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.seg = nn.Conv2d(64, 1, kernel_size=3,padding=1, stride=1, bias=False)
        self.edge = nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=1, bias=False)


    def forward(self, img):
        x1, x2, x3 = self.cnn(img)
        x3 = self.cov1(x3)
        x3_up = F.interpolate(x3,scale_factor=2, mode='bilinear')
        x2 = torch.cat([x3_up, x2],dim =1)
        x2 = self.cov2(x2)
        x2_up = F.interpolate(x2,scale_factor=2, mode='bilinear')
        x1 = torch.cat([x2_up, x1],dim =1)
        x1 = self.cov3(x1)
        x0 = F.interpolate(x1,scale_factor=2, mode='bilinear')
        seg = self.seg(x0)
        edge = self.edge(x0)
        return seg,edge






if __name__ == '__main__':
    x = torch.randn(2,3,256,256).cuda()
    md = SegModel().cuda()
    md(x)
