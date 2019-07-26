from models import resnet_coord as resnet
import torch
from torch import nn
from torch.nn import functional as F
from layer import renet
class SegModel(nn.Module):
    def __init__(self):
        super(SegModel, self).__init__()
        self.cnn = resnet.resnet50(use_dsl=True)
        self.cov1 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False),
            nn.GroupNorm(16, 512),
            nn.ReLU(),

        )

        self.cov2 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1, stride=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU()
        )

        self.cov3 = nn.Sequential(
            nn.Conv2d(320, 256, kernel_size=3, padding=1, stride=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU()
        )

        self.renet1 = renet.ReNet(512, 256, use_coordinates=False)
        self.renet2 = renet.ReNet(512, 256, use_coordinates=False)

        self.final_seg = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=3, padding=1, stride=1, bias=False)
        )
        self.final_instance =nn.Sequential(
            nn.Conv2d(256, 16, kernel_size=3, padding=1, stride=1, bias=False)
        )


    def forward(self, img):
        x1, x2, x3 = self.cnn(img)

        x3 = self.cov1(x3)
        x3 = self.renet1(x3)
        x3 = self.renet2(x3)
        x3_up = F.interpolate(x3,scale_factor=2)
        x2 = torch.cat([x3_up, x2],dim =1)
        x2 = self.cov2(x2)
        x2_up = F.interpolate(x2,scale_factor=2)
        x1 = torch.cat([x2_up, x1],dim =1)
        x1 = self.cov3(x1)
        #x1 = F.interpolate(x1,scale_factor=2)
        seg = self.final_seg(x1)
        ins = self.final_instance(x1)

        return seg, ins






if __name__ == '__main__':
    x = torch.randn(2,3,256,256)
    md = SegModel()
    md(x)
