from torch import nn, optim
import torch
from torch.nn import functional as F
from models.resnet import resnet50
from layer.coord_conv import CoordConv, CoordConvTranspose

class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super(Up, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_c),
            nn.ReLU()
        )
    def forward(self, x1, x2):
        x = torch.cat([x1, x2],1)
        x = self.deconv(x)
        return x

class EdgeModel(nn.Module):
    def __init__(self):
        super(EdgeModel, self).__init__()
        self.cnn = resnet50(pretrained=False)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2048, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU()
        )
        self.up2 = Up(768, 256)
        self.mask = nn.ConvTranspose2d(in_channels=320, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.edge = nn.ConvTranspose2d(in_channels=320, out_channels=1, kernel_size=4, stride=2, padding=1)


    def forward(self, img):
        x_128, x_64, x_32 = self.cnn(img)
        x = self.up1(x_32)
        x = self.up2(x,x_64)
        x = torch.cat([x,x_128],1)
        mask_logits = self.mask(x)
        edge_logits = self.edge(x)

        return mask_logits, edge_logits

if __name__ == '__main__':
    md = EdgeModel()
    md.cuda()
    ig = torch.randn(1, 3, 256, 256).cuda()
    md(ig)



