from torch import nn
import torch
from torch.nn import functional as F
from layer.coord_conv import CoordConv, CoordConvTranspose



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

class Generater(nn.Module):
    def __init__(self, ip_dim):
        super(Generater, self).__init__()
        self.down1 = Down(ip_dim, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)
        self.down5 = Down(256, 256)
        self.down6 = Down(256, 512)
        self.decov = nn.Sequential(
            CoordConvTranspose(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            CoordConvTranspose(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),

            CoordConvTranspose(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),

            CoordConvTranspose(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),

            CoordConvTranspose(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            CoordConvTranspose(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1),
        )


    def forward(self, inputs):
        dw1 = self.down1(inputs)
        dw2 = self.down2(dw1)
        dw3 = self.down3(dw2)
        dw4 = self.down4(dw3)
        dw5 = self.down5(dw4)
        dw6 = self.down6(dw5)
        p_logits = self.decov(dw6)
        p_out_put = F.sigmoid(p_logits)
        return p_logits, p_out_put