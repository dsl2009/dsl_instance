import torch
import torch.nn as nn
import torch.nn.functional as F
from deelab.assp import build_aspp
from deelab.decoder import build_decoder
from deelab.resnet_norm import resnet50
from layer.coord_conv import CoordConv

if False:
    Conv = nn.Conv2d
else:
    Conv = CoordConv

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=8, num_classes=1,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()



        self.backbone = resnet50()
        self.conv4 = nn.Sequential(
            Conv(in_channels=2048, out_channels=256, kernel_size=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            Conv(in_channels=1024, out_channels=256, kernel_size=1, padding=0, bias=False),

            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            Conv(in_channels=512, out_channels=256, kernel_size=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            Conv(in_channels=512, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        )

        self.conv43 = nn.Sequential(
            Conv(in_channels=512, out_channels=256, kernel_size=3, padding=1, bias=False),

            nn.ReLU()
        )
        self.conv32 = nn.Sequential(
            Conv(in_channels=512, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        )
        self.conv21 = nn.Sequential(
            Conv(in_channels=512, out_channels=256, kernel_size=3, padding=1, bias=False),

            nn.ReLU()
        )
        self.seg = Conv(in_channels=256, out_channels=3, kernel_size=3, padding=1, bias=False)

        if freeze_bn:
                self.freeze_bn()
    def forward(self, img):
        x1,x2,x3,x4  = self.backbone(img)
        x4 = self.conv4(x4)
        x3 = self.conv3(x3)
        x2 = self.conv2(x2)
        x4 = F.upsample(x4,scale_factor=2, mode='bilinear', align_corners=True)
        x43 = torch.cat([x4,x3],dim=1)
        x3 = self.conv43(x43)
        x3 = F.upsample(x3, scale_factor=2, mode='bilinear', align_corners=True)
        x32 = torch.cat([x3, x2], dim=1)
        x2 = self.conv32(x32)
        x2 = F.upsample(x2, scale_factor=2, mode='bilinear', align_corners=True)
        x21 = torch.cat([x2, x1], dim=1)
        x1 = self.conv21(x21)
        x1 = F.upsample(x1, img.size()[2:], mode='bilinear', align_corners=True)
        seg = self.seg(x1)
        return seg
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    import time
    model = DeepLab(backbone='resnet', output_stride=8)
    model = model.cuda()
    inputs = torch.rand(1, 3, 256, 256)
    inputs = inputs.cuda()
    t = time.time()
    for i in range(10):
        output = model(inputs)
        print(output.size())
        print(time.time() -t)
        t = time.time()