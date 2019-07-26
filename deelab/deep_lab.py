import torch
import torch.nn as nn
import torch.nn.functional as F
from deelab.assp import build_aspp
from deelab.decoder import build_decoder
from deelab.resnet import resnet50

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=8, num_classes=1,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()

        BatchNorm = nn.BatchNorm2d

        self.backbone = resnet50()
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        if freeze_bn:
            self.freeze_bn()
    def forward(self, img):
        low_level_feat,x  = self.backbone(img)
        x = self.aspp(x)
        print(x.size())
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=img.size()[2:], mode='bilinear', align_corners=True)
        return x
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
    inputs = torch.rand(16, 3, 256, 256)
    inputs = inputs.cuda()
    t = time.time()
    for i in range(10):
        output = model(inputs)
        print(output.size())
        print(time.time() -t)
        t = time.time()