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

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU()
        )
        self.up1 = Up(1024,512)
        self.up2 = Up(1024, 512)
        self.up3 = Up(1024, 512)
        self.up4 = Up(1024, 256)
        self.up5 = Up(512, 256)
        self.up6 = Up(256, 64)
        self.finnal = nn.ConvTranspose2d(in_channels=512, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.heats = nn.Sequential(
            CoordConv(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            CoordConv(in_channels=256, out_channels=1, kernel_size=1),
        )
        self.tages = nn.Sequential(
            CoordConv(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            CoordConv(in_channels=256, out_channels=8, kernel_size=1),
        )
        self.regres = nn.Sequential(
            CoordConv(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            CoordConv(in_channels=256, out_channels=2, kernel_size=1),
        )


    def forward(self, inputs, ip_tages):
        dw1 = self.down1(inputs)
        dw2 = self.down2(dw1)
        dw3 = self.down3(dw2)
        dw4 = self.down4(dw3)
        dw5 = self.down5(dw4)
        dw6 = self.down6(dw5)
        dw7 = self.down7(dw6)
        dw8 = self.down8(dw7)

        decov1 = self.decov(dw8)
        up = self.up1(dw7, decov1)
        up = self.up2(dw6, up)
        up = self.up3(dw5, up)
        up = self.up4(dw4, up)
        conv = self.up5(dw3, up)

        heat_maps = self.heats(conv)
        tages = self.tages(conv)
        regers = self.regres(conv)


        tages = tranpose_and_gather_feat(tages, ip_tages)
        regers = tranpose_and_gather_feat(regers, ip_tages)


        return heat_maps, tages, regers

def train():
    data_set = data_gen.get_land(8,max_detect=100)

    gen_mod = Generater()
    gen_mod.cuda()
    gen_optm = optim.Adam(gen_mod.parameters(), lr=0.002)
    gen_lr = optim.lr_scheduler.StepLR(gen_optm, step_size=30, gamma=0.5)

    for epoch in range(100):
        for step  in range(10000):
            try:
                org_img, heatmaps, regres, tags, masks, num_centers = next(data_set)
            except:
                continue
            img = np.transpose(org_img, axes=[0,3,1,2])
            img = torch.from_numpy(img)
            tags = torch.from_numpy(tags)
            masks = torch.from_numpy(masks)
            heatmaps_target = torch.from_numpy(heatmaps).float()
            regres = torch.from_numpy(regres).float()


            tags = torch.autograd.Variable(tags.cuda())
            masks = torch.autograd.Variable(masks.cuda())
            regres_target = torch.autograd.Variable(regres.cuda())
            data,heatmaps_target = torch.autograd.Variable(img.cuda()),torch.autograd.Variable(heatmaps_target.cuda())


            p_logits, tages, regers_pred = gen_mod(data, tags)

            clus_ls = cluster_loss(tages, masks)
            heatmap_pred, ls = loss.loss_function(p_logits, heatmaps_target, regers_pred, regres_target, masks)
            ls = ls+clus_ls

            gen_optm.zero_grad()
            ls.backward()
            gen_optm.step()


            if step%10==0:
                print(ls.item())
            pred_heat_map = heatmap_pred.cpu().detach().numpy()
            if step%500 ==0:
                plt.subplot(131)
                plt.imshow((org_img[0, :, :, :]+ [123.15, 115.90, 103.06])/225.0)
                plt.subplot(132)
                plt.imshow(heatmaps[0,0,:,:])
                plt.subplot(133)
                plt.imshow(pred_heat_map[0, 0, :, :])
                plt.savefig('d.jpg')
            if step%5000==0:
                torch.save(gen_mod.state_dict(),'net'+str(step)+'.pth')


train()