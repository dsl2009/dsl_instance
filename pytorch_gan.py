from torch import nn, optim
import torch
from torch.nn import functional as F
import numpy as np
import os
from matplotlib import pyplot as plt
import glob
from layer.coord_conv import CoordConv, CoordConvTranspose
from dsl_data.data_loader_multi import get_thread
from data.jingwei import Jinwei
from dsl_data import data_loader_multi
from data import data_gen
from loss.dice import DiceLoss

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super(Down, self).__init__()
        self.cnn = nn.Sequential(
            CoordConv(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=16,num_channels=out_c),
            nn.ReLU()
        )

    def forward(self, inputs):

        x = self.cnn(inputs)
        return x

class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super(Up, self).__init__()
        self.deconv = nn.Sequential(
            CoordConvTranspose(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1,bias=False),
            nn.GroupNorm(num_groups=16,num_channels=out_c),
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
        self.up5 = Up(512, 128)
        self.up6 = Up(256, 64)
        self.finnal = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1,bias=False)

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
        up = self.up5(dw3, up)
        up = self.up6(dw2, up)
        up = torch.cat([dw1, up],1)
        p_logits = self.finnal(up)
        return p_logits




def run():
    EPS = 1e-12

    criterion_dice = DiceLoss(optimize_bg=True, smooth=1e-5)
    criterion_dice.cuda()
    gen_mod = Generater()

    gen_optm = optim.SGD(gen_mod.parameters(), lr=0.01,momentum=0.9, weight_decay=1e-4)

    gen_mod.cuda()

    gen_lr= optim.lr_scheduler.StepLR(gen_optm, step_size=30, gamma=0.5)
    data_gener = data_gen.get_jingwei(batch_size=8)
    q = data_loader_multi.get_thread(data_gener, 4)
    for epoch in range(100):
        for step  in range(10000):
            org_img, org_mask = q.get()


            img = np.transpose(org_img, axes=[0,3,1,2])
            mask = np.transpose(org_mask, axes=[0,3,1,2])



            img = torch.from_numpy(img)
            mask = torch.from_numpy(mask)

            data, target = torch.autograd.Variable(img.cuda()), torch.autograd.Variable(mask.cuda())
            p_logits = gen_mod(data)
            p_out_put = torch.sigmoid(p_logits)

            loc_loss = F.binary_cross_entropy_with_logits(p_logits, target)


            gen_loss =  loc_loss
            gen_loss.backward()
            gen_optm.step()

            s_loss = gen_loss.cpu().detach().numpy()
            p_out_put = p_out_put.cpu().detach().numpy()
            p_out_put = np.transpose(p_out_put,(0,2,3,1))


            if step%1 ==0:
                print(epoch,  s_loss)
            if step%100==0:
                plt.subplot(121)
                plt.imshow(org_mask[0])
                plt.subplot(122)
                plt.imshow(p_out_put[0])
                plt.savefig('dd.jpg')

            if step %1000 ==0:
                torch.save(gen_mod.state_dict(), 'jingwei.pth')

        gen_lr.step(epoch)



def eval_test():
    from skimage import io
    EPS = 1e-12
    gen_mod = Generater()
    gen_mod.load_state_dict(torch.load('log/gen_3.pth'))
    gen_mod.cuda()

    with torch.no_grad():
        gen_mod.eval()
        for x in glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/zuixin/be224/57831739-d102-4f52-aabe-d90d5d0a49d0/*.png'):
            name = x.split('/')[-1]
            ig = io.imread(x)
            org = ig[:,:,0:3]
            ig = (org-  [123.15, 115.90, 103.06])/255.0
            ig = np.expand_dims(ig, 0)
            ig = np.transpose(ig, [0,3,1,2])
            img = torch.from_numpy(ig).float()
            data = torch.autograd.Variable(img.cuda())
            p_logits, p_out_put = gen_mod(data)
            out_put_msk = p_out_put.cpu().detach().numpy()

            plt.subplot(121)
            plt.imshow(org)
            plt.subplot(122)
            plt.imshow(out_put_msk[0, 1, :, :])
            io.imsave(os.path.join('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/land_test',name),out_put_msk[0, 1, :, :])



if __name__ == '__main__':
    run()





















