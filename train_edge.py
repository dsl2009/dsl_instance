from torch import nn, optim
import torch
from torch.nn import functional as F
from data import data_gen
import numpy as np
import os
from matplotlib import pyplot as plt
import glob
import cv2
from models.edge_model import Generater
from loss import dice

def run():
    EPS = 1e-12
    gen_mod = Generater(4)
    #gen_mod.load_state_dict(torch.load('edge.pth'))
    criterion_dice = dice.DiceLoss(optimize_bg=True, smooth=1e-5)


    gen_mod.cuda()
    gen_optm = optim.SGD(gen_mod.parameters(), lr=0.01, momentum=0.9)
    gen_lr= optim.lr_scheduler.StepLR(gen_optm, step_size=10, gamma=0.1)
    gen = data_gen.get_edge_seg(16)
    for epoch in range(100):
        for step  in range(10000):
            org_img, org_mask = next(gen)
            img = torch.from_numpy(org_img)
            mask = torch.from_numpy(org_mask)

            data, target = torch.autograd.Variable(img.cuda()), torch.autograd.Variable(mask.cuda())
            p_logits, p_out_put = gen_mod(data)

            loc_loss = F.binary_cross_entropy_with_logits(p_logits, target)
            dic_loss = criterion_dice(p_logits, target)
            loss = loc_loss+dic_loss
            gen_optm.zero_grad()
            loss.backward()
            gen_optm.step()

            print(step, loc_loss.item(), dic_loss.item())
            if step%1000==0:
                ip = p_out_put.cpu().detach().numpy()
                plt.subplot(132)
                plt.title('org_mask')
                plt.imshow(org_mask[0, 0, :, :])
                plt.subplot(131)
                plt.title('org_img')
                plt.imshow(org_img[0,0,:,:])
                plt.subplot(133)
                plt.title('out_put')
                plt.imshow(ip[0,0,:,:])
                plt.savefig('dd.jpg')
                torch.save(gen_mod.state_dict(),'line2edge_new.pth')
        gen_lr.step(epoch)

def eval():
    EPS = 1e-12
    gen_mod = Generater()
    gen_mod.load_state_dict(torch.load('edge.pth'))
    gen_mod.cuda()
    gen_mod.eval()
    with torch.no_grad():
        for x in glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/edge/*.jpg'):
            igs = cv2.imread(x)
            re_igs = cv2.resize(igs, dsize=(64, 512))
            re_igs = re_igs[:,:,0:1]/255.0


            ig = np.expand_dims(re_igs, 0)
            ig = np.transpose(ig, axes=(0, 3, 1, 2))
            ig = torch.from_numpy(ig).float()
            data = torch.autograd.Variable(ig.cuda())
            p_logits, p_out_put = gen_mod(data)
            ip = p_out_put.cpu().detach().numpy()

            plt.subplot(121)
            plt.imshow(re_igs[ :, :,0])
            plt.subplot(122)
            plt.imshow(ip[0, 0, :, :])
            plt.show()





if __name__ == '__main__':
    run()