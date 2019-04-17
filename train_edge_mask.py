from models.model_instance_dsl import Generater
from torch import optim
import torch
from data.data_gen import get_land_edge
import numpy as np
from torch.nn import functional as F
from matplotlib import pyplot as plt
from loss import dice
import time
def run():
    EPS = 1e-12
    gen_mod = Generater()
    gen_mod.cuda()
    #gen_mod.cnn.load_state_dict(torch.load('/home/dsl/all_check/resnet50-19c8e357.pth'),strict=False)

    criterion_dice = dice.DiceLoss(optimize_bg=True, smooth=1e-5)

    gen_optm = optim.SGD(gen_mod.parameters(), lr=0.001, momentum=0.9)

    gen_lr= optim.lr_scheduler.StepLR(gen_optm, step_size=30, gamma=0.5)

    gen = get_land_edge(batch_size=8, image_size=[256,256],output_size=[256,256])

    for epoch in range(100):
        for step  in range(10000):
            org_img, org_mask, org_edge_mask = next(gen)
            img = org_img[:,:,:,0:3]

            point_mask = org_edge_mask
            count_neg = np.sum(1. - point_mask)
            count_pos = np.sum(point_mask)
            beta = count_neg / (count_neg + count_pos)
            pos_weight_point = beta / (1 - beta)



            img = np.transpose(img, axes=[0,3,1,2])
            mask = np.transpose(org_mask, axes=[0,3,1,2])
            edge_mask = np.transpose(org_edge_mask, axes=[0,3,1,2])



            img = torch.from_numpy(img)
            mask = torch.from_numpy(mask)
            edge_mask = torch.from_numpy(edge_mask)

            data, mask_target, edge_target = torch.autograd.Variable(img.cuda()), torch.autograd.Variable(mask.cuda()), torch.autograd.Variable(edge_mask.cuda())
            mask_logits, edge_logits = gen_mod(data)

            mask_out = torch.sigmoid(mask_logits)
            edge_out = torch.sigmoid(edge_logits)


            mask_loss = F.binary_cross_entropy_with_logits(mask_logits, mask_target)
            edge_loss = F.binary_cross_entropy_with_logits(edge_logits, edge_target,
                                                           weight=torch.tensor(pos_weight_point))

            mask_dice_loss = criterion_dice(mask_logits, mask_target)
            edge_dice_loss = criterion_dice(edge_logits, edge_target)





            gen_loss =  mask_loss+edge_loss+mask_dice_loss+edge_dice_loss
            gen_optm.zero_grad()
            gen_loss.backward()
            gen_optm.step()


            mask_out = mask_out.cpu().detach().numpy()
            edge_out = edge_out.cpu().detach().numpy()


            print(epoch,step,  gen_loss.item(), mask_loss.item(), edge_loss.item(), mask_dice_loss.item(), edge_dice_loss.item())
            if step%200==0:
                plt.subplot(221)
                plt.imshow(org_mask[0,:,:,0])
                plt.subplot(222)
                plt.imshow(mask_out[0, 0, :,:])
                plt.subplot(223)
                plt.imshow(org_edge_mask[0, :, :, 0])
                plt.subplot(224)
                plt.imshow(edge_out[0, 0, :, :])
                plt.savefig('dd.jpg')

            if step %2000 ==0:
                torch.save(gen_mod.state_dict(), '/home/dsl/all_check/edge/'+str(epoch)+'_'+str(step)+'_tree.pth')

        gen_lr.step(epoch)
if __name__ == '__main__':
    run()