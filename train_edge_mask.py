import torch
torch.cuda.set_device(1)
from models.seg_model import SegModel
from torch import optim
import torch
from data.data_gen import get_land_edge
import numpy as np
from torch.nn import functional as F
from matplotlib import pyplot as plt
from loss import dice
from torch.utils.tensorboard import SummaryWriter

import time

def get_lr(optimizers):
    for group in optimizers.param_groups:
        return float(group['lr'])
def run():
    EPS = 1e-12
    gen_mod = SegModel()
    gen_mod.cnn.load_state_dict(torch.load('/home/dsl/all_check/resnet50-19c8e357.pth'),strict=False)
    gen_mod.cuda()
    writer = SummaryWriter(log_dir='/home/dsl/all_check/pytorch_tb')

    criterion_dice = dice.DiceLoss(optimize_bg=True, smooth=1e-5)

    gen_optm = optim.SGD(gen_mod.parameters(), lr=0.01, momentum=0.9,weight_decay=0.00005)


    stm = optim.lr_scheduler.ReduceLROnPlateau(gen_optm,'min', factor=0.7, patience=5,verbose=True)


    gen = get_land_edge(batch_size=8, image_size=[256,256],output_size=[256,256])
    aver_loss = []
    for step in range(10000000):
        org_img, org_mask, org_edge_mask = next(gen)
        img = org_img[:, :, :, 0:3]

        point_mask = org_edge_mask
        count_neg = np.sum(1. - point_mask)
        count_pos = np.sum(point_mask)
        beta = count_neg / (count_neg + count_pos)
        pos_weight_point = beta / (1 - beta)

        img = np.transpose(img, axes=[0, 3, 1, 2])
        mask = np.transpose(org_mask, axes=[0, 3, 1, 2])
        edge_mask = np.transpose(org_edge_mask, axes=[0, 3, 1, 2])

        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        edge_mask = torch.from_numpy(edge_mask)

        data, mask_target, edge_target= torch.autograd.Variable(img.cuda()), torch.autograd.Variable(mask.cuda()), torch.autograd.Variable(edge_mask.cuda())
        mask_logits, edge_logits = gen_mod(data)

        mask_out = torch.sigmoid(mask_logits)
        edge_out = torch.sigmoid(edge_logits)

        mask_loss = F.binary_cross_entropy_with_logits(mask_logits, mask_target)
        mask_dice_loss = criterion_dice(mask_logits, mask_target)

        edge_loss = F.binary_cross_entropy_with_logits(edge_logits, edge_target, weight=torch.tensor(pos_weight_point).float())
        edge_dice_loss = criterion_dice(edge_logits, edge_target)

        be_loss = torch.sum(mask_out*edge_out)/torch.sum(edge_target)

        gen_loss = mask_loss +  mask_dice_loss+edge_loss+edge_dice_loss+be_loss

        aver_loss.append(gen_loss.item())

        gen_optm.zero_grad()
        gen_loss.backward()
        gen_optm.step()




        writer.add_scalar('gen_loss', gen_loss, step)
        writer.add_scalar('mask_loss', mask_loss, step)
        writer.add_scalar('mask_dice_loss', mask_dice_loss, step)
        writer.add_scalar('edge_loss', edge_loss, step)
        writer.add_scalar('edge_dice_loss', edge_dice_loss, step)
        writer.add_scalar('be_loss', be_loss, step)

        writer.add_scalar('lr', get_lr(gen_optm), step)
        writer.add_scalar('ave_loss', sum(aver_loss) / len(aver_loss), step)

        if step%100==0:
            writer.add_image('target_mask', torch.cat((mask_target, mask_target, mask_target), dim=1)[0],step)
            writer.add_image('pred_mask', torch.cat((mask_out, mask_out, mask_out), dim=1)[0],step)

            writer.add_image('target_edge', torch.cat((edge_target, edge_target, edge_target), dim=1)[0], step)
            writer.add_image('pred_edge', torch.cat((edge_out, edge_out, edge_out), dim=1)[0], step)

        if step % 1000 == 0 and step>2:
            stm.step(sum(aver_loss) / len(aver_loss))
        if step % 2000 == 0:
            torch.save(gen_mod.state_dict(), '/home/dsl/all_check/edge/' + str(step) + '_' + str(step) + '_tree.pth')

if __name__ == '__main__':
    run()