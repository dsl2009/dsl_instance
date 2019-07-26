import torch
import os
torch.cuda.set_device(1)

from torch import nn
from data import data_gen
from loss.dice import DiceLoss

import numpy as np
from torch import optim
from torch.nn import functional as F
from deelab.deep_lab_faster import DeepLab
import time
from torch.utils.tensorboard import SummaryWriter
from loss import loss
from dsl_data import data_loader_multi

criterion_dice = DiceLoss(optimize_bg=True, smooth=1e-5)
criterion_dice.cuda()


model = DeepLab(backbone='resnet', output_stride=8)
model.backbone.load_state_dict(torch.load('/home/dsl/all_check/resnet50-19c8e357.pth'),strict=False)
model = model.cuda()
criterion_mse = nn.MSELoss(reduction='sum')
data_gener = data_gen.get_jingwei(batch_size=4)
q = data_loader_multi.get_thread(data_gener,4)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=1e-4)
#stm = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100000, gamma=0.7)
#stm = optim.lr_scheduler.CyclicLR(optimizer,base_lr=0.001,max_lr=0.1,step_size_up=10000)
stm = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min', factor=0.5, patience=5,verbose=True)
def get_lr(optimizers):
    for group in optimizers.param_groups:
        return float(group['lr'])

def run():
    writer = SummaryWriter('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/jingwei/jingwei_round1_train_20190619/log/jingwei')
    ave_loss = []
    for x in range(1000000):

        org_img, seg_mask = q.get()
        img = np.transpose(org_img, axes=[0, 3, 1, 2])
        seg_mask = np.transpose(seg_mask, axes=[0, 3, 1, 2])
        img = torch.from_numpy(img)
        #instance_mask = torch.from_numpy(instance_mask)
        seg_mask = torch.from_numpy(seg_mask)
        #num_obj = torch.from_numpy(num_obj)
        img, seg_mask = torch.autograd.Variable(img.cuda()), torch.autograd.Variable(seg_mask.cuda())
        #seg_mask, num_obj = torch.autograd.Variable(seg_mask.cuda()), torch.autograd.Variable(num_obj.cuda())
        #torch.cuda.synchronize()

        sem_seg_out = model(img)
        dice_loss = criterion_dice(sem_seg_out, seg_mask)
        be_loss = F.binary_cross_entropy_with_logits(input=sem_seg_out, target=seg_mask)

        totoal_loss =be_loss+dice_loss
        writer.add_scalar('totoal_loss', totoal_loss, x)
        ave_loss.append(totoal_loss.item())
        writer.add_scalar('ave_loss', sum(ave_loss) / len(ave_loss), x)
        writer.add_scalar('lr', get_lr(optimizer), x)
        writer.add_scalar('be_loss', be_loss, x)
        writer.add_scalar('dice_loss', dice_loss, x)


        optimizer.zero_grad()
        totoal_loss.backward()
        optimizer.step()
        if x%2000==0 and x>2:
            lx = sum(ave_loss)/len(ave_loss)
            stm.step(lx)
            ave_loss = []
            torch.save(model.state_dict(), '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/jingwei/jingwei_round1_train_20190619/log'
                                           '/land_deeplab_res50_19_' + str(x) + '.pth')
        if x%1 == 0:
            t1 = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            print(t1, x,be_loss.item(),dice_loss.item())




run()