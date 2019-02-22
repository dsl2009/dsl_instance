from layer.reseg import ReSeg
from layer.stacked_recurrent_hourglass import StackedRecurrentHourglass as SRecHg
from loss.dice import DiceLoss, DiceCoefficient
from loss.discriminative import DiscriminativeLoss
from torch import nn
from data import data_gen
import torch
import numpy as np
from torch import optim
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
n_class = 1

max_detect = 10

model = ReSeg(n_classes=n_class,pretrained=False,use_coordinates=True, num_filter=32)
model.cuda()

criterion_discriminative = DiscriminativeLoss(delta_var=0.5, delta_dist=1.0, norm=2, usegpu=True)
criterion_dice = DiceLoss(optimize_bg=True, smooth=1e-5)
criterion_mse = nn.MSELoss(reduction='sum')
criterion_be = nn.BCEWithLogitsLoss()

criterion_dice.cuda()
criterion_mse.cuda()

data_gener = data_gen.get_land_seg(batch_size=8, max_detect=10)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9,weight_decay=1e-5)





def run():
    for x in range(1000000):
        org_img, instance_mask, seg_mask, num_obj_org = next(data_gener)


        img = np.transpose(org_img, axes=[0, 3, 1, 2])
        instance_mask = np.transpose(instance_mask, axes=[0, 3, 1, 2])
        seg_mask = np.transpose(seg_mask, axes=[0, 3, 1, 2])
        #num_obj = np.reshape(num_obj_org,(-1,1))
        num_obj = num_obj_org

        img = torch.from_numpy(img)
        instance_mask = torch.from_numpy(instance_mask)
        seg_mask = torch.from_numpy(seg_mask)
        num_obj = torch.from_numpy(num_obj)

        img, instance_mask = torch.autograd.Variable(img.cuda()), torch.autograd.Variable(instance_mask.cuda())
        seg_mask, num_obj = torch.autograd.Variable(seg_mask.cuda()), torch.autograd.Variable(num_obj.cuda())

        sem_seg_out, ins_seg_out, ins_cls_out = model(img)

        discri_loss = criterion_discriminative(ins_seg_out,instance_mask,(max_detect*num_obj_org).astype(np.int32), max_detect)
        dice_loss = criterion_dice(sem_seg_out, seg_mask)
        #mse_loss = criterion_mse(num_obj,ins_cls_out)
        mse_loss = torch.pow((num_obj-torch.squeeze(ins_cls_out,1)),2).sum()
        be_loss = criterion_be(sem_seg_out,seg_mask)


        totoal_loss =dice_loss+discri_loss+be_loss

        optimizer.zero_grad()
        totoal_loss.backward()
        optimizer.step()

        if x%2000==0:
            torch.save(model.state_dict(), 'net2_' + str(x) + '.pth')
        if x%10 == 0:
            print(x, discri_loss.item(), dice_loss.item(), mse_loss.item(),be_loss.item())
            print(ins_cls_out.squeeze(1).cpu().detach().numpy())
            print(num_obj.cpu().detach().numpy())



run()