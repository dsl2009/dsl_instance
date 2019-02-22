from layer.reseg import ReSeg
from layer.stacked_recurrent_hourglass import StackedRecurrentHourglass as SRecHg
from loss.dice import DiceLoss, DiceCoefficient
from loss.discriminative import DiscriminativeLoss
from torch import nn
from data import data_gen
import torch
import numpy as np
from torch import optim
n_class = 1

max_detect = 40

model = ReSeg(n_classes=n_class,pretrained=False)
model.cuda()

criterion_discriminative = DiscriminativeLoss(delta_var=0.5, delta_dist=1.0, norm=2, usegpu=True)
criterion_dice = DiceLoss(optimize_bg=True, smooth=1e-5)
criterion_mse = nn.MSELoss()

criterion_dice.cuda()
criterion_mse.cuda()

data_gener = data_gen.get_land_seg(batch_size=4, max_detect=40)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9,weight_decay=1e-5)





def run():
    for x in range(1000000):
        org_img, instance_mask, seg_mask, num_obj_org = next(data_gener)


        img = np.transpose(org_img, axes=[0, 3, 1, 2])
        instance_mask = np.transpose(instance_mask, axes=[0, 3, 1, 2])
        seg_mask = np.transpose(seg_mask, axes=[0, 3, 1, 2])
        num_obj = np.reshape(num_obj_org,(-1,1))

        img = torch.from_numpy(img)
        instance_mask = torch.from_numpy(instance_mask)
        seg_mask = torch.from_numpy(seg_mask)
        num_obj = torch.from_numpy(num_obj)

        img, instance_mask = torch.autograd.Variable(img.cuda()), torch.autograd.Variable(instance_mask.cuda())
        seg_mask, num_obj = torch.autograd.Variable(seg_mask.cuda()), torch.autograd.Variable(num_obj.cuda())

        sem_seg_out, ins_seg_out, ins_cls_out = model(img)

        discri_loss = criterion_discriminative(ins_seg_out,instance_mask,(max_detect*num_obj_org).astype(np.int32), max_detect)
        dice_loss = criterion_dice(sem_seg_out, seg_mask)
        mse_loss = criterion_mse(num_obj,ins_cls_out)

        totoal_loss =dice_loss+mse_loss+discri_loss

        optimizer.zero_grad()
        totoal_loss.backward()
        optimizer.step()
        print(discri_loss.item(),dice_loss.item(),mse_loss.item())
        if x%1000:
            torch.save(model.state_dict(), 'net1' + str(x) + '.pth')



run()