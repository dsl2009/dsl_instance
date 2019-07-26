from layer.reseg import ReSeg
from matplotlib import pyplot as plt
from loss.dice import DiceLoss, DiceCoefficient
from loss.discriminative import DiscriminativeLoss
from torch import nn
from data import data_gen
import torch
import numpy as np
from torch import optim
import os
from torch.nn import functional as F
from models.widresnet_seg_64 import SegModel
import time
from torch.utils.tensorboard import SummaryWriter
from dsl_data import data_loader_multi
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
n_class = 1

max_detect = 60
#model = ReSeg(n_classes=n_class,pretrained=False,use_coordinates=True, num_filter=32)
model = SegModel()
model.cnn.load_state_dict(torch.load('/home/dsl/all_check/resnet50-19c8e357.pth'),strict=False)
model.cuda()
criterion_discriminative = DiscriminativeLoss(delta_var=0.5, delta_dist=2.0, norm=2, usegpu=True)
criterion_dice = DiceLoss(optimize_bg=True, smooth=1e-5)
criterion_mse = nn.MSELoss(reduction='sum')
criterion_be = nn.BCEWithLogitsLoss()
criterion_dice.cuda()
criterion_mse.cuda()
data_gener = data_gen.get_tree_seg(batch_size=8, max_detect=max_detect, output_size=[64,64])
q = data_loader_multi.get_thread(data_gener,3)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9,weight_decay=1e-5)
stm = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=60000, gamma=0.7)




def run():
    writer = SummaryWriter(log_dir='/home/dsl/all_check/tree')
    for x in range(10000000):
        org_img, instance_mask, seg_mask, num_obj_org = q.get()


        img = np.transpose(org_img, axes=[0, 3, 1, 2])
        instance_mask = np.transpose(instance_mask, axes=[0, 3, 1, 2])
        seg_mask = np.transpose(seg_mask, axes=[0, 3, 1, 2])
        num_obj = num_obj_org

        img = torch.from_numpy(img)
        instance_mask = torch.from_numpy(instance_mask)
        seg_mask = torch.from_numpy(seg_mask)
        num_obj = torch.from_numpy(num_obj)

        img, instance_mask = torch.autograd.Variable(img.cuda()), torch.autograd.Variable(instance_mask.cuda())
        seg_mask, num_obj = torch.autograd.Variable(seg_mask.cuda()), torch.autograd.Variable(num_obj.cuda())

        t = time.time()
        sem_seg_out, ins_seg_out = model(img)
        discri_loss = criterion_discriminative(ins_seg_out,instance_mask,(max_detect*num_obj_org).astype(np.int32), max_detect)
        dice_loss = criterion_dice(sem_seg_out, seg_mask)
        be_loss = F.binary_cross_entropy_with_logits(input=sem_seg_out, target=seg_mask)
        writer.add_scalar('dice_loss', dice_loss, x)
        writer.add_scalar('discri_loss', discri_loss, x)
        writer.add_scalar('be_loss', be_loss, x)
        totoal_loss =dice_loss+discri_loss+be_loss
        optimizer.zero_grad()
        totoal_loss.backward()
        optimizer.step()
        if x%2000==0:
            torch.save(model.state_dict(), '/home/dsl/all_check/tree/tree_64_19' + str(x) + '.pth')
        if x%1 == 0:
            t1 = time.time()-t
            print(t1, x, discri_loss.item(), dice_loss.item(),be_loss.item())
            t = time.time()
        stm.step(x)



run()