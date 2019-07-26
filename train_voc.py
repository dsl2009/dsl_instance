import torch
from loss.dice import DiceLoss, DiceCoefficient
from loss.discriminative import DiscriminativeLoss
from torch import nn
from data import data_gen_new as data_gen
import torch
import numpy as np
from torch import optim
from torch.nn import functional as F
from models.widresnet_seg_voc import SegModel
import time


#torch.backends.cudnn.benchmark = True
print(torch.cuda.device_count())
n_class = 1

max_detect = 20

#model = ReSeg(n_classes=n_class,pretrained=False,use_coordinates=True, num_filter=32)
model = SegModel()
model.cnn.load_state_dict(torch.load('/home/dsl/all_check/resnet50-19c8e357.pth'),strict=False)
#model.load_state_dict(torch.load('/home/dsl/all_check/instance_land/net_res_64_3000.pth'))
model.cuda()
criterion_discriminative = DiscriminativeLoss(delta_var=0.5, delta_dist=2.0, norm=2, usegpu=True)
criterion_dice = DiceLoss(optimize_bg=True, smooth=1e-5)
criterion_mse = nn.MSELoss(reduction='sum')
criterion_be = nn.BCEWithLogitsLoss()
criterion_dice.cuda()
criterion_mse.cuda()
data_gener = data_gen.get_voc_detect(batch_size=8, max_detect=max_detect, image_size=[256,256], output_size=[256,256])
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9,weight_decay=1e-5)
stm = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=60000, gamma=0.7)




def run():

    for x in range(10000000):
        org_img, instance_mask, seg_mask, num_obj_org = next(data_gener)
        point_mask = seg_mask
        count_neg = np.sum(1. - point_mask)
        count_pos = np.sum(point_mask)
        beta = count_neg / (count_neg + count_pos)
        pos_weight_point = beta / (1 - beta)
        img = np.transpose(org_img, axes=[0, 3, 1, 2])
        instance_mask = np.transpose(instance_mask, axes=[0, 3, 1, 2])
        #seg_mask = np.transpose(seg_mask, axes=[0, 3, 1, 2])

        num_obj = num_obj_org

        img = torch.from_numpy(img)
        instance_mask = torch.from_numpy(instance_mask)
        seg_mask = torch.from_numpy(seg_mask)
        num_obj = torch.from_numpy(num_obj)

        img, instance_mask = torch.autograd.Variable(img.cuda()), torch.autograd.Variable(instance_mask.cuda())
        seg_mask, num_obj = torch.autograd.Variable(seg_mask.cuda()), torch.autograd.Variable(num_obj.cuda())


        t = time.time()
        sem_seg_out, ins_seg_out = model(img)

        sem_seg_out = sem_seg_out.permute(0,2,3,1)

        sem_seg_out = torch.reshape(sem_seg_out,shape=[8,65536,-1])
        seg_mask = torch.reshape(seg_mask,shape=[8,-1]).long()
        f_lbs = []
        logits = []
        for i in range(8):
            num_pos = torch.sum(seg_mask[i]>0)
            num_neg = num_pos*3
            pos_lbs = seg_mask[i][seg_mask[i]>0]
            neg_lbs = seg_mask[i][seg_mask[i]==0][:num_neg]
            pos_data = sem_seg_out[i][seg_mask[i]>0]
            neg_data = sem_seg_out[i][seg_mask[i]==0]
            prem = torch.randperm(neg_data.size(0))
            idx = prem[:num_neg]
            neg_data = neg_data[idx]
            dts = torch.cat([pos_data, neg_data],dim=0)
            lbs = torch.cat([pos_lbs.unsqueeze(-1), neg_lbs.unsqueeze(-1)],dim=0)
            logits.append(dts)
            f_lbs.append(lbs)
        logits = torch.cat(logits,dim=0)
        labels = torch.cat(f_lbs,dim=0).squeeze(1)


        discri_loss = criterion_discriminative(ins_seg_out,instance_mask,num_obj_org.astype(np.int32), max_detect)

        #dice_loss = criterion_dice(sem_seg_out, seg_mask)


        #be_loss = F.binary_cross_entropy_with_logits(input=sem_seg_out, target=seg_mask,weight=torch.tensor(pos_weight_point))

        #softmax_loss = F.binary_cross_entropy_with_logits(input=sem_seg_out, target=seg_mask)

        softmax_loss = F.cross_entropy(input=logits, target=labels)

        totoal_loss =softmax_loss+discri_loss

        optimizer.zero_grad()
        totoal_loss.backward()
        optimizer.step()

        if x%2000==0:
            torch.save(model.state_dict(), '/home/dsl/all_check/instance_land/land_res_edge_128_' + str(x) + '.pth')

        if x%1 == 0:
            t1 = time.time()-t
            print(t1, x, discri_loss.item(), softmax_loss.item())
            t = time.time()
        stm.step(x)



run()