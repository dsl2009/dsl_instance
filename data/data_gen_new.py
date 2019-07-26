import random
import numpy as np
from matplotlib import pyplot as plt
from pretrainedmodels.models import resnext101_32x4d
from data.voc import VOCDetection

def get_voc_detect(batch_size,is_shuff = True,max_detect = 80, image_size=[256,256],output_size = [64,64]):
    image_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/VOCdevkit/VOCdevkit'
    data_set = VOCDetection(root=image_dr, is_crop=False, image_size=[256, 256])
    idx = list(range(data_set.len()))
    print(data_set.len())
    b = 0
    index = 0
    while True:
        if True:
            if index>= data_set.len():
                index = 0
            if is_shuff and index==0:
                random.shuffle(idx)
            try:
                img,  seg_mask, instance_mask = data_set.pull_item(idx[index])
                if img is None:
                    index = index + 1
                    continue
            except:
                index = index+1
                continue
            img = (img - [123.15, 115.90, 103.06])/255.0
            instance_mask = instance_mask/255.0
            true_num = instance_mask.shape[2]
            #seg_mask = seg_mask/255.0

            if true_num>20:
                index = index + 1
                continue


            if b== 0:
                images = np.zeros(shape=[batch_size,image_size[0],image_size[1],3],dtype=np.float32)
                instance_masks = np.zeros(shape=[batch_size,output_size[0],output_size[1], max_detect], dtype=np.float32)
                seg_masks = np.zeros(shape=[batch_size,output_size[0],output_size[1]], dtype=np.float32)
                num_objs = np.zeros(batch_size,dtype=np.int)
                images[b] = img
                instance_masks[b, :, :,0:true_num] = instance_mask
                seg_masks[b] = seg_mask
                num_objs[b] = true_num
                b=b+1
                index = index + 1
            else:
                images[b] = img
                instance_masks[b, :, :, 0:true_num] = instance_mask
                seg_masks[b] = seg_mask
                num_objs[b] = true_num
                b = b + 1
                index = index + 1
            if b>=batch_size:
                yield [images,instance_masks,seg_masks,num_objs]
                b = 0

            if index>= data_set.len():
                index = 0



