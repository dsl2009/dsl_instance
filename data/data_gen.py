from dsl_data import aichanellger,BigLand,xair_guoshu
import random
import numpy as np
from matplotlib import pyplot as plt
from utils import edge_handler
from data.data_sets import LineArea
from data.voc import VOCDetection
from data import jingwei


def get_land(batch_size,is_shuff = True,max_detect = 80, image_size=[256,256],output_size = [64,64]):
    data_set = BigLand.BigLandMask(image_size=image_size, output_size=output_size)
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
                img, heatmap, regre, tag, mask, num_center = data_set.pull_item(idx[index])
                if regre.shape[0] ==0 or regre.shape[0]>max_detect:
                    index = index + 1
                    continue
            except:
                index = index+1
                continue
            img = img - [123.15, 115.90, 103.06]

            if b== 0:
                images = np.zeros(shape=[batch_size,image_size[0],image_size[1],3],dtype=np.float32)
                heatmaps = np.zeros(shape=[batch_size, 1, output_size[0], output_size[1]], dtype=np.float32)
                regres = np.zeros(shape=[batch_size, max_detect, 2], dtype=np.float32)
                tags = np.zeros(shape=[batch_size, max_detect], dtype=np.int)
                masks = np.zeros(shape=[batch_size, max_detect], dtype=np.int)
                num_centers = np.zeros(batch_size, dtype=np.int)
                images[b] = img
                heatmaps[b] = heatmap
                regres[b, 0:regre.shape[0], :] = regre
                tags[b, 0:regre.shape[0]] = tag
                masks[b,0:regre.shape[0]] = mask
                num_centers[b] = num_center
                b=b+1
                index = index + 1
            else:
                images[b] = img
                heatmaps[b] = heatmap
                regres[b,0:regre.shape[0],:] = regre
                tags[b,0:regre.shape[0]] = tag
                masks[b, 0:regre.shape[0]] = mask
                num_centers[b] = num_center
                b = b + 1
                index = index + 1
            if b>=batch_size:
                yield [images,heatmaps,regres,tags, masks,num_centers]
                b = 0

            if index>= data_set.len():
                print('hhh')
                index = 0

def get_land_seg(batch_size,is_shuff = True,max_detect = 80, image_size=[256,256],output_size = [64,64]):
    data_set = BigLand.BigLandArea(image_size=image_size)
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
                img,  instance_mask, seg_mask, object_num = data_set.pull_item(idx[index])
                if img is None:
                    index = index + 1
                    continue
                dd = np.sum(np.sum(instance_mask,axis=0),0)

                if object_num ==0 or object_num>max_detect:
                    index = index + 1
                    continue
            except:
                index = index+1
                continue
            img = (img - [123.15, 115.90, 103.06])/255.0
            instance_mask = instance_mask/255.0
            seg_mask = seg_mask / 255.0
            object_num_float = object_num/max_detect
            if b== 0:
                images = np.zeros(shape=[batch_size,image_size[0],image_size[1],3],dtype=np.float32)
                instance_masks = np.zeros(shape=[batch_size,output_size[0],output_size[1], max_detect], dtype=np.float32)
                seg_masks = np.zeros(shape=[batch_size,output_size[0],output_size[1], 1], dtype=np.float32)
                num_centers = np.zeros(batch_size, dtype=np.float32)

                images[b] = img
                instance_masks[b, :, :,0:object_num] = instance_mask
                seg_masks[b, :,:,0] = seg_mask
                num_centers[b] = object_num_float
                b=b+1
                index = index + 1
            else:
                images[b] = img
                instance_masks[b, :, :, 0:object_num] = instance_mask
                seg_masks[b, :, :, 0] = seg_mask
                num_centers[b] = object_num_float
                b = b + 1
                index = index + 1
            if b>=batch_size:
                yield [images,instance_masks,seg_masks,num_centers]
                b = 0

            if index>= data_set.len():
                print('hhh')
                index = 0


def get_tree_seg(batch_size,is_shuff = True,max_detect = 80, image_size=[256,256],output_size = [64,64]):
    data_set = xair_guoshu.Tree_mask_ins(root_dr='/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/biao_zhu/tree/',
                                         image_size=image_size,output_size=output_size)
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
                img,  instance_mask, seg_mask, object_num = data_set.pull_item(idx[index])
                if img is None:
                    index = index + 1
                    continue
                dd = np.sum(np.sum(instance_mask,axis=0),0)

                if object_num ==0 or object_num>max_detect:
                    index = index + 1
                    continue
            except:
                index = index+1
                continue
            img = (img - [123.15, 115.90, 103.06])/255.0
            instance_mask = instance_mask/255.0
            seg_mask = seg_mask / 255.0
            object_num_float = object_num/max_detect
            if b== 0:
                images = np.zeros(shape=[batch_size,image_size[0],image_size[1],3],dtype=np.float32)
                instance_masks = np.zeros(shape=[batch_size,output_size[0],output_size[1], max_detect], dtype=np.float32)
                seg_masks = np.zeros(shape=[batch_size,output_size[0],output_size[1], 1], dtype=np.float32)
                num_centers = np.zeros(batch_size, dtype=np.float32)

                images[b] = img
                instance_masks[b, :, :,0:object_num] = instance_mask
                seg_masks[b, :,:,0] = seg_mask
                num_centers[b] = object_num_float
                b=b+1
                index = index + 1
            else:
                images[b] = img
                instance_masks[b, :, :, 0:object_num] = instance_mask
                seg_masks[b, :, :, 0] = seg_mask
                num_centers[b] = object_num_float
                b = b + 1
                index = index + 1
            if b>=batch_size:
                yield [images,instance_masks,seg_masks,num_centers]
                b = 0

            if index>= data_set.len():
                print('hhh')
                index = 0


def get_edge_seg(batch_size):
    Md = LineArea()
    idx = list(range(Md.len()))
    b = 0
    index = 0
    while True:
        if True:
            if index >= Md.len():
                index = 0
            if index == 0:
                random.shuffle(idx)
            try:
                img,  instance_mask = Md.pull_item(index)
            except:
                index += 1
                continue

            if img is None:
                index+=1
                continue
            img = img/255.0
            instance_mask = instance_mask/255.0
            if b== 0:
                images = np.zeros(shape=[batch_size,4, 256, 256],dtype=np.float32)
                seg_masks = np.zeros(shape=[batch_size,1, 256, 256], dtype=np.float32)
                images[b, :, :,:] = img
                seg_masks[b, 0,:,:] = instance_mask
                b=b+1
            else:
                images[b, :, :, :] = img
                seg_masks[b, 0, :, :] = instance_mask
                b = b + 1
            index = index + 1

            if b>=batch_size:
                yield [images,seg_masks]
                b = 0
            if index>= Md.len():
                index = 0


def get_land_edge(batch_size,is_shuff = True,max_detect = 80, image_size=[256,256],output_size = [64,64]):
    data_set = BigLand.BigLand(image_size=image_size)
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
                img,  instance_mask, edge_mask = data_set.pull_item(idx[index])
                if img is None:
                    index = index + 1
                    continue
            except:
                index = index+1
                continue
            img = (img - [123.15, 115.90, 103.06])/255.0
            instance_mask = instance_mask/255.0
            edge_mask = edge_mask / 255.0

            if b== 0:
                images = np.zeros(shape=[batch_size,image_size[0],image_size[1],3],dtype=np.float32)
                instance_masks = np.zeros(shape=[batch_size,output_size[0],output_size[1], 1], dtype=np.float32)
                edge_masks = np.zeros(shape=[batch_size,output_size[0],output_size[1], 1], dtype=np.float32)

                images[b] = img
                instance_masks[b, :, :,0] = instance_mask
                edge_masks[b, :,:,0] = edge_mask
                b=b+1
                index = index + 1
            else:
                images[b] = img
                instance_masks[b, :, :, 0] = instance_mask
                edge_masks[b, :, :, 0] = edge_mask
                b = b + 1
                index = index + 1
            if b>=batch_size:
                yield [images,instance_masks,edge_masks]
                b = 0

            if index>= data_set.len():
                index = 0

def get_edge_seg_mask(batch_size):
    b = 0
    index = 0
    while True:
        if True:
            try:
                img,  instance_mask = edge_handler.gen_pic()
            except:
                continue
            img = img/255.0
            instance_mask = instance_mask/255.0
            if b== 0:
                images = np.zeros(shape=[batch_size,1, 256, 256],dtype=np.float32)
                seg_masks = np.zeros(shape=[batch_size,1, 256, 256], dtype=np.float32)
                images[b, 0, :,:] = img
                seg_masks[b, 0,:,:] = instance_mask
                b=b+1
            else:
                images[b, :, :, :] = img
                seg_masks[b, 0, :, :] = instance_mask
                b = b + 1
            if b>=batch_size:
                yield [images,seg_masks]
                b = 0

def get_voc_detect(batch_size,is_shuff = True,max_detect = 80, image_size=[256,256],output_size = [64,64]):
    image_dr = 'D:/deep_learn_data/VOCdevkit'
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
            true_num = instance_mask.shape[0]



            if b== 0:
                images = np.zeros(shape=[batch_size,image_size[0],image_size[1],3],dtype=np.float32)
                instance_masks = np.zeros(shape=[batch_size,output_size[0],output_size[1], max_detect], dtype=np.float32)
                seg_masks = np.zeros(shape=[batch_size,output_size[0],output_size[1]], dtype=np.float32)
                num_objs = np.zeros(batch_size,dtype=np.int)
                images[b] = img
                instance_masks[b, :, :,0:true_num] = instance_mask
                seg_masks[b, :,:] = seg_mask
                num_objs[b] = true_num
                b=b+1
                index = index + 1
            else:
                images[b] = img
                instance_masks[b, :, :, 0:true_num] = instance_mask
                seg_masks[b, :, :] = seg_mask
                num_objs[b] = true_num
                b = b + 1
                index = index + 1
            if b>=batch_size:
                yield [images,instance_masks,seg_masks]
                b = 0

            if index>= data_set.len():
                index = 0


def get_jingwei(batch_size,is_shuff = True, image_size=[512,512]):
    data_set = jingwei.Jinwei(image_size=512)
    idx = list(range(data_set.len()))
    b = 0
    index = 0
    while True:
        if True:
            if index>= data_set.len():
                index = 0
            if is_shuff and index==0:
                random.shuffle(idx)
            try:
                img,  seg_mask = data_set.pull_item(idx[index])
                if img is None or np.sum(seg_mask)==0:
                    index = index + 1
                    continue
            except:
                index = index+1
                continue
            img = (img - [123.15, 115.90, 103.06])/255.0
            seg_mask = seg_mask/255.0

            if b== 0:
                images = np.zeros(shape=[batch_size,image_size[0],image_size[1],3],dtype=np.float32)
                instance_masks = np.zeros(shape=[batch_size,image_size[0],image_size[1], 3], dtype=np.float32)
                images[b] = img
                instance_masks[b, :, :,:] = seg_mask
                b=b+1
                index = index + 1
            else:
                images[b] = img
                instance_masks[b, :, :, :] = seg_mask
                b = b + 1
                index = index + 1
            if b>=batch_size:
                yield [images,instance_masks]
                b = 0

            if index>= data_set.len():
                index = 0

if __name__ == '__main__':
    d = get_edge_seg(batch_size=4)
    for x in range(1000):
        images,seg_masks = next(d)
        plt.subplot(121)
        plt.imshow(images[0,0,:,:])
        plt.subplot(122)
        plt.imshow(seg_masks[0, 0, :, :])
        plt.show()