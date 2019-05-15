import random
import numpy as np
from matplotlib import pyplot as plt
from data.voc import VOCDetection

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



if __name__ == '__main__':
    d = get_edge_seg(batch_size=4)
    for x in range(1000):
        images,seg_masks = next(d)
        plt.subplot(121)
        plt.imshow(images[0,0,:,:])
        plt.subplot(122)
        plt.imshow(seg_masks[0, 0, :, :])
        plt.show()