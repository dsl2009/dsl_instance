from skimage import io
import glob
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from PIL import Image
import os
import cv2
Image.MAX_IMAGE_PIXELS = 50000*80000

def gen_data(aim_size):
    dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/jingwei/jingwei_round1_train_20190619'
    image_dr = ['image_2.png']
    names = ['image2']
    for k in range(1):
        ig = io.imread(os.path.join(dr,image_dr[k]))
        h, w = ig.shape[0:2]
        ct_h = 0
        while ct_h < h:
            ct_w = 0
            while ct_w < w:
                offset_h = 0
                offset_w = 0
                if ct_w + aim_size < w and ct_h + aim_size < h:
                    new_ig = ig[ct_h:ct_h + aim_size, ct_w:ct_w + aim_size, :]
                elif ct_w + aim_size < w and ct_h + aim_size > h:
                    new_ig = np.zeros(shape=(aim_size, aim_size, 4), dtype=np.uint8)
                    new_ig[0:h - ct_h, :, :] = ig[ct_h:h, ct_w:ct_w + aim_size, :]
                    offset_h = h - ct_h
                elif ct_w + aim_size > w and ct_h + aim_size < h:
                    new_ig = np.zeros(shape=(aim_size, aim_size, 4), dtype=np.uint8)
                    new_ig[:, 0:w - ct_w, :] = ig[ct_h:ct_h + aim_size, ct_w:w, :]
                    offset_h = h - ct_h
                else:
                    new_ig = np.zeros(shape=(aim_size, aim_size, 4), dtype=np.uint8)
                    new_ig[0:h - ct_h, 0:w - ct_w, :] = ig[ct_h:h, ct_w:w, :]
                    offset_h = h - ct_h
                    offset_w = w - ct_w
                name = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/jingwei/jingwei_round1_train_20190619/images/'\
                       +names[k]+'_' + str(ct_h) + '_' + str(ct_w) + '_' + str(offset_h) + '_' + str(offset_w) + '_.png'
                io.imsave(name, new_ig)
                ct_w += aim_size
            ct_h += aim_size
        del ig

def gen_label(aim_size):
    dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/jingwei/jingwei_round1_train_20190619/'
    image_dr = ['image_1_label.png','image_2_label.png']
    names = ['image1','image2']
    for k in range(2):
        ig = io.imread(os.path.join(dr,image_dr[k]))
        h, w = ig.shape[0:2]
        ct_h = 0
        while ct_h < h:
            ct_w = 0
            while ct_w < w:
                offset_h = 0
                offset_w = 0
                if ct_w + aim_size < w and ct_h + aim_size < h:
                    new_ig = ig[ct_h:ct_h + aim_size, ct_w:ct_w + aim_size]
                elif ct_w + aim_size < w and ct_h + aim_size > h:
                    new_ig = np.zeros(shape=(aim_size, aim_size), dtype=np.uint8)
                    new_ig[0:h - ct_h, :] = ig[ct_h:h, ct_w:ct_w + aim_size]
                    offset_h = h - ct_h

                elif ct_w + aim_size > w and ct_h + aim_size < h:
                    new_ig = np.zeros(shape=(aim_size, aim_size), dtype=np.uint8)
                    new_ig[:, 0:w - ct_w] = ig[ct_h:ct_h + aim_size, ct_w:w]
                    offset_h = h - ct_h
                else:
                    new_ig = np.zeros(shape=(aim_size, aim_size), dtype=np.uint8)
                    new_ig[0:h - ct_h, 0:w - ct_w] = ig[ct_h:h, ct_w:w]
                    offset_h = h - ct_h
                    offset_w = w - ct_w
                name = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/jingwei/jingwei_round1_train_20190619/label/'\
                       +names[k]+'_' + str(ct_h) + '_' + str(ct_w) + '_' + str(offset_h) + '_' + str(
                    offset_w) + '_.png'
                io.imsave(name, new_ig)
                ct_w += aim_size
            ct_h += aim_size
        del ig


def remove_nu():
    for i in glob.glob(
            '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/jingwei/jingwei_round1_train_20190619/images/*.png'):
        ig = cv2.imread(i)
        if np.sum(ig) == 0:
            os.remove(i)
            os.remove(i.replace('images', 'label'))

class Jinwei(object):
    def __init__(self, image_size):
        self.data = glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/jingwei/'
                              'jingwei_round1_train_20190619/images/*.png')
        self.image_size = image_size
    def len(self):
        return len(self.data)
    def pull_item(self,idx):
        pth = self.data[idx]
        ig = io.imread(pth)[:,:,0:3]
        label = io.imread(pth.replace('images','label'))
        mask = np.zeros(shape=ig.shape,dtype=np.uint8)
        for i in range(3):
            mask[:,:,i][np.where(label==i+1)] =255

        h, w = ig.shape[0:2]
        pad_h = h - self.image_size
        pad_w = w - self.image_size
        offset_h = np.random.randint(0, pad_h)
        offset_w = np.random.randint(0, pad_w)
        new_image = ig[offset_h:offset_h + self.image_size, offset_w:offset_w + self.image_size, :]
        new_mask = mask[offset_h:offset_h + self.image_size, offset_w:offset_w + self.image_size, :]


        return new_image, new_mask

if __name__ == '__main__':
    #gen_data(aim_size=2048)
    #gen_label(aim_size=2048)
    #remove_nu()


    m = Jinwei(512)
    for j in range(100):
        ig, mask = m.pull_item(j)
        print(np.sum(mask))
        plt.subplot(221)
        plt.imshow(ig)
        plt.subplot(222)
        plt.imshow(mask[:,:,0])
        plt.subplot(223)
        plt.imshow(mask[:, :, 1])
        plt.subplot(224)
        plt.imshow(mask[:, :, 2])
        plt.show()




