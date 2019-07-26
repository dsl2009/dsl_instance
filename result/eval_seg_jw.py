import os
import glob
import torch
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
import time
import cv2
import json
from torch.nn import functional as F
from data.jingwei import Jinwei
torch.cuda.set_device(0)
from deelab.deep_lab_faster import DeepLab
torch.backends.cudnn.benchmark = True

result_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result'
n_class = 1

max_detect = 20

# model = ReSeg(n_classes=n_class,pretrained=False,use_coordinates=False,num_filter=32)

model = DeepLab()
device= torch.device('cpu')
# model.load_state_dict(torch.load('/home/dsl/all_check/instance_land/net3_168000.pth'))
model.load_state_dict(torch.load('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/jingwei/'
                                 'jingwei_round1_train_20190619/log/land_deeplab_res50_19_88000.pth',map_location=device))
model.cuda()




def run():
    ave=[]
    av_loss = []
    dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/jingwei/jingwei_round1_train_20190619/test'
    md = Jinwei(image_size=512)
    dd = glob.glob(os.path.join(dr, '*.png'))
    np.random.shuffle(dd)
    for x in range(1000):
        # org_imgs = io.imread(x)[:, :, 0:3]
        org_imgs, lables = md.pull_item(x)

        org_img = (org_imgs - [123.15, 115.90, 103.06]) / 255.0
        org_img = np.expand_dims(org_img, 0)
        img = np.transpose(org_img, axes=[0, 3, 1, 2])
        img = torch.from_numpy(img).float()
        img = img.cuda()
        img = torch.autograd.Variable(img)

        t = time.time()
        out_put = model(img)

        sem_seg_out = torch.sigmoid(out_put)
        sem_seg_out = sem_seg_out.cpu().detach().numpy()
        sem_seg_out = np.squeeze(sem_seg_out, 0)
        sem_seg_out = np.transpose(sem_seg_out, (1, 2, 0))

        plt.subplot(131)
        plt.imshow(org_imgs)
        plt.subplot(132)
        plt.imshow(sem_seg_out)
        plt.subplot(133)
        plt.imshow(lables)

        plt.show()


if __name__ == '__main__':
    run()


