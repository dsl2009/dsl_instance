from models.seg_model import SegModel
from torch import optim
import torch
from data.data_gen import get_land_edge
import numpy as np
from torch.nn import functional as F
from matplotlib import pyplot as plt
import glob
import os
import cv2
from skimage import io
import os

def run():
    gen_mod = SegModel()
    gen_mod.cuda()
    gen_mod.load_state_dict(torch.load('/home/dsl/all_check/edge/66000_66000_tree.pth'))
    gen_mod.eval()

    dr = '/home/dsl/fsdownload/8acad878-9d03-4b1a-93e0-aa10143ef1ae'

    dd = glob.glob(os.path.join(dr, '*.png'))
    np.random.shuffle(dd)

    with torch.no_grad():
        for x in dd:
            print(x)
            ig_name = x.split('/')[-1]

            tt = cv2.imread(x)
            tt1 = np.zeros(shape=(256,256,3),dtype=np.uint8)
            org_imgs = io.imread(x)[:,:,0:3]
            org_img = (org_imgs - [123.15, 115.90, 103.06]) / 255.0
            org_img = np.expand_dims(org_img,0)
            img = np.transpose(org_img, axes=[0, 3, 1, 2])
            img = torch.from_numpy(img).float()
            img = torch.autograd.Variable(img.cuda())

            mask_logits,edge_logits = gen_mod(img)
            mask_out = torch.sigmoid(mask_logits)
            edge_out = torch.sigmoid(edge_logits)


            mask_out = mask_out.cpu().detach().numpy()
            edge_out = edge_out.cpu().detach().numpy()
            final_out = mask_out-edge_out

            final_out = final_out[0, 0, :, :]
            final_out = final_out*255
            final_out[np.where(final_out<128)]=0
            final_out[np.where(final_out >= 128)] = 255



            plt.subplot(221)
            plt.imshow(org_imgs)
            plt.subplot(222)
            plt.imshow(mask_out[0, 0, :, :])
            plt.subplot(223)
            plt.imshow(edge_out[0, 0, :, :])

            plt.subplot(224)
            plt.imshow(final_out)
            plt.show()

if __name__ == '__main__':
    run()