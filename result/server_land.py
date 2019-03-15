from layer.reseg import ReSeg
import glob
import torch
import numpy as np
from skimage import io
import os
from sklearn.cluster import KMeans,DBSCAN,MeanShift
from matplotlib import pyplot as plt
import cv2
from result import instance_handler,utils_cv,shape_utils
from models.native_senet import se_resnext50_32x4d
from torch import nn
from torchvision import transforms
import json
from PIL import Image
from models.widresnet_seg_128 import SegModel
from utils import pytorch_cluster
import time
from sklearn.decomposition import PCA
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
result_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result'
n_class = 1

max_detect = 10

#model = ReSeg(n_classes=n_class,pretrained=False,use_coordinates=True,num_filter=32)
#model.load_state_dict(torch.load('../net2_524000.pth'))

model = SegModel()

model.load_state_dict(torch.load('/home/dsl/all_check/instance_land/net_res_128_30000.pth'))
model.cuda()
model.eval()



trans = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def cluster(sem_seg_prediction, ins_seg_prediction):

    seg_height, seg_width = ins_seg_prediction.shape[1:]

    sem_seg_prediction = sem_seg_prediction*255
    sem_seg_prediction = sem_seg_prediction.astype(np.uint8)
    sem_seg_prediction = np.squeeze(sem_seg_prediction,0)
    sem_seg_prediction[sem_seg_prediction<150] = 0

    embeddings = ins_seg_prediction

    embeddings = embeddings.transpose(1, 2, 0)  # h, w, c


    embeddings = np.stack([embeddings[:, :, i][sem_seg_prediction != 0]
                           for i in range(embeddings.shape[2])], axis=1)

    pca = PCA(n_components=2).fit_transform(embeddings)

    t = time.time()
    nums, labels = pytorch_cluster.cluster(torch.from_numpy(pca).cuda())
    print('gpu', time.time() - t)

    #labels = DBSCAN(eps=0.2,min_samples=100, n_jobs=-1).fit_predict(embeddings)


    instance_mask = np.zeros((seg_height, seg_width), dtype=np.uint8)

    fg_coords = np.where(sem_seg_prediction != 0)
    for si in range(len(fg_coords[0])):
        y_coord = fg_coords[0][si]
        x_coord = fg_coords[1][si]
        _label = labels[si]+1
        instance_mask[y_coord, x_coord] = _label

    return sem_seg_prediction, instance_mask, nums


def color_pic(_n_clusters,ins_seg_pred):
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, _n_clusters)]

    ins_seg_pred_color = np.zeros(
        (128, 128, 3), dtype=np.uint8)
    for i in range(_n_clusters):
        ins_seg_pred_color[ins_seg_pred == (i+1) ] = (np.array(colors[i][:3]) * 255).astype('int')
    return ins_seg_pred_color



def run(task_drs,save_drs):


    for d in glob.glob(os.path.join(save_drs,'*.*')):
        os.remove(d)
    x_min, x_max, y_min, y_max = utils_cv.get_xy(task_drs)
    result = dict()
    handler_num = 0
    with torch.no_grad():
        for x in sorted(glob.glob(os.path.join(task_drs,'*.png'))):
            current_loc = x.split('.')[0].split('/')[-1].split('_')
            x_offset, y_offset = int(current_loc[1]), int(current_loc[2])
            pading_x = (x_offset-x_min)*256
            pading_y = (y_offset-y_min)*256

            ig_name = x.split('/')[-1]
            ig_dr_name = x.split('/')[-2]
            #tt = cv2.imread(x)
            tt = np.zeros(shape=(256,256,3),dtype=np.uint8)
            org_imgs = io.imread(x)[:,:,0:3]
            org_img = (org_imgs - [123.15, 115.90, 103.06]) / 255.0
            org_img = np.expand_dims(org_img,0)
            img = np.transpose(org_img, axes=[0, 3, 1, 2])
            img = torch.from_numpy(img).float()
            img = torch.autograd.Variable(img.cuda())
            sem_seg_out, ins_seg_out = model(img)
            sem_seg_out = torch.sigmoid(sem_seg_out)
            sem_seg_out = sem_seg_out.cpu().detach().numpy()
            ins_seg_out = ins_seg_out.cpu().detach().numpy()
            ins_seg_out = np.squeeze(ins_seg_out,0)
            sem_seg_out = np.squeeze(sem_seg_out,0)
            try:
                _, instance_mask, ins_cls_out = cluster(sem_seg_out, ins_seg_out)
                rt = []
                lbs_20 = []
                lbs_98 = []


                for ix in range(ins_cls_out):
                    k = (instance_mask == (ix + 1))
                    k = k.astype(np.int32)
                    k = cv2.resize(k, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)

                    if np.sum(k) > 0:
                        # num, ct = instance_handler.smooth_edge(tt,k,pading_x,pading_y)
                        num, ct = instance_handler.get_counter(tt, k, pading_x, pading_y)
                        rt.extend(ct)

                result[str(int(pading_x / 256)) + '_' + str(int(pading_y / 256))] = rt
                cl = color_pic(ins_cls_out + 1, instance_mask)
                plt.subplot(221)
                plt.imshow(tt)
                plt.subplot(222)
                seg = (sem_seg_out[0, :, :] * 255).astype(np.uint8)
                seg = cv2.resize(seg, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
                seg[seg < 150] = 0
                seg[seg > 150] = 255
                plt.imshow(seg)
                seg[np.where(tt[:, :, 0] > 0)] = 0
                plt.subplot(223)
                plt.imshow(seg)
                plt.subplot(224)
                cl = cv2.resize(cl, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
                plt.imshow(cl)
                #plt.show()
                # plt.savefig('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/line_edge/'+ig_name)
                cv2.imwrite(os.path.join(save_drs, ig_name), seg)
            except:
                pass







if __name__ == '__main__':
    task_name = 'eb2bec6b-9e6a-49d4-aee8-41f120fe5e59'
    root_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/land'
    save_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/line_edge/'
    tmp_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result'

    for task_name in os.listdir('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/land'):
        mask_dr = os.path.join(tmp_dr,task_name+'_seg.jpg')
        img_dr = os.path.join(tmp_dr,task_name+'.jpg')
        result_img_dr = os.path.join(tmp_dr,task_name+'_ok.jpg')

        task_dr = os.path.join(root_dr,task_name)
        run(task_dr,save_dr)
        utils_cv.hebing_image(task_dr,img_dr)
        utils_cv.hebing_image(save_dr,mask_dr,xy_root=task_dr)
        utils_cv.get_counter(mask_dr,img_dr,result_img_dr)


















