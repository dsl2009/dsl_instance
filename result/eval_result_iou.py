from layer.reseg import ReSeg
import glob
import torch
import numpy as np
from skimage import io
import os
from sklearn.cluster import KMeans,DBSCAN,MeanShift
from matplotlib import pyplot as plt
import time
import cv2
from result import instance_handler,shape_utils, utils_cv
import json
from models.native_senet import se_resnext50_32x4d
from torch import nn
from torchvision import transforms
import json
from PIL import Image

from utils  import pytorch_cluster
from models.widresnet_seg import SegModel as InstanceModel
from sklearn.decomposition import PCA

torch.backends.cudnn.benchmark = True
imge_size = [128, 128]

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
result_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result'
n_class = 1

max_detect = 10

#model = ReSeg(n_classes=n_class,pretrained=False,use_coordinates=False,num_filter=32)

model = InstanceModel()

#model.load_state_dict(torch.load('/home/dsl/all_check/instance_land/net3_168000.pth'))
model.load_state_dict(torch.load('/home/dsl/fsdownload/land_edge_res50_19_90000.pth'))
model.cuda()
model.eval()


trans = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
'''
labels = json.loads(open('/home/dsl/fsdownload/land.json').read())['label_ix']
lbs = dict()
for x in labels:
    lbs[labels[x]] = x
model_num = se_resnext50_32x4d(num_classes=1000, pretrained=None)
model_num.last_linear = nn.Linear(2048, 11)
model_num.load_state_dict(torch.load('/home/dsl/fsdownload/land.pth'), strict=True)
model_num.cuda()
model_num.eval()

'''


def cluster(sem_seg_prediction, ins_seg_prediction):
    seg_height, seg_width = ins_seg_prediction.shape[1:]

    sem_seg_prediction = sem_seg_prediction*255
    sem_seg_prediction = sem_seg_prediction.astype(np.uint8)
    sem_seg_prediction = np.squeeze(sem_seg_prediction,0)
    sem_seg_prediction[sem_seg_prediction<50] = 0

    embeddings = ins_seg_prediction

    embeddings = embeddings.transpose(1, 2, 0)  # h, w, c


    embeddings = np.stack([embeddings[:, :, i][sem_seg_prediction != 0]
                           for i in range(embeddings.shape[2])], axis=1)
    #np.save('emd', embeddings)
    pca = PCA(0.95).fit_transform(embeddings)
    if pca.shape[1]<2:
        pca = PCA(n_components=2).fit_transform(embeddings)

    print(pca.shape)

    #np.save('emd', embeddings)
    #km = KMeans(n_clusters=n_objects_prediction,n_init=15, max_iter=500,n_jobs=-1).fit(embeddings)

    #labels = km.labels_
    '''
    
    centroids, labels ,dis= kmeans_cuda(embeddings, n_objects_prediction, tolerance=0.0001, init="k-means++",
                yinyang_t=0.1, metric="L2", average_distance=True,
                device=1, verbosity=0)
    '''


    #
    t = time.time()
    try:
        nums, labels  = pytorch_cluster.cluster(torch.from_numpy(pca).cuda())
    except:
        labels = MeanShift(bandwidth=0.8, n_jobs=-1).fit_predict(pca)
        nums=10

    print('gpu',time.time() - t)

    #nums =10
    #
    #labels = MeanShift(bandwidth=1.0, n_jobs=-1).fit_predict(pca)



    #

    instance_mask = np.zeros((seg_height, seg_width), dtype=np.uint8)

    fg_coords = np.where(sem_seg_prediction != 0)
    for si in range(len(fg_coords[0])):
        y_coord = fg_coords[0][si]
        x_coord = fg_coords[1][si]
        _label = labels[si]+1
        instance_mask[y_coord, x_coord] = _label

    return sem_seg_prediction, instance_mask,  nums


def color_pic(_n_clusters,ins_seg_pred):
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, _n_clusters)]

    ins_seg_pred_color = np.zeros(
        (imge_size[0], imge_size[1], 3), dtype=np.uint8)
    for i in range(_n_clusters):
        ins_seg_pred_color[ins_seg_pred == (i+1) ] = (np.array(colors[i][:3]) * 255).astype('int')
    return ins_seg_pred_color



def run():

    dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/land_19_bk/*'
    for d in glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/line_edge/*.*'):
        os.remove(d)
    for d in glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/line/*.*'):
        os.remove(d)
    result = dict()
    handler_num = 0
    dd = glob.glob(os.path.join(dr,'*.png'))
    np.random.shuffle(dd)
    total_iou = []
    with torch.no_grad():
        for x in dd[0:2000]:
            print(x)
            #x = '/home/dsl/fsdownload/add/2afcb628-108b-45a3-a9cd-e75739ebc793_seg/19_436266_210776.png'
            json_pth = x.replace('.png','.json')
            label_masks = []
            js_data = json.loads(open(json_pth).read())
            for b in js_data:
                label = b['correction_type']
                if label == 'land':
                    points = b['boundary']
                    p = []
                    for pp in points:
                        p.append([pp['x'], pp['y']])
                    direct = np.zeros(shape=[256, 256, 3], dtype=np.uint8)
                    cv2.fillPoly(direct, np.asarray([p], np.int), (255, 255, 255))
                    # cv2.polylines(direct,np.asarray([p], np.int),True, (255,255,255), thickness=2)
                    label_masks.append(direct[:, :, 0]/255)
            if len(label_masks)==0:
                continue

            ig_name = x.split('/')[-1]
            tt = cv2.imread(x)
            tt1 = np.zeros(shape=(256,256,3),dtype=np.uint8)
            org_imgs = io.imread(x)[:,:,0:3]
            org_img = (org_imgs - [123.15, 115.90, 103.06]) / 255.0
            org_img = np.expand_dims(org_img,0)
            img = np.transpose(org_img, axes=[0, 3, 1, 2])
            img = torch.from_numpy(img).float()
            img = torch.autograd.Variable(img.cuda())
            t = time.time()
            sem_seg_out, ins_seg_out = model(img)

            sem_seg_out = torch.sigmoid(sem_seg_out)
            sem_seg_out = sem_seg_out.cpu().detach().numpy()
            ins_seg_out = ins_seg_out.cpu().detach().numpy()
            ins_seg_out = np.squeeze(ins_seg_out,0)
            sem_seg_out = np.squeeze(sem_seg_out,0)
            t = time.time()


            if True:
                try:
                    pred_mask = []
                    _, instance_mask,  ins_cls_out = cluster(sem_seg_out, ins_seg_out)

                    for ix in range(ins_cls_out):
                        k = (instance_mask==(ix+1))
                        k = k.astype(np.int32)
                        k = cv2.resize(k, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
                        if np.sum(k)>0:
                            pred_mask.append(k)
                    for real_mask in label_masks:
                        max_iou = 0
                        for pd_mask in pred_mask:
                            plt.subplot(121)
                            plt.imshow(pd_mask)
                            plt.subplot(122)
                            plt.imshow(real_mask)
                            #plt.show()
                            tmp = np.sum(pd_mask*real_mask)/np.sum(real_mask)
                            if tmp>max_iou:
                                max_iou=tmp
                        total_iou.append(max_iou)
                    print('average_iou=', sum(total_iou) / len(total_iou))
                except:
                    pass
    print(total_iou)
    print('final_average_iou=',sum(total_iou)/len(total_iou))
























run()