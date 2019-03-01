from layer.reseg import ReSeg
import glob
import torch
import numpy as np
from skimage import io
import os
from sklearn.cluster import KMeans,DBSCAN
from matplotlib import pyplot as plt
import time
import cv2
from result import instance_handler,utils,shape_utils
import json
from models.native_senet import se_resnext50_32x4d
from torch import nn
from torchvision import transforms
import json
from PIL import Image
from libKMCUDA import kmeans_cuda
from models.model_instance_dsl import InstanceModel


os.environ["CUDA_VISIBLE_DEVICES"] = '1'
result_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result'
n_class = 1

max_detect = 10

model = ReSeg(n_classes=n_class,pretrained=False,use_coordinates=False,num_filter=32)
print(model)
#model = InstanceModel()

#model.load_state_dict(torch.load('/home/dsl/all_check/instance_land/net3_168000.pth'))
model.load_state_dict(torch.load('../net2_406000.pth'))
model.cuda()
model.eval()


trans = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
labels = json.loads(open('/home/dsl/fsdownload/land.json').read())['label_ix']
lbs = dict()
for x in labels:
    lbs[labels[x]] = x
model_num = se_resnext50_32x4d(num_classes=1000, pretrained=None)
model_num.last_linear = nn.Linear(2048, 11)
model_num.load_state_dict(torch.load('/home/dsl/fsdownload/land.pth'), strict=True)
model_num.cuda()
model_num.eval()

def cluster(sem_seg_prediction, ins_seg_prediction,
            n_objects_prediction):
    print(ins_seg_prediction.shape)
    seg_height, seg_width = ins_seg_prediction.shape[1:]

    sem_seg_prediction = sem_seg_prediction*255
    sem_seg_prediction = sem_seg_prediction.astype(np.uint8)
    sem_seg_prediction = np.squeeze(sem_seg_prediction,0)
    sem_seg_prediction[sem_seg_prediction<100] = 0

    embeddings = ins_seg_prediction

    embeddings = embeddings.transpose(1, 2, 0)  # h, w, c


    embeddings = np.stack([embeddings[:, :, i][sem_seg_prediction != 0]
                           for i in range(embeddings.shape[2])], axis=1)

    #km = KMeans(n_clusters=n_objects_prediction,n_init=15, max_iter=500,n_jobs=-1).fit(embeddings)

    #labels = km.labels_

    centroids, labels = kmeans_cuda(embeddings, n_objects_prediction, tolerance=0.001, init="k-means++",
                yinyang_t=0.1, metric="L2", average_distance=False,
                device=1, verbosity=0)


    #labels = DBSCAN(eps=0.2,min_samples=100, n_jobs=-1).fit_predict(embeddings)


    instance_mask = np.zeros((seg_height, seg_width), dtype=np.uint8)

    fg_coords = np.where(sem_seg_prediction != 0)
    for si in range(len(fg_coords[0])):
        y_coord = fg_coords[0][si]
        x_coord = fg_coords[1][si]
        _label = labels[si]+1
        instance_mask[y_coord, x_coord] = _label

    return sem_seg_prediction, instance_mask, n_objects_prediction


def color_pic(_n_clusters,ins_seg_pred):
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, _n_clusters)]
    print(colors)
    ins_seg_pred_color = np.zeros(
        (256, 256, 3), dtype=np.uint8)
    for i in range(_n_clusters):
        ins_seg_pred_color[ins_seg_pred == (i+1) ] = (np.array(colors[i][:3]) * 255).astype('int')
    return ins_seg_pred_color

def get_num(pth):
    org = io.imread(pth)
    org = org[:, :, 0:3]

    ig = Image.fromarray(org)
    ig = trans(ig)
    ig = ig.unsqueeze(0)
    ig = torch.autograd.Variable(ig.cuda())
    _, output = model_num(ig)
    output = torch.softmax(output, 1)
    prop = output.cpu().detach().numpy()[0]
    idx = np.argmax(prop)
    pd = np.max(prop)
    return int(lbs[idx])

def run():

    dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/land/d58200e3-2b29-4b99-b8ed-791031dd9b06'
    for d in glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/line_edge/*.*'):
        os.remove(d)
    x_min, x_max, y_min, y_max = utils.get_xy(dr)
    result = dict()
    handler_num = 0
    with torch.no_grad():
        for x in sorted(glob.glob(os.path.join(dr,'*.png'))):
            current_loc = x.split('.')[0].split('/')[-1].split('_')
            x_offset, y_offset = int(current_loc[1]), int(current_loc[2])
            pading_x = (x_offset-x_min)*256
            pading_y = (y_offset-y_min)*256

            ig_name = x.split('/')[-1]
            ig_dr_name = x.split('/')[-2]
            dt_seg = os.path.join(result_dr,ig_dr_name+'_seg')
            dt_ins = os.path.join(result_dr, ig_dr_name + '_ins')
            if not os.path.exists(dt_ins):
                os.makedirs(dt_ins)
                os.makedirs(dt_seg)
            dt_ins = os.path.join(dt_ins,ig_name)
            dt_seg = os.path.join(dt_seg,ig_name)

            tt = cv2.imread(x)
            #tt = np.zeros(shape=(256,256,3),dtype=np.uint8)
            org_imgs = io.imread(x)[:,:,0:3]
            org_img = (org_imgs - [123.15, 115.90, 103.06]) / 225.0
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
            ins_cls_out = get_num(x)
            print(ins_cls_out,time.time() - t)
            if ins_cls_out>0:
                try:
                    _, instance_mask, _ = cluster(sem_seg_out, ins_seg_out, ins_cls_out)
                    print(time.time() - t)
                    rt = []
                    lbs_20 = []
                    lbs_98 = []
                    for i in range(ins_cls_out):
                        k = (instance_mask==(i+1))
                        k = k.astype(np.int32)
                        num = instance_handler.get_ct_num(tt,k,pading_x,pading_y)
                        print(num)
                        #num , ct = instance_handler.get_counter(k,pading_x,pading_y)
                        if num>20 and num<98:
                            lbs_20.append(i+1)
                        elif num>=98:
                            lbs_98.append(i+1)




                    if len(lbs_20) >1:
                        for i in range(1,len(lbs_20)):
                            instance_mask[np.where(instance_mask == lbs_20[i])] = lbs_20[0]
                            handler_num += 1

                    if len(lbs_98) >1:
                        for i in range(1,len(lbs_98)):
                            instance_mask[np.where(instance_mask == lbs_98[i])] = lbs_98[0]
                            handler_num += 1




                    for ix in range(ins_cls_out):
                        k = (instance_mask==(ix+1))
                        k = k.astype(np.int32)
                        if np.sum(k)>0:
                            #num, ct = instance_handler.smooth_edge(tt,k,pading_x,pading_y)
                            num , ct = instance_handler.get_counter(tt, k,pading_x,pading_y)
                            rt.extend(ct)



                    result[str(int(pading_x/256))+'_'+str(int(pading_y/256))] = rt
                    cl = color_pic(ins_cls_out+1, instance_mask)
                    plt.subplot(221)
                    plt.imshow(tt)
                    plt.subplot(222)
                    seg = (sem_seg_out[0,:,:]*255).astype(np.uint8)
                    seg[seg < 150] = 0
                    seg[seg > 150] = 255
                    plt.imshow(seg)
                    seg[np.where(tt[:, :, 0] > 0)] = 0
                    plt.subplot(223)
                    plt.imshow(seg)

                    plt.subplot(224)
                    plt.imshow(cl)
                    plt.show()
                    #plt.savefig('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/line_edge/'+ig_name)
                    cv2.imwrite('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/line_edge/'+ig_name,seg)
                except:
                    pass






        print('handler_num',handler_num)

        with open('result_handler2.json','w') as f:
            print(len(result))
            f.write(json.dumps(result))
            f.flush()













run()