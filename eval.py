from layer.reseg import ReSeg
import glob
import torch
import numpy as np
from skimage import io
import os
from sklearn.cluster import KMeans,DBSCAN
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
from models.native_senet import se_resnext50_32x4d
from torch import nn
from torchvision import transforms
import json
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
result_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result'
n_class = 1

max_detect = 10

model = ReSeg(n_classes=n_class,pretrained=False,use_coordinates=True,num_filter=32)
model.load_state_dict(torch.load('net2_44000.pth'))
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
    sem_seg_prediction[sem_seg_prediction<200]=0

    embeddings = ins_seg_prediction

    embeddings = embeddings.transpose(1, 2, 0)  # h, w, c


    embeddings = np.stack([embeddings[:, :, i][sem_seg_prediction != 0]
                           for i in range(embeddings.shape[2])], axis=1)

    labels = KMeans(n_clusters=n_objects_prediction,
                    n_init=35, max_iter=500,
                    n_jobs=-1).fit_predict(embeddings)


    #labels = DBSCAN(eps=0.2,min_samples=100, n_jobs=-1).fit_predict(embeddings)
    print(labels)

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
    with torch.no_grad():
        for x in glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/land/38220485-2708-4b9a-96f1-ebc9bbb1ba21/*.png'):

            ig_name = x.split('/')[-1]
            ig_dr_name = x.split('/')[-2]
            dt_seg = os.path.join(result_dr,ig_dr_name+'_seg')
            dt_ins = os.path.join(result_dr, ig_dr_name + '_ins')
            if not os.path.exists(dt_ins):
                os.makedirs(dt_ins)
                os.makedirs(dt_seg)
            dt_ins = os.path.join(dt_ins,ig_name)
            dt_seg = os.path.join(dt_seg,ig_name)



            org_imgs = io.imread(x)[:,:,0:3]
            org_img = (org_imgs - [123.15, 115.90, 103.06]) / 225.0
            org_img = np.expand_dims(org_img,0)
            img = np.transpose(org_img, axes=[0, 3, 1, 2])
            img = torch.from_numpy(img).float()
            img = torch.autograd.Variable(img.cuda())
            sem_seg_out, ins_seg_out, ins_cls_out = model(img)
            sem_seg_out = torch.sigmoid(sem_seg_out)
            sem_seg_out = sem_seg_out.cpu().detach().numpy()
            ins_seg_out = ins_seg_out.cpu().detach().numpy()
            ins_seg_out = np.squeeze(ins_seg_out,0)
            sem_seg_out = np.squeeze(sem_seg_out,0)
            print(ins_cls_out.squeeze(0).item()*10)
            ins_cls_out = get_num(x)
            print(ins_cls_out)
            try:
                _, instance_mask, _ = cluster(sem_seg_out, ins_seg_out, ins_cls_out)
                cl = color_pic(ins_cls_out+1, instance_mask)


                plt.subplot(241)
                plt.imshow(org_imgs)
                plt.subplot(242)
                seg = (sem_seg_out[0,:,:]*255).astype(np.uint8)
                seg[seg<200]=0
                plt.imshow(seg)


                plt.subplot(243)
                plt.imshow(ins_seg_out[0, :, :])
                plt.subplot(244)
                plt.imshow(ins_seg_out[1, :, :])
                plt.subplot(245)
                plt.imshow(ins_seg_out[2, :, :])
                plt.subplot(246)
                plt.imshow(ins_seg_out[3, :, :])
                plt.subplot(247)
                plt.imshow(ins_seg_out[4, :, :])
                plt.subplot(248)
                plt.imshow(cl)
                io.imsave(dt_seg, seg)
                io.imsave(dt_ins, cl)
                plt.show()
            except:
                pass










run()