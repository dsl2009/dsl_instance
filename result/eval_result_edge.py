from layer.reseg import ReSeg
import glob
import torch
import numpy as np
from skimage import io
import os

from matplotlib import pyplot as plt
import time
import cv2
from result import instance_handler
from models.edge_model import Generater

from utils  import pytorch_cluster
from models.widresnet_seg import SegModel as InstanceModel
from sklearn.decomposition import PCA

torch.backends.cudnn.benchmark = True
imge_size = [256, 256]

#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model = InstanceModel()
#model.load_state_dict(torch.load('/home/dsl/all_check/instance_land/net3_168000.pth'))
#model.load_state_dict(torch.load('/home/dsl/fsdownload/land_edge_128_18_74000.pth'))
model.load_state_dict(torch.load('/home/dsl/fsdownload/land_edge_res50_19_78000.pth'))
model.cuda()
model.eval()

gen_mod = Generater(1)
gen_mod.load_state_dict(torch.load('../line2edge.pth'))
gen_mod.cuda()
gen_mod.eval()


def cluster(sem_seg_prediction, ins_seg_prediction):
    seg_height, seg_width = ins_seg_prediction.shape[1:]

    sem_seg_prediction = sem_seg_prediction*255
    sem_seg_prediction = sem_seg_prediction.astype(np.uint8)
    sem_seg_prediction = np.squeeze(sem_seg_prediction,0)
    sem_seg_prediction[sem_seg_prediction<128] = 0
    embeddings = ins_seg_prediction
    embeddings = embeddings.transpose(1, 2, 0)  # h, w, c
    embeddings = np.stack([embeddings[:, :, i][sem_seg_prediction != 0]
                           for i in range(embeddings.shape[2])], axis=1)
    pca = PCA(n_components=2).fit_transform(embeddings)
    t = time.time()
    nums, labels  = pytorch_cluster.cluster(torch.from_numpy(pca).cuda())
    print('gpu',time.time() - t)
    instance_mask = np.zeros((seg_height, seg_width), dtype=np.uint8)
    fg_coords = np.where(sem_seg_prediction != 0)
    for si in range(len(fg_coords[0])):
        y_coord = fg_coords[0][si]
        x_coord = fg_coords[1][si]
        _label = labels[si]+1
        instance_mask[y_coord, x_coord] = _label
    return sem_seg_prediction, instance_mask,  nums

def color_pic(_n_clusters,ins_seg_pred, ig_size):
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, _n_clusters)]

    ins_seg_pred_color = np.zeros(
        (ig_size[0], ig_size[1], 3), dtype=np.uint8)
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
    #dd = glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/land/1a5dafff-e4a3-42e8-8706-411164df0122/19_455459_183644.png')
    np.random.shuffle(dd)
    with torch.no_grad():
        for x in dd:
            print(x)
            #x='/home/dsl/fsdownload/826f264e-a602-4858-881f-9c68c6e9b294/18_208413_100570.png'
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
                    _, instance_mask,  ins_cls_out = cluster(sem_seg_out, ins_seg_out)
                    print(instance_mask.shape)

                    new_mask = np.zeros(shape=(256, 256),dtype=np.int)
                    for ix in range(ins_cls_out):
                        k = (instance_mask==(ix+1))
                        k = k.astype(np.uint8)
                        k = cv2.resize(k, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
                        k = k.astype(np.float)
                        ip_ig = np.expand_dims(k,-1)
                        #ip_ig = np.concatenate([(org_imgs-[123.15, 115.90, 103.06])/255.0, ip_ig],2)
                        ip_ig = np.expand_dims(np.transpose(ip_ig, (2,0,1)),0)

                        ip_ig = torch.from_numpy(ip_ig).float().cuda()
                        _, p_out_put = gen_mod(ip_ig)
                        ip = p_out_put.cpu().detach().numpy()
                        ip = ip[0, 0, :, :]
                        ip[np.where(ip>=0.6)] =1
                        ip[np.where(ip < 0.6)] = 0
                        k = np.copy(ip)
                        if np.sum(k)>0:
                            num , ct = instance_handler.get_counter(tt1, k,0,0)

                        new_mask = new_mask+ip*(ix+1)


                    cl = color_pic(ins_cls_out, new_mask, [256, 256])
                    cl2 = color_pic(ins_cls_out, instance_mask, [128, 128])
                    plt.subplot(221)
                    plt.imshow(tt)
                    plt.subplot(222)
                    seg = (sem_seg_out[0,:,:]*255).astype(np.uint8)
                    seg[seg < 150] = 0
                    seg[seg > 150] = 255
                    seg = cv2.resize(seg,dsize=(256,256), interpolation=cv2.INTER_NEAREST)
                    plt.imshow(seg)
                    plt.subplot(223)
                    plt.imshow(cl2)

                    plt.subplot(224)
                    cl = cv2.resize(cl, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
                    plt.imshow(cl)
                    plt.show()
                    plt.savefig('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/line_edge/'+ig_name)

                    #cv2.imwrite('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/line_edge/'+ig_name,seg)
                    #cv2.imwrite('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/line/' + ig_name, tt1)
                except:
                    pass




















run()