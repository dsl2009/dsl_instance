from layer.reseg import ReSeg
import glob
import torch
import numpy as np
from skimage import io
import os
from sklearn.cluster import KMeans,DBSCAN
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
import cv2
from result import instance_handler,utils
import json

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
result_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result'
n_class = 1

max_detect = 10

model = ReSeg(n_classes=n_class,pretrained=False,use_coordinates=True,num_filter=32)
model.load_state_dict(torch.load('../net_118000.pth'))
model.cuda()
model.eval()


def cluster(sem_seg_prediction, ins_seg_prediction,
            n_objects_prediction):
    print(ins_seg_prediction.shape)
    seg_height, seg_width = ins_seg_prediction.shape[1:]

    sem_seg_prediction = sem_seg_prediction*255
    sem_seg_prediction = sem_seg_prediction.astype(np.uint8)
    sem_seg_prediction = np.squeeze(sem_seg_prediction,0)
    sem_seg_prediction[sem_seg_prediction<150]=0

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

igs = cv2.imread('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/seg.jpg')
sp = igs.shape
mk = np.zeros(sp,dtype=np.uint8)
print(mk.shape)
def run():
    dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/land/d58200e3-2b29-4b99-b8ed-791031dd9b06'
    x_min, x_max, y_min, y_max = utils.get_xy(dr)
    result = dict()
    handler_num = 0
    with torch.no_grad():
        for x in glob.glob(os.path.join(dr,'*.png')):
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

            ins_cls_out = round(ins_cls_out.squeeze(0).item()*10)

            try:
                _, instance_mask, _ = cluster(sem_seg_out, ins_seg_out, ins_cls_out)
                rt = []
                lbs = []
                for x in range(ins_cls_out):
                    k = (instance_mask==(x+1))
                    k = k.astype(np.int32)
                    num = instance_handler.get_ct_num(tt,k,pading_x,pading_y)

                    #num , ct = instance_handler.get_counter(k,pading_x,pading_y)
                    if num>98:
                        lbs.append(x+1)
                if len(lbs) ==2:
                    instance_mask[np.where(instance_mask==lbs[0])] = lbs[1]
                    handler_num+=1

                for x in range(ins_cls_out):
                    k = (instance_mask==(x+1))
                    k = k.astype(np.int32)
                    num, ct = instance_handler.smooth_edge(tt,k,pading_x,pading_y)
                    #num , ct = instance_handler.get_counter(k,pading_x,pading_y)
                    rt.extend(ct)

                for k in ct:
                    instance_handler.draw_line(mk,k)

                result[str(int(pading_x/256))+'_'+str(int(pading_y/256))] = rt
                cl = color_pic(ins_cls_out+1, instance_mask)
                plt.subplot(131)
                plt.imshow(tt)
                plt.subplot(132)
                seg = (sem_seg_out[0,:,:]*255).astype(np.uint8)
                seg[seg<200]=0
                plt.imshow(seg)
                plt.subplot(133)
                plt.imshow(cl)

                cv2.imwrite('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/line/'+ig_name,tt)

            except:
                pass


        print('handler_num',handler_num)
        cv2.imwrite('edge.jpg',mk)
        plt.imshow(mk)
        plt.show()
        with open('result_handler.json','w') as f:
            print(len(result))
            f.write(json.dumps(result))
            f.flush()













run()