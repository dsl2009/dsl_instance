from layer.reseg import ReSeg
import glob
import torch
import numpy as np
from skimage import io
import os
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import cv2
from result import instance_handler,utils,shape_utils
from models.native_senet import se_resnext50_32x4d
from torch import nn
from torchvision import transforms
import json
from PIL import Image
from models.model_instance_dsl import InstanceModel
from libKMCUDA import kmeans_cuda
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
result_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result'
n_class = 1

max_detect = 10

#model = ReSeg(n_classes=n_class,pretrained=False,use_coordinates=True,num_filter=32)
#model.load_state_dict(torch.load('../net2_524000.pth'))

model = ReSeg(n_classes=n_class,pretrained=False,use_coordinates=True,num_filter=32)
#model = InstanceModel()

#model.load_state_dict(torch.load('/home/dsl/all_check/instance_land/net3_72000.pth'))
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
    sem_seg_prediction[sem_seg_prediction<150] = 0

    embeddings = ins_seg_prediction

    embeddings = embeddings.transpose(1, 2, 0)  # h, w, c


    embeddings = np.stack([embeddings[:, :, i][sem_seg_prediction != 0]
                           for i in range(embeddings.shape[2])], axis=1)
    if n_objects_prediction>1:
        centroids, labels = kmeans_cuda(embeddings, n_objects_prediction, tolerance=0.001, init="k-means++",
                                        yinyang_t=0.1, metric="L2", average_distance=False,
                                        device=1, verbosity=0)
    else:
        labels = np.zeros(embeddings.shape[0])

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

def run(task_drs,save_drs):


    for d in glob.glob(os.path.join(save_drs,'*.*')):
        os.remove(d)
    x_min, x_max, y_min, y_max = utils.get_xy(task_drs)
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
            org_img = (org_imgs - [123.15, 115.90, 103.06]) / 225.0
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

            ins_cls_out = get_num(x)

            if ins_cls_out>0:
                if True:
                    _, instance_mask, _ = cluster(sem_seg_out, ins_seg_out, ins_cls_out)
                    rt = []
                    lbs_20 = []
                    lbs_98 = []
                    for i in range(ins_cls_out):
                        k = (instance_mask==(i+1))
                        k = k.astype(np.int32)
                        num = instance_handler.get_ct_num(tt,k,pading_x,pading_y)
                        print(num)
                        #num , ct = instance_handler.get_counter(k,pading_x,pading_y)
                        if num>20 and num<60:
                            lbs_20.append(i+1)
                        elif num>=60:
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
                    #plt.show()
                    #plt.savefig('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/line_edge/'+ig_name)
                    cv2.imwrite(os.path.join(save_drs,ig_name),seg)


if __name__ == '__main__':
    task_name = 'eb2bec6b-9e6a-49d4-aee8-41f120fe5e59'
    root_dr = '/home/dsl/fsdownload/land'
    save_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/line_edge/'
    tmp_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result'

    for task_name in os.listdir('/home/dsl/fsdownload/land'):
        mask_dr = os.path.join(tmp_dr,task_name+'_seg.jpg')
        img_dr = os.path.join(tmp_dr,task_name+'.jpg')
        result_img_dr = os.path.join(tmp_dr,task_name+'_ok.jpg')

        task_dr = os.path.join(root_dr,task_name)
        run(task_dr,save_dr)
        utils.hebing_image(task_dr,img_dr)
        utils.hebing_image(save_dr,mask_dr,xy_root=task_dr)
        utils.get_counter(mask_dr,img_dr,result_img_dr)


















