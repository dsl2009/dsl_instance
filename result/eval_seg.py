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
torch.cuda.set_device(0)
from deelab.deep_lab_faster import DeepLab
torch.backends.cudnn.benchmark = True
imge_size = [128, 128]

result_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result'
n_class = 1

max_detect = 20

# model = ReSeg(n_classes=n_class,pretrained=False,use_coordinates=False,num_filter=32)

model = DeepLab()
device= torch.device('cpu')
# model.load_state_dict(torch.load('/home/dsl/all_check/instance_land/net3_168000.pth'))
model.load_state_dict(torch.load('/home/dsl/fsdownload/land_deeplab_res50_19_4000.pth',map_location=device))
model.cuda()
model.eval()


def draw(json_pth):
    instance_masks = []
    js_data = json.loads(open(json_pth).read())
    direct = np.zeros(shape=[256, 256, 3], dtype=np.uint8)
    for b in js_data:
        label = b['correction_type']
        if label == 'land':
            points = b['boundary']
            p = []
            for pp in points:
                p.append([pp['x'], pp['y']])
            cv2.fillPoly(direct, np.asarray([p], np.int), (255, 255, 255))
            edges = list(p)
            edges.append(p[0])
            for i in range(len(edges) - 1):
                p1 = edges[i]
                p2 = edges[i + 1]
                if p1[0] == 0 and p2[0] == 0:
                    pass
                elif p1[1] == 0 and p2[1] == 0:
                    pass
                elif p1[0] == 255 and p2[0] == 255:
                    pass
                elif p1[1] == 255 and p2[1] == 255:
                    pass
                else:
                    cv2.line(direct, tuple(p1), tuple(p2), (0, 0, 0), thickness=1)
    return direct


def run():
    ave=[]
    av_loss = []
    dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/land_18/*'
    for d in glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/line_edge/*.*'):
        os.remove(d)
    for d in glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/line/*.*'):
        os.remove(d)
    result = dict()
    handler_num = 0
    dd = glob.glob(os.path.join(dr, '18_*_*.png'))
    np.random.shuffle(dd)
    with torch.no_grad():
        for x in dd:
            #x='/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/land_18/2966ea4d-5e0a-45dc-9c09-4fdc4ae840da/18_218931_105455.png'
            if os.path.exists(x.replace('.png','.json')):
                mask = draw(x.replace('.png','.json'))
            else:
                continue
            # x = '/home/dsl/fsdownload/add/2afcb628-108b-45a3-a9cd-e75739ebc793_seg/19_436266_210776.png'
            ig_name = x.split('/')[-1]

            org_imgs = io.imread(x)[:, :, 0:3]
            org_img = (org_imgs - [123.15, 115.90, 103.06]) / 255.0
            org_img = np.expand_dims(org_img, 0)
            img = np.transpose(org_img, axes=[0, 3, 1, 2])
            img = torch.from_numpy(img).float()
            img = img.cuda()
            #img = torch.autograd.Variable(img)

            t = time.time()
            out_put = model(img)

            sem_seg_out = torch.sigmoid(out_put)
            sem_seg_out = sem_seg_out.cpu().detach().numpy()
            sem_seg_out = np.squeeze(sem_seg_out, 0)
            out = sem_seg_out[0].copy()
            out[np.where(out>=0.5)]=1.0
            out[np.where(out <0.5)] = 0
            mk = mask[:,:,0]/255.0
            mk_tensor = torch.from_numpy(mk).float().cuda()
            ls = F.binary_cross_entropy_with_logits(out_put[0,0,:,:],mk_tensor).item()



            lb = mk-out
            nb = len(np.where(lb==0)[0])
            ac = nb/256/256
            ave.append(ac)
            av_loss.append(ls)
            print(ls)
            #print(len(ave),sum(ave)/len(ave))
            print(len(av_loss), sum(av_loss) / len(av_loss))

            plt.subplot(221)
            plt.imshow(org_imgs)
            plt.subplot(222)
            plt.imshow(out)
            plt.subplot(223)
            plt.imshow(sem_seg_out[0])
            plt.subplot(224)
            plt.imshow(mask)
            plt.show()
            #if ac<0.7:
                #plt.savefig('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/error/'+ig_name)
if __name__ == '__main__':
    run()


