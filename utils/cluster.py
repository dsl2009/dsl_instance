from sklearn import cluster
from sklearn import metrics
import glob
from skimage import io
import json
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import shutil

def kms_cluster(data, num):
    km_mod = cluster.KMeans(n_clusters=num,n_jobs=6).fit(data)
    labels = km_mod.labels_
    print(data.shape)
    ls = metrics.silhouette_score(data, labels)
    return ls,labels,km_mod.cluster_centers_

def gen_data():
    nbs = 0
    rt = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/num'
    for json_pth in glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/land/*/*.json'):
        image_pth = json_pth.replace('.json', '.png')
        ig_data = io.imread(image_pth)[:, :, 0:3]
        instance_masks = []
        js_data = json.loads(open(json_pth).read())

        direct = np.zeros(shape=[256, 256, 3], dtype=np.uint8)
        num = 0
        for b in js_data:
            label = b['correction_type']
            if label == 'land':
                points = b['boundary']
                p = []

                for pp in points:
                    p.append([pp['x'], pp['y']])
                ct = np.asarray(p)
                ct = np.expand_dims(ct,1)

                if cv2.contourArea(ct)<30:
                    print(ct.shape, cv2.contourArea(ct))
                    print(ct)
                    nbs+=1
                    #cv2.fillPoly(direct, np.asarray([p], np.int), (255, 255, 255))
                    #plt.imshow(direct)
                    #plt.show()
                else:
                    num += 1




                #
                # cv2.polylines(direct,np.asarray([p], np.int),True, (255,255,255), thickness=2)
                #instance_masks.append(direct[:, :, 0:1])

        if num<11:
            dr = str(num)
            pth = os.path.join(rt, dr)
            if not os.path.exists(pth):
                os.makedirs(pth)
            #name = json_pth.split('/')[-1].split('.')[0]+'.jpg'

            shutil.copy(image_pth, pth)
            #io.imsave(os.path.join(pth,name),direct)



