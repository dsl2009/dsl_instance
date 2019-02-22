import glob
import os
import json
import numpy as np
import cv2
from matplotlib import pyplot as plt
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8');
        return json.JSONEncoder.default(self, obj)


def get_xy(dr):
    x = []
    y = []
    for pth in glob.glob(os.path.join(dr, '*.png')):
        current_loc = pth.split('.')[0].split('/')[-1].split('_')
        c_x, c_y, c_z = int(current_loc[1]), int(current_loc[2]), int(current_loc[0])
        x.append(c_x)
        y.append(c_y)

    x_min, x_max, y_min, y_max = min(x), max(x), min(y), max(y)
    return x_min, x_max, y_min, y_max

def sub_conters():

    org = []
    with open('result1.json') as f:
        data = json.loads(f.read())
    print(data['3_7'])

def hebing_image(result_dr, sv_image):

    x_min, x_max, y_min, y_max = get_xy(result_dr)
    print(x_min, x_max, y_min, y_max)
    w = (x_max-x_min+1)*256
    h = (y_max-y_min+1)*256
    print(h,w)
    ig = np.zeros(shape=(h, w, 3),dtype=np.uint8)
    print(len(glob.glob(os.path.join(result_dr, '*.png'))))
    print((x_max-x_min)*(y_max-y_min))
    for pth in glob.glob(os.path.join(result_dr, '*.png')):
        img = cv2.imread(pth)
        current_loc = pth.split('.')[0].split('/')[-1].split('_')
        c_x, c_y, c_z = int(current_loc[1]), int(current_loc[2]), int(current_loc[0])

        start_x = c_x -x_min
        start_y = c_y - y_min
        print(start_y*256+256)
        ig[start_y*256:start_y*256+256, start_x*256:start_x*256+256,:] = img[:,:, 0:3]
    cv2.imwrite(sv_image,ig)


def get_remove():
    ig = cv2.imread('edge.jpg')
    ig = ig[:,:,0]
    ig2 = cv2.imread('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/seg.jpg')
    ig2 = ig2[:,:,0]
    ig2[np.where(ig==255)] = 0
    plt.imshow(ig2)
    plt.show()





if __name__ == '__main__':
    hebing_image('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/line','/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/line.jpg')
    #get_remove()