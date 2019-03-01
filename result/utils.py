import glob
import os
import json
import numpy as np
import cv2
from matplotlib import pyplot as plt
from result import instance_handler,shape_utils
from skimage import measure
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

def hebing_image(result_dr, sv_image,xy_root=None):
    if xy_root is None:
        x_min, x_max, y_min, y_max = get_xy(result_dr)
    else:
        x_min, x_max, y_min, y_max = get_xy(xy_root)

    w = (x_max-x_min+1)*256
    h = (y_max-y_min+1)*256
    ig = np.zeros(shape=(h, w, 3),dtype=np.uint8)
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
    ig = cv2.imread('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/line_edge.jpg')
    ig = ig[:,:,0]
    ig2 = cv2.imread('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/seg.jpg')
    ig2 = ig2[:,:,0]
    ig2[np.where(ig==255)] = 0
    cv2.imwrite('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/final.jpg',ig2)
    plt.imshow(ig2)
    plt.show()


def draw_final():
    img = cv2.imread('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/tm.jpg')
    mask = cv2.imread('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/line_edge1.jpg',0)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i, x in enumerate(contours):
        area = cv2.contourArea(x)
        if area > 50:
            cv2.drawContours(img,contours,i,color=(0,0,255),thickness=2)

    cv2.imwrite('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/tm_fin.jpg',img)

def get_counter(mask_dr, img_dr, save_dr):
    mask = cv2.imread(mask_dr, 0)
    couter = measure.find_contours(mask, level=1)
    img = cv2.imread(img_dr)
    for x in couter:
        b1 = x[:, 1:]
        b2 = x[:, 0:1]
        b = np.concatenate((b1, b2), axis=1)
        area = shape_utils.get_area_edge(b)
        if area>50:
            c = shape_utils.simlyfy(b,2)
            cv2.polylines(img, np.asarray([c], np.int), True, (0, 0, 255), thickness=1)
    cv2.imwrite(save_dr,img)

def convert():
    rt = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/land/d58200e3-2b29-4b99-b8ed-791031dd9b06'
    for x in glob.glob(os.path.join(rt,'*.png')):
        ig = cv2.imread(x)
        cv2.imwrite(os.path.join('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/dd',x.split('/')[-1]),ig)




def watershed():
    img  = cv2.imread('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/line_edge1.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    print(dist_transform.shape)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    plt.imshow(thresh)
    plt.show()



def handler():
    dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/land/d58200e3-2b29-4b99-b8ed-791031dd9b06'
    x_min, x_max, y_min, y_max = get_xy(dr)
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    print(w,h)
    ok_result = []
    not_ok_result = []

    row = dict()
    nm = '*_x_y.png'
    with open('result_handler2.json') as f:
        data = json.loads(f.read())

    for i in range(w):
        row_edge = None
        for j in range(h):
            idx = str(i) + '_' + str(j)
            print(idx)

            if data.get(idx):
                cux = x_min+i
                cuy = y_min+j

                cux = str(cux)
                cuy = str(cuy)
                nm_ew = nm.replace('x',cux)
                nm_ew = nm_ew.replace('y', cuy)

                pth = glob.glob(os.path.join(dr,nm_ew))
                pth = pth[0]

                ig = cv2.imread(pth)

                for k in data.get(idx):


                    instance_handler.jiaozheng(k)
                    #k = shape_utils.simlyfy(k,0.5)
                    k = np.asarray(k)
                    k = k-[i*256,j*256]

                    cv2.polylines(ig, np.asarray([k], np.int), True, (255, 255, 255), thickness=1)
                    plt.imshow(ig)
                    #plt.show()











if __name__ == '__main__':
    convert()