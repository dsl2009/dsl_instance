import glob
import os
import json
import numpy as np
import cv2
from matplotlib import pyplot as plt
from result import instance_handler,shape_utils
from skimage import measure
from models.edge_model import Generater
import torch
#device = torch.device('cpu')
#gen_mod = Generater(1)
#gen_mod.load_state_dict(torch.load('/home/dsl/PycharmProjects/dsl_instance/mask_optm.pth',map_location=device))
#gen_mod.cuda()
#gen_mod.eval()



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
        try:
            img = cv2.imread(pth)
            current_loc = pth.split('.')[0].split('/')[-1].split('_')
            c_x, c_y, c_z = int(current_loc[1]), int(current_loc[2]), int(current_loc[0])

            start_x = c_x -x_min
            start_y = c_y - y_min
            print(start_y*256+256)
            ig[start_y*256:start_y*256+256, start_x*256:start_x*256+256,:] = img[:,:, 0:3]
        except:
            pass
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

def remove_union(max_pol, min_pol):
    c = np.vstack((max_pol, min_pol))
    x_min, y_min, x_max, y_max = np.min(c[:, 0]), np.min(c[:, 1]), np.max(c[:, 0]), np.max(c[:, 1])
    h = y_max - y_min
    w = x_max - x_min
    ig_max = np.zeros(shape=(h,w ,3), dtype=np.uint8)
    ig_min = np.zeros(shape=(h,w ,3), dtype=np.uint8)

    max_pol = max_pol -[x_min, y_min]
    min_pol =min_pol -[x_min, y_min]

    cv2.fillPoly(ig_max, np.asarray([max_pol]), (255, 255, 255))
    cv2.fillPoly(ig_min, np.asarray([min_pol]), (255, 255, 255))
    ig_max = ig_max[:,:,0]
    ig_min = ig_min[:,:,0]
    ig_max[np.where(ig_min>0)] = 0
    cters, _ = cv2.findContours(ig_max, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cc in cters:
        if cv2.contourArea(cc) > 5000:
            cc = np.squeeze(cc, 1)
            cc = cc + [x_min, y_min]
            return cc

def spre(ig):
    thod = 0.1
    dist = cv2.distanceTransform(ig, cv2.DIST_L2, 3)
    max_distance = np.max(dist)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    _, dist = cv2.threshold(dist, thod, 1.0, cv2.THRESH_BINARY)
    kernel_size = max_distance*thod
    kernel_size = int(kernel_size)*1
    dist_8u = dist.astype('uint8')*255
    return dist_8u,kernel_size




def get_right_counter(mask_dr):
    mask = cv2.imread(mask_dr, 0)


    final_couter = []
    couter = measure.find_contours(mask, level=1)
    img = cv2.imread(mask_dr)
    for x in couter:
        b1 = x[:, 1:]
        b2 = x[:, 0:1]
        b = np.concatenate((b1, b2), axis=1)
        ppol = shape_utils.convert_poly(b)
        area = ppol.area
        length = ppol.length
        if area>3000:
            if area<12000:
                aim_size = (256,256)
            elif 12000<=area<250000:
                aim_size=(512, 512)
            else:
                aim_size=(1024, 1024)
            c = shape_utils.simlyfy(b,1)
            c = np.asarray(c, np.int)
            x_min, y_min, x_max, y_max = np.min(c[:, 0]),np.min(c[:, 1]),np.max(c[:, 0]),np.max(c[:, 1])

            ig_w = x_max - x_min
            ig_h = y_max - y_min
            sqr_w = max(ig_h, ig_w)
            mk = np.zeros(shape=(sqr_w, sqr_w, 3), dtype=np.uint8)
            c = c-[x_min, y_min]
            cv2.fillPoly(mk, np.asarray([c], np.int), (255, 255, 255))
            print(np.sum(mk[:,:,0]/255.0))

            ds, kernel_size = spre(mk[:,:,0])
            kernel11 = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            print('kernel=',kernel_size)
            counters, _ = cv2.findContours(ds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(counters)>1:
                plt.imshow(mk)
                plt.show()
                for i in range(len(counters)):
                    if cv2.contourArea(counters[i]) > 100:
                        tmp = np.zeros(shape=(sqr_w, sqr_w, 3), dtype=np.uint8)
                        cv2.drawContours(tmp, counters, i, color=(255,255,255),thickness=-1)
                        plt.imshow(tmp)
                        plt.show()

                        re_igs = tmp[:, :, 0] / 255.0
                        re_igs = cv2.dilate(re_igs, kernel11)
                        print(i, np.sum(re_igs))
                        plt.imshow(re_igs)
                        plt.show()
                        re_igs = cv2.resize(re_igs, dsize=aim_size)
                        re_igs = np.expand_dims(re_igs,-1)
                        print(re_igs.shape)
                        ip = smooth_edge(re_igs, sqr_w)
                        plt.imshow(ip)
                        plt.show()
                        cters, _ = cv2.findContours(ip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for c in cters:
                            if cv2.contourArea(c) > 1000:
                                c = np.squeeze(c, 1)
                                c = c + [x_min, y_min]
                                final_couter.append(c)

            else:
                re_igs = cv2.resize(mk,dsize=aim_size)
                re_igs = re_igs[:, :, 0:1] / 255.0
                ip = smooth_edge(re_igs, sqr_w)
                cters, _ = cv2.findContours(ip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cters:
                    if cv2.contourArea(c) > 2500:
                        c = np.squeeze(c, 1)
                        c = c + [x_min, y_min]
                        final_couter.append(c)

    totoal_len = len(final_couter)
    to_remove = []
    for  i in range(totoal_len):
        for j in range(i+1, totoal_len):
            try:
                pol1 = shape_utils.convert_poly(final_couter[i])
                pol2 = shape_utils.convert_poly(final_couter[j])
                inter = pol1.intersection(pol2)
                if inter.area>0:
                    if pol1.area > pol2.area:
                        if inter.area == pol2.area:
                            to_remove.append(j)
                        else:
                            final_couter[i] = remove_union(final_couter[i], final_couter[j])
                    else:
                        if inter.area == pol2.area:
                            to_remove.append(i)
                        else:
                            final_couter[j] = remove_union(final_couter[j], final_couter[i])
            except:
                pass
    for i in to_remove:
        final_couter[i] = None

    return final_couter


def smooth_edge(re_igs, sqr_w):
    ig = np.expand_dims(re_igs, 0)
    ig = np.transpose(ig, axes=(0, 3, 1, 2))
    ig = torch.from_numpy(ig).float()
    data = torch.autograd.Variable(ig.cuda())
    p_logits, p_out_put = gen_mod(data)
    ip = p_out_put.cpu().detach().numpy()
    ip = np.squeeze(ip, axis=(0, 1))
    ip = cv2.resize(ip, dsize=(sqr_w, sqr_w))
    ip = np.asarray(ip * 255, np.uint8)
    ip[np.where(ip >= 50)] = 255
    ip[np.where(ip < 50)] = 0
    return ip


def draw_edge(final_couter, image_dr):
    ig = cv2.imread(image_dr)
    for x in final_couter:
        if x is not None:
            cv2.polylines(ig, np.asarray([x], np.int), True, (0, 0, 255), thickness=1)
    cv2.imwrite(image_dr.replace('.jpg','_fin.jpg'),ig)


def draw_corner():
    jpth = json.loads(open('/home/dsl/fsdownload/e9e13b1b-8fa2-4c7a-982b-6ee6ee72a57a.json').read())
    img = cv2.imread('ddd.jpg')
    for d in jpth['boundarys']:
        point = []
        for p in d['boundary']:
            point.append([p['x'], p['y']])
        cv2.polylines(img, np.asarray([point], np.int), True, (0, 0, 255), thickness=1)
    cv2.imwrite('seg.jpg',img)
    plt.imshow(img)
    plt.show()


def draw_circle():
    i = json.loads(open('/home/dsl/fsdownload/5304_2908.txt'))
    print(i)



















if __name__ == '__main__':
    #hebing_image('/home/dsl/fsdownload/e9e13b1b-8fa2-4c7a-982b-6ee6ee72a57a','ddd.jpg')
    draw_corner()