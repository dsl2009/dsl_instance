import cv2
from matplotlib import pyplot as plt
import numpy as np
import json
from result import utils
from skimage.measure import find_contours
from scipy.interpolate import splprep, splev

def get_counter(mask, pad_x, pad_y):
    cout = []
    mask[np.where(mask>0)] = 255
    mask = np.asarray(mask, np.uint8)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for x in contours:
        area = cv2.contourArea(x)
        if area>50:
            x = np.squeeze(x,axis=1)
            x = x+[pad_x,pad_y]
            x_ls = []
            for d in range(x.shape[0]):
                x_ls.append((x[d].tolist()))
            cout.append(x_ls)
    return len(contours), cout
def draw_tmp(img, data):

    cv2.polylines(img, np.asarray([data], np.int), True, (255, 255, 255), thickness=1)
    plt.imshow(img)
    plt.show()

def clc_angle(x1,x2, y ):
    dist = np.sqrt((y[0] - x1[0]) ** 2 + (y[1] - x1[1]) ** 2)
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y = np.asarray(y)
    x = x1 - y
    y = x2 - y



    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))

    cos_angle = x.dot(y) / (Lx * Ly)


    angle = np.arccos(cos_angle)
    angle2 = angle * 360 / 2 / np.pi
    angle2 = round(angle2,2)
    return dist, angle2




def smooth_edge1(imgs, mask, pad_x, pad_y):
    cout = []
    mask[np.where(mask>0)] = 255
    mask = np.asarray(mask, np.uint8)

    plt.imshow(mask)
    plt.show()
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sm = []
    for x in contours:
        area = cv2.contourArea(x)
        if area > 50:
            tp = []
            x = np.squeeze(x,1)
            left_index = np.argmin(x[:,0])
            print(np.argmin(x[:,0]),np.argmax(x[:, 0]),np.argmin(x[:, 1]),np.argmax(x[:, 1]))
            left = x[np.argmin(x[:,0])].tolist()
            right = x[np.argmax(x[:,0])].tolist()
            top = x[np.argmin(x[:, 1])].tolist()
            bottom = x[np.argmax(x[:, 1])].tolist()

            ct = [x[i].tolist() for i in range(x.shape[0])]
            l1 = ct[0:left_index]
            l2 = ct[left_index:]
            ct = l2+l1
            left_index ,top_index, right_index, bottom_index = ct.index(left),ct.index(top),ct.index(right),ct.index(bottom)
            print(left_index ,top_index, right_index, bottom_index)
            print([left,top,right,bottom])


            keep_index = [0]

            if left!=top:

                min_index = min(left_index,top_index)
                max_index = max(left_index,top_index)
                data = ct[min_index+1: max_index]
                x1 = ct[min_index]
                x2 = ct[max_index]

                x_st = ct[min_index]
                keep_index.append(min_index + 1)
                keep_index.append(max_index)
                for i in range(len(data) - 1):
                    cen = data[i]
                    end = data[i + 1]
                    ange = clc_angle(x_st, end, cen)
                    if ange < 120:
                        x_st = data[i]
                        keep_index.append(min_index + 2 + i)




            if top!=right:

                min_index = min(top_index,right_index)
                max_index = max(top_index,right_index)
                data = ct[min_index+1: max_index]
                x1 = ct[min_index]
                x2 = ct[max_index]
                x_st = ct[min_index]
                keep_index.append(min_index + 1)
                keep_index.append(max_index)
                for i in range(len(data) - 1):
                    cen = data[i]
                    end = data[i + 1]
                    ange = clc_angle(x_st, end, cen)
                    if ange < 120:
                        x_st = data[i]
                        keep_index.append(min_index + 2 + i)


            if right!=bottom:
                min_index = min(right_index,right_index)
                max_index = max(right_index,right_index)
                data = ct[min_index+1: max_index]
                x1 = ct[min_index]
                x2 = ct[max_index]
                x_st = ct[min_index]

                keep_index.append(min_index + 1)
                keep_index.append(max_index)

                for i in range(len(data) - 1):
                    cen = data[i]
                    end = data[i + 1]
                    ange = clc_angle(x_st, end, cen)
                    if ange < 120:
                        x_st = data[i]
                        keep_index.append(min_index + 2 + i)




            if bottom!=left:
                min_index = min(bottom_index,left_index)
                max_index = max(bottom_index,left_index)
                data = ct[min_index+1: max_index]
                x1 = ct[min_index]
                x2 = ct[max_index]
                x_st = ct[min_index]

                keep_index.append(min_index + 1)
                keep_index.append(max_index)

                for i in range(len(data) - 1):
                    cen = data[i]
                    end = data[i + 1]
                    ange = clc_angle(x_st, end, cen)
                    if ange < 120:
                        x_st = data[i]
                        keep_index.append(min_index+2+i)





            keep_index = sorted(keep_index)

            nn_edge = []
            for ix in keep_index:
                if ct[ix] not in nn_edge:
                    nn_edge.append(ct[ix])



            draw_tmp(imgs,nn_edge)

def draw_line(imgs, edges):
    d = list(edges)
    d.append(d[0])
    for i in range(len(d)-1):
        x1, y1 = d[i]
        x2, y2 = d[i+1]
        if (x1==x2 and (x1%256==0 or x1%256==255)) or (y1==y2 and (y1%256==0 or y1%256==255)):

            pass
        else:
            if (abs(x1-x2) >0 and abs(x1-x2)<4) and (min(x1,x2)%256==0 or max(x1,x2)%256==255):

                pass
            elif (abs(y1-y2) >0 and abs(y1-y2)<4) and (min(y1,y2)%256==0 or max(y1,y2)%256==255):

                pass
            else:
                cv2.line(imgs, tuple(d[i]), tuple(d[i+1]), color=(255,255,255), thickness=2)

    return imgs




def get_ct_num(imgs, mask, pad_x, pad_y):
    cout = []
    mask[np.where(mask > 0)] = 255
    mask = np.asarray(mask, np.uint8)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)


def smooth_edge(imgs, mask, pad_x, pad_y):
    cout = []
    mask[np.where(mask > 0)] = 255
    mask = np.asarray(mask, np.uint8)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sm = []
    for x in contours:
        area = cv2.contourArea(x)
        if area > 50:
            tp = []
            x = np.squeeze(x, 1)
            left_index = np.argmin(x[:, 0])
            left = x[np.argmin(x[:, 0])].tolist()
            right = x[np.argmax(x[:, 0])].tolist()
            top = x[np.argmin(x[:, 1])].tolist()
            bottom = x[np.argmax(x[:, 1])].tolist()

            ct = [x[i].tolist() for i in range(x.shape[0])]
            ct.append(ct[0])

            keep_index = [0]

            x_st = ct[0]

            for i in range(len(ct) - 1):
                cen = ct[i]
                end = ct[i + 1]
                dist, ange = clc_angle(x_st, end, cen)
                if ange < 120 or dist>10:
                    x_st = ct[i]
                    keep_index.append(i)


            keep_index = sorted(keep_index)

            nn_edge = []
            for ix in keep_index:
                if ct[ix] not in nn_edge:
                    nn_edge.append(ct[ix])
            draw_line(imgs,nn_edge)

            nn_edge = np.asarray(nn_edge)
            nn_edge = nn_edge+[pad_x,pad_y]

            nn_edge = [nn_edge[i].tolist() for i in range(nn_edge.shape[0])]
            sm.append(nn_edge)

    return len(contours), sm





def sub_conters():

    org = []
    with open('result.json') as f:
        data = json.loads(f.read())
    for k in range(len(data)):
        org.append(data[k][0])
        org.append(data[k][1])






def draw():
    org_pth = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/tm.jpg'
    org_img = cv2.imread(org_pth)
    tmp = np.zeros(org_img.shape,dtype=np.uint8)

    with open('result.json') as f:
        data = json.loads(f.read())
    for x in data:
        cv2.polylines(tmp, np.asarray([x], np.int), True, (255, 255, 255), thickness=2)
    cv2.imwrite('result.jpg',tmp)
    plt.imshow(tmp)
    plt.show()


def draw_ok_edge(data):

    org_pth = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/tm.jpg'
    org_img = cv2.imread(org_pth)
    for x in data:
        cv2.polylines(org_img, np.asarray([x], np.int), True, (255, 255, 255), thickness=3)
    cv2.imwrite('ok.jpg',org_img)
    plt.imshow(org_img)
    plt.show()

def get_attr(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dist = np.sqrt((y2-y1)**2+(x2-x1)**2)
    if dist>5 and (x2-x1==0 or y2-y1==0):
        if x2-x1 ==0:
            if x1%256==0 or x1%256==255:
                return {'x':x1,'start':min(y1,y2),'end':max(y1,y2)}
        else:
            if y1 % 256 == 0 or y1 % 256 == 255:
                return {'y':y1,'start':min(x1,x2),'end':max(x1,x2)}
    else:
        return None
        #print(k,p1,p2,dist)
def is_ok_edge(d):
    k = list(d)
    k.append(d[0])

    for i in range(len(k) - 1):
        ds = get_attr(k[i], k[i + 1])
        if ds:
            return False
    return True


def get_combine_loc(edge):
    fin_result = []
    for x in edge:
        rt = {'left': None, 'right': None, 'top': None, 'bootom': None, 'data':list(x)}
        x_copy = list(x)
        x_copy.append(x[0])
        for k in range(len(x_copy)-1):
            att = get_attr(x_copy[k],x_copy[k+1])
            if att is not None  and att.get('x') is not None:
                dk = att['x']
                if dk%256==0:
                    if rt['left'] is None:
                        rt['left'] = att
                    else:
                        if att['end'] - att['start'] > rt['left']['end'] - rt['left']['start']:
                            rt['left'] = att


                if dk%256==255:
                    if rt['right'] is None:
                        rt['right'] = att
                    else:
                        if att['end'] - att['start'] > rt['right']['end'] - rt['right']['start']:
                            rt['right'] = att



            if att is not None and att.get('y') is not None:
                dk = att['y']
                if dk % 256 == 0:
                    if rt['top'] is None:
                        rt['top'] = att
                    else:
                        if att['end'] - att['start'] > rt['top']['end'] - rt['top']['start']:
                            rt['top'] = att


                if dk % 256 == 255:
                    if rt['bootom'] is None:
                        rt['bootom'] = att
                    else:
                        if att['end'] - att['start'] > rt['bootom']['end'] - rt['bootom']['start']:
                            rt['bootom'] = att

        fin_result.append(rt)
    return fin_result



def combine_row(edge_x,edge_y):
    not_ok_result = []
    ok_result = []
    copy_edge_x = list(edge_x)
    copy_edge_y = list(edge_y)
    num = 0
    print(len(copy_edge_x),len(copy_edge_y))

    for index_x, x in enumerate(edge_x):
        for index_y, y in enumerate(edge_y):
            if x['bootom'] is not None and y['top'] is not None:
                dx = x['bootom']
                dy = y['top']
                if dx['y']+1 == dy['y']:


                    st1 = dx['start']
                    end1 = dx['end']

                    st2 = dy['start']
                    end2 = dy['end']

                    l1 = end1 -st1
                    l2 = end2 -st2

                    l = min(end1,end2) - max(st1, st2)


                    rate = l/max(l1,l2)

                    if rate>0.5:
                        #print('come', rate)
                        #print(x)
                        #print(y)

                        x1_point = [st1, dx['y']]
                        x2_point = [end1, dx['y']]

                        y1_point = [st2, dy['y']]
                        y2_point = [end2, dy['y']]
                        x_data = x['data']
                        y_data = y['data']
                        #print(x1_point,x2_point,y1_point,y2_point)
                        ix1, ix2 = x_data.index(x1_point),x_data.index(x2_point)
                        iy1, iy2 = y_data.index(y1_point), y_data.index(y2_point)
                        #print(ix1,ix2, iy1, iy2)
                        num =num+ 1

                        if ix2-ix1==1:
                            kl1 = x_data[:ix1+1]
                            kl1.reverse()
                            kl2 = x_data[ix2:]
                            kl2.reverse()
                            fin_l1 = kl1+kl2
                        elif ix2-ix1==-1:
                            kl1 = x_data[:ix2+1]
                            kl2 = x_data[ix1:]
                            fin_l1 =  kl2+kl1
                        elif ix2 -ix1>1:
                            x_data.reverse()
                            fin_l1 = x_data
                        else:
                            fin_l1 = x_data



                        if iy2 - iy1 == 1:
                            kl1 = y_data[:iy1+1]
                            kl2 = y_data[iy2:]
                            fin_l2 = kl2+kl1
                        elif iy2 -iy1 ==-1:
                            kl1 = y_data[:iy2+1]
                            kl2 = y_data[iy1:]
                            kl1.reverse()
                            kl2.reverse()

                            fin_l2 = kl1+kl2
                        elif iy2 -iy1<-1:
                            fin_l2 = y_data
                        else:
                            y_data.reverse()
                            fin_l2 = y_data


                        fin_result = fin_l1[1:-1] +fin_l2[1:-1]

                        copy_edge_x.remove(x)
                        copy_edge_y.remove(y)

                        if is_ok_edge(fin_result):

                            ok_result.append(fin_result)
                        else:
                            not_ok_result.append(fin_result)

    for x in copy_edge_x:
        if is_ok_edge(x['data']):
            ok_result.append(x['data'])
        else:
            not_ok_result.append(x['data'])
    for y in copy_edge_y:
        if is_ok_edge(y['data']):
            ok_result.append(y['data'])
        else:
            not_ok_result.append(y['data'])
    print(num,len(not_ok_result),len(ok_result))
    return not_ok_result, ok_result



def combine_col(edge_x,edge_y,num_idx):
    not_ok_result = []
    ok_result = []
    copy_edge_x = list(edge_x)
    copy_edge_y = list(edge_y)
    num = 0


    for index_x, x in enumerate(edge_x):
        for index_y, y in enumerate(edge_y):
            if x['right'] is not None and y['left'] is not None:
                dx = x['right']
                dy = y['left']
                if num_idx == 3:
                    draw_ok_edge([x['data']])
                    draw_ok_edge([y['data']])


                if dx['x']+1 == dy['x']:
                    st1 = dx['start']
                    end1 = dx['end']

                    st2 = dy['start']
                    end2 = dy['end']

                    l1 = end1 -st1
                    l2 = end2 -st2

                    l = min(end1,end2) - max(st1, st2)


                    rate = l/max(l1,l2)

                    if rate>0.5:

                        x1_point = [dx['x'],st1]
                        x2_point = [dx['x'],end1]

                        y1_point = [dy['x'],st2]
                        y2_point = [ dy['x'],end2]
                        x_data = x['data']
                        #draw_ok_edge([x_data])
                        y_data = y['data']
                        #draw_ok_edge([y_data])
                        '''
                        to_rm_x = []
                        for k in x_data:
                            if k[0] == x1_point[0] and k not  in [x1_point,x2_point]:
                                to_rm_x.append(k)
                        to_rm_y = []
                        for k in y_data:
                            if k[0] == y1_point[0] and k not in [y1_point, y2_point]:
                                to_rm_y.append(k)
                        for rmx in to_rm_x:
                            x_data.remove(rmx)
                        for rmx in to_rm_y:
                            y_data.remove(rmx)
                        print(to_rm_x,to_rm_y)
                        '''

                        ix1, ix2 = x_data.index(x1_point), x_data.index(x2_point)
                        iy1, iy2 = y_data.index(y1_point), y_data.index(y2_point)



                        #print(rate, ix1,ix2,iy1,iy2,len(x_data),len(y_data))
                        #print(x_data)
                        #print(y_data)
                        if ix2-ix1==1:
                            kl1 = x_data[:ix1+1]
                            kl1.reverse()
                            kl2 = x_data[ix2:]
                            kl2.reverse()
                            fin_l1 = kl1+kl2
                        elif ix2-ix1==-1:
                            kl1 = x_data[:ix2+1]
                            kl2 = x_data[ix1:]
                            fin_l1 =  kl2+kl1
                        elif ix2 -ix1>1:

                            fin_l1 = x_data
                        else:
                            x_data.reverse()
                            fin_l1 = x_data



                        if iy2 - iy1 == 1:
                            kl1 = y_data[:iy1+1]
                            kl2 = y_data[iy2:]
                            fin_l2 = kl2+kl1
                        elif iy2 -iy1 ==-1:
                            kl1 = y_data[:iy2+1]
                            kl2 = y_data[iy1:]
                            kl1.reverse()
                            kl2.reverse()

                            fin_l2 = kl1+kl2
                        elif iy2 -iy1<-1:
                            fin_l2 = y_data
                        else:
                            y_data.reverse()
                            fin_l2 = y_data

                        #print(fin_l1)
                        #print(fin_l2)
                        fin_result = fin_l1[1:-1] +fin_l2[1:-1]

                        copy_edge_x.remove(x)
                        copy_edge_y.remove(y)
                        if num_idx ==3:
                           draw_ok_edge([fin_result])
                        num+=1

                        if is_ok_edge(fin_result):
                            ok_result.append(fin_result)
                        else:
                            not_ok_result.append(fin_result)
    for x in copy_edge_x:
        if is_ok_edge(x['data']):
            ok_result.append(x['data'])
        else:
            not_ok_result.append(x['data'])
    for y in copy_edge_y:
        if is_ok_edge(y['data']):
            ok_result.append(y['data'])
        else:
            not_ok_result.append(y['data'])
    print('combine_num',num)
    return not_ok_result, ok_result






















def hebing_xy():
    dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/land/d58200e3-2b29-4b99-b8ed-791031dd9b06'
    x_min, x_max, y_min, y_max = utils.get_xy(dr)
    w = x_max-x_min+1
    h = y_max - y_min+1
    ok_result = []
    not_ok_result = []


    row = dict()


    with open('result_handler.json') as f:
        data = json.loads(f.read())

    for i in range(w):

        row_edge =None



        for j in range(h):
            idx = str(i)+'_'+str(j)
            if data.get(idx):

                if row_edge is not None:

                    not_ok_edges, ok_edges = combine_row(row_edge, get_combine_loc(data[idx]))
                    row_edge = get_combine_loc(not_ok_edges)
                    ok_result.extend(ok_edges)
                else:
                    row_edge = get_combine_loc(data[idx])

        not_ok_result.append(not_ok_edges)
        #draw_ok_edge(ok_result)
        #draw_ok_edge(not_ok_result)
    t1 = list(ok_result)
    t2 = []
    for x in not_ok_result:
        t2.extend(x)
    #t2 = list(not_ok_result)
    t1.extend(t2)
    #draw_ok_edge(t2)
    print(len(not_ok_result))



    row_edge = None
    for i in range(w):
        print('num',i)
        if row_edge is not None:
            nn =  get_combine_loc(not_ok_result[i])


            not_ok_edges, ok_edges = combine_col(row_edge,nn, -1)
            row_edge = get_combine_loc(not_ok_edges)
            '''
               for x in row_edge:
                if x.get('left') is not None:
                    print(x['data'])
                    draw_ok_edge([x['data']])
                if x.get('right') is not None and i==1:
                    print(x['data'])
                    draw_ok_edge([x['data']])
            '''
            #draw_ok_edge(not_ok_edges)
            ok_result.extend(ok_edges)
            print(len(ok_result),len(nn))

        else:
            row_edge = get_combine_loc(not_ok_result[i])


        #draw_ok_edge(ok_result)
        #draw_ok_edge(not_ok_edges)
    #ok_result.extend(not_ok_edges)
    for x in ok_result:
        b = list(x)
        b.append(x[0])
        st = b[0]

        for i in range(1,len(b)-2):
            dist, ang = clc_angle(st, b[i+1],b[i])
            st =  b[i]
            if ang<25 and dist>10:
                print(dist, ang)
                print(st, b[i+1],b[i])
                #draw_ok_edge([x])




    draw_ok_edge(not_ok_edges)






















def hebing_lines():
    kk = []
    with open('result.json') as f:
        data = json.loads(f.read())
    for k in range(len(data)):
        d = data[k]
        for i in range(len(d)-1):
            ds = get_attr(k,d[i],d[i+1])
            if ds:
                kk.append([ds,d])
    for i in range(len(kk)):
        for j in range(i+1,len(kk)):
            if abs(kk[i][0] - kk[j][0]) ==1:
                print(i,j, kk[i][1],kk[j][1])


    print(kk)






if __name__ == '__main__':

    hebing_xy()