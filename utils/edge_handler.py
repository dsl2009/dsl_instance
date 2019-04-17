import numpy as np
import cv2
from skimage import measure
from result import shape_utils
from matplotlib import pyplot as plt

def gen_result():
    x_min = np.random.randint(0,100)
    x_max = np.random.randint(x_min+50,255)
    y_min = np.random.randint(0, 100)
    y_max = np.random.randint(y_min+50, 255)

    x1 = [x_min, np.random.randint(y_min+5,y_max-5)]
    x2 = [np.random.randint(x_min + 5, x_max - 5), y_min]
    x3 = [x_max, np.random.randint(y_min + 5, y_max - 5)]
    x4 = [np.random.randint(x_min + 5, x_max - 5), y_max]


    p_line1 = [np.random.randint(x_min, x_max),0]
    p_line2 = [int((x_min+x_max)/2)+np.random.randint(0,5),int((y_min+y_max)/2)+np.random.randint(0,5)]

    return [x1,x2,x3,x4],p_line1, p_line2






def gen_pic():
    pts, pline1, pline2 = gen_result()
    ig = np.zeros(shape=(256,256,3),dtype=np.uint8)
    cv2.fillPoly(ig,[np.asarray(pts)],(255, 255, 255))
    org = np.copy(ig)
    cv2.line(ig, pt1=tuple(pline1), pt2=tuple(pline2),color=(0,0,0),thickness=np.random.randint(2,5))
    return  ig[:,:,0],org[:,:,0]

if __name__ == '__main__':
    for x in range(100):
        gg ,g2 = gen_pic()
        plt.subplot(121)
        plt.imshow(gg)
        plt.subplot(122)
        plt.imshow(g2)
        plt.show()