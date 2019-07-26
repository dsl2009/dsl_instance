import numpy as np
import cv2
from matplotlib import pyplot as plt
im = cv2.imread('/home/dsl/fsdownload/0f6c97d9-2948-4f91-be60-f3d98cfb41cf_seg.jpg')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):

    (x, y), radius = cv2.minEnclosingCircle(contours[i])
    center = (int(x), int(y))
    radius =radius
    if 6<radius<50:
        (x_start,y_start),(w,h),angele = cv2.minAreaRect(contours[i])
        if 0.7<w/h<1.5:
            cv2.circle(im, center, int(radius), (0, 255, 0), 2)
        else:
            x_offset = np.sin(angele/180*np.pi)*radius/2
            y_offset = np.cos(angele/180*np.pi)*radius/2
            x1, y1 = x+x_offset, y+y_offset
            x2, y2 = x - x_offset, y - y_offset
            print(x1,y1,x2, y2,radius)
            cv2.circle(im, (int(x1),int(y1)), int(radius/2), (0, 255, 0), 2)
            cv2.circle(im, (int(x2),int(y2)), int(radius/2), (0, 255, 0), 2)



cv2.imwrite('seg.jpg',im)

plt.imshow(im)
plt.show()
