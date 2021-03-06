import glob
import cv2
from matplotlib import pyplot as plt
import json
import numpy as np
from skimage import io
class LineArea1(object):
    def __init__(self):
        self.dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/line_area/*.png'
        self.imgs = glob.glob(self.dr)
    def len(self):
        return len(self.imgs)

    def pull_item(self,ix):

        igs = self.imgs[ix]
        org = cv2.imread(igs)
        img = org[:,0:256,0]
        mask = org[:,256:512,0]
        return img, mask

class LineArea(object):
    def __init__(self):
        self.dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/land/*/*.json'
        self.imgs = glob.glob(self.dr)
        self.image_size = [256, 256]
        self.md = LineArea1()
    def len(self):
        return len(self.imgs)

    def draw_line(self, p):
        edges = list(p)
        edges.append(p[0])
        msk = np.zeros(shape=[self.image_size[0], self.image_size[1], 3], dtype=np.uint8)
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
                cv2.line(msk, tuple(p1), tuple(p2), (255, 255, 255), thickness=np.random.randint(1,3))
        return msk[:, :, 0:1]

    def pull_item(self,ix):
        if True:
            json_pth = self.imgs[ix]
            js_data = json.loads(open(json_pth).read())
            image_pth = json_pth.replace('.json', '.png')
            ig_data = io.imread(image_pth)[:, :, 0:3]

            lands = []
            for b in js_data:
                label = b['correction_type']
                if label == 'land':
                    points = b['boundary']
                    lands.append(points)
            if len(lands)<1:
                return None, None
            land_ix = np.random.randint(0, len(lands))
            line, mask = self.draw(lands[land_ix])

            d = np.where(line==255)

            k = np.random.choice(np.linspace(0, len(d[0])-1, num=len(d[0])), int(len(d[0])*np.random.randint(2,10)/10))
            k = np.asarray(k, np.int)
            x = d[0]
            x = x[k]
            y = d[1]
            y = y[k]
            line[x,y] =0
            line = np.expand_dims(line,-1)
            line = np.concatenate([ig_data- [123.15, 115.90, 103.06], line],axis=2)
            return np.transpose(line,(2,0,1)), mask

    def draw(self, points):
        p = []
        for pp in points:
            p.append([pp['x'], pp['y']])
        #line = np.zeros(shape=[self.image_size[0], self.image_size[1], 3], dtype=np.uint8)
        mask = np.zeros(shape=[self.image_size[0], self.image_size[1], 3], dtype=np.uint8)
        cv2.fillPoly(mask, np.asarray([p], np.int), (255, 255, 255))
        #cv2.polylines(line, np.asarray([p], np.int), True, (255, 255, 255), )
        line = self.draw_line(p)
        return line[:,:,0], mask[:,:,0]


if __name__ == '__main__':
    md = LineArea()
    print(md.len())
    ig,mk = md.pull_item(2)
    plt.subplot(121)
    plt.imshow(ig[:,:])
    plt.subplot(122)
    plt.imshow(mk)
    plt.show()


