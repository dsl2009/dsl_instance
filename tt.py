import numpy as np
from matplotlib import pyplot
from libKMCUDA import kmeans_cuda
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn import metrics,model_selection
from visdom import Visdom
import numpy as np
import time
import torch
from torch import cuda

from utils import test_utils

from shapely.geometry import Point,Polygon

from skimage import io
import glob
from result import utils_cv
import cv2
utils_cv.get_right_counter('/home/dsl/fsdownload/b0d64daa-6eb8-4f4e-8a3b-36f90ec5091b_seg.jpg')
ig = np.zeros(shape=(256,256,3),dtype=np.uint8)

cv2.rectangle(ig,(20,20),(100,100),color=(255,255,255),thickness=-1)
pyplot.imshow(ig)
pyplot.show()

igs = ig[:,:,0]/255.0
print(np.sum(igs))
igs = cv2.dilate(igs,kernel=np.ones(shape=(5,5),dtype=np.uint8))
print(np.sum(igs))
pyplot.imshow(igs)
pyplot.show()

