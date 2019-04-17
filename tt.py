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


a = [[0,0],[0.5,0.5],[0,1],[1,1],[1,0]]

pol = Polygon(a)
print(pol.convex_hull)
r = np.sqrt(pol.area/3.1415)/2
print(pol.is_simple)
print(r)
print(pol.area/pol.length)

