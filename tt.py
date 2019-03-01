import numpy as np
from matplotlib import pyplot
from libKMCUDA import kmeans_cuda
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn import metrics,model_selection


def clus():
    data = np.load('/home/dsl/fsdownload/gaofen_X.npy')
    data = preprocessing.scale(data)
    data = np.asarray(data, np.float32)
    label = np.load('/home/dsl/fsdownload/gaofen_Y.npy')
    label = np.squeeze(label, 1).astype(np.int32)
    print(data)
    print(data.shape)
    print(data.max())
    centroids, assignments, avg_distance = kmeans_cuda(data, 3, tolerance=0.0001, init="k-means++",
                                                       yinyang_t=0.1, metric="L2", average_distance=True,
                                                       device=1, verbosity=1)
    print(assignments)
    print(avg_distance)
    print(label[0:20])
    print(assignments[0:20])
    print(metrics.cluster.homogeneity_score(label, assignments))
    print(metrics.cluster.completeness_score(label, assignments))
    print(metrics.cluster.homogeneity_completeness_v_measure(label, assignments))
    np.save('pred', assignments)
    X_train, X_test, Y_train, y_test = model_selection.train_test_split(data, label, test_size=0.3)
    knn = KNeighborsClassifier(n_neighbors=30)
    knn.fit(X_train, Y_train)
    print(knn.score(X_test, y_test))

label = np.load('/home/dsl/fsdownload/gaofen_Y.npy')
label = np.squeeze(label, 1).astype(np.int32)
pred =  np.load('pred.npy')
a = pred==label
print(np.sum(a.astype(np.int32))/label.shape[0])

print(pred==label)

print(label[0:20])
print(pred[0:20])


