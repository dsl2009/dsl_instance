import torch
import numpy as np
import time
from sklearn.decomposition import PCA

def Pca(data):
    u, s, v = torch.svd(data.t())
    u = u[:,0:2]

    return torch.mm(data, u)

def cluster(data):
    data_te = torch.unsqueeze(data.t(), 0)
    data_e = torch.unsqueeze(data, 0)
    old_points = data
    offset = 1.0
    it = 0
    while offset > 1e-5:
        points = torch.unsqueeze(old_points, 2)
        distance = torch.sum(torch.pow((points - data_te) / 1.5, 2), dim=1)
        g_distance = torch.exp(-distance)
        num = torch.sum(torch.unsqueeze(g_distance, 2) * data_e, dim=1)
        denom = torch.sum(g_distance, dim=1)
        new_points = num / denom.unsqueeze(1)
        max_diff = torch.max(torch.sqrt(torch.sum(torch.pow(new_points - old_points, 2), dim=1)))
        offset = max_diff.item()
        old_points = new_points
        del distance,g_distance,num
        it+=1

    print(it)
    dt = new_points.cpu().detach().numpy()
    dt = dt[:, 0] / (dt[:, 1] * np.sum(dt, axis=1))
    dt = np.asarray(dt, np.float16)
    dt, label = np.unique(dt, return_inverse=True)
    return len(dt),label





if __name__ == '__main__':
    d = np.load('/home/dsl/PycharmProjects/dsl_instance/result/emd.npy')
    print(d.shape)
    d = PCA(n_components=2).fit_transform(d)
    d = torch.from_numpy(d).float().cuda()
    t = time.time()
    print(cluster(d))
    print(time.time()-t)




