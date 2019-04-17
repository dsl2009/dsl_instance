import torch
import time
class Mt(object):
    def tt(self,):
        t = time.time()
        a = torch.zeros(1000, 1000).cuda()
        print(time.time() - t)
