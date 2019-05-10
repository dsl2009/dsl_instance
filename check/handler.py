import os
import glob
import  shutil
aim = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/fabuhui/'
f_d = os.path.join(aim,'951')
for d1 in os.listdir(f_d):
    c_d = os.path.join(f_d,d1)
    for ig in os.listdir(c_d):
        ig_path = os.path.join(c_d,ig)
        aim_path = os.path.join(aim,'aim','22_'+d1+'_'+ig)
        shutil.copy(ig_path,aim_path)
