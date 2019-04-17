from result import instance_handler,utils_cv,shape_utils
from models.model_instance_dsl import Generater
import torch
import numpy as np
from matplotlib import pyplot as plt
import glob
import os
import cv2
from skimage import io

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
result_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result'
gen_mod = Generater()
gen_mod.cuda()
gen_mod.load_state_dict(torch.load('../tree.pth'))
gen_mod.eval()


def run(task_drs,save_drs):


    for d in glob.glob(os.path.join(save_drs,'*.*')):
        os.remove(d)
    dd = glob.glob(os.path.join(task_drs, '*.png'))

    with torch.no_grad():
        for x in dd:
            print(x)
            try:
                ig_name = x.split('/')[-1]

                tt = cv2.imread(x)
                tt1 = np.zeros(shape=(256,256,3),dtype=np.uint8)
                org_imgs = io.imread(x)[:,:,0:3]
                org_img = (org_imgs - [123.15, 115.90, 103.06]) / 255.0
                org_img = np.expand_dims(org_img,0)
                img = np.transpose(org_img, axes=[0, 3, 1, 2])
                img = torch.from_numpy(img).float()
                img = torch.autograd.Variable(img.cuda())

                mask_logits, edge_logits = gen_mod(img)
                mask_out = torch.sigmoid(mask_logits)
                edge_out = torch.sigmoid(edge_logits)

                mask_out = mask_out.cpu().detach().numpy()
                edge_out = edge_out.cpu().detach().numpy()
                final_out = mask_out-edge_out

                final_out = final_out[0, 0, :, :]
                final_out = final_out*255
                final_out[np.where(final_out<125)]=0
                final_out[np.where(final_out >= 125)] = 255

                cv2.imwrite(os.path.join(save_drs, ig_name), final_out)
            except:
                pass



if __name__ == '__main__':
    task_name = 'eb2bec6b-9e6a-49d4-aee8-41f120fe5e59'
    root_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/land'
    save_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result/line_edge/'
    tmp_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/result'

    for task_name in os.listdir('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/land'):
        mask_dr = os.path.join(tmp_dr,task_name+'_seg.jpg')
        img_dr = os.path.join(tmp_dr,task_name+'.jpg')
        result_img_dr = os.path.join(tmp_dr,task_name+'_ok.jpg')

        task_dr = os.path.join(root_dr,task_name)
        run(task_dr,save_dr)
        utils_cv.hebing_image(task_dr,img_dr)
        utils_cv.hebing_image(save_dr,mask_dr,xy_root=task_dr)
        ct = utils_cv.get_right_counter(mask_dr)
        utils_cv.draw_edge(ct, img_dr)


















