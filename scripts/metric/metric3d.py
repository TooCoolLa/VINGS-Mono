import numpy as np
import cv2
import sys
sys.path.append('/data/wuke/workspace/droid_metric/')
from modules import Metric 

CKPT_PATH  = '/data/wuke/workspace/droid_metric/weights/metric_depth_vit_small_800k.pth'
MODEL_NAME = 'v2-S' # choices=['v2-L', 'v2-S', 'v2-g']


class Metric3D_Model:
    def __init__(self, cfg):
        self.cfg = cfg
        self.predictor = Metric(checkpoint=CKPT_PATH, model_name=MODEL_NAME)
    
    def predict(self, img):
        '''
        img: (H, W, 3 or 4), np.array
        '''
        H, W = img.shape[:2]
        intrinsic_dict = self.cfg['intrinsic']
        intr  = np.array([intrinsic_dict['fv'], intrinsic_dict['fu'], intrinsic_dict['cv'], intrinsic_dict['cu']])
        pred_depth     = self.predictor(rgb_image=img, intrinsic=intr, d_max=300.0)
        pred_depth_npy = cv2.resize(pred_depth.cpu().numpy(), (W, H))  # (H, W)
        pred_depth_npy = pred_depth_npy[np.newaxis, :, :, np.newaxis]
        return pred_depth_npy
    


