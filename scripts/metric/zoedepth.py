import torch
import cv2
import numpy as np

class ZoeDepth_Model:
    def __init__(self, cfg):
        self.cfg = cfg
        device = self.cfg['device']['tracker']
        repo = "isl-org/ZoeDepth"
        # Online.
        # self.predictor = torch.hub.load(repo, "ZoeD_N", pretrained=True).to(device)
        # Offline, git clone https://github.com/isl-org/ZoeDepth
        self.predictor = torch.hub.load("/data/wuke/workspace/ZoeDepth/", "ZoeD_N", source="local", pretrained=True).to(device)
        
    
    def predict(self, img):
        '''
        img: (H, W, 3 or 4), np.array
        '''
        H, W = img.shape[:2]
        pred_depth = self.predictor.infer_pil(img[..., :3], output_type="tensor")  # as torch tensor
        pred_depth_npy = cv2.resize(pred_depth.cpu().numpy(), (W, H))  # (H, W)
        pred_depth_npy = pred_depth_npy[np.newaxis, :, :, np.newaxis]
        return pred_depth_npy
    