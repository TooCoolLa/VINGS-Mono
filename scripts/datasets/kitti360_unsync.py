import os
import numpy as np
import torch
import cv2
from datetime import datetime
from tqdm import tqdm
from lietorch import SE3

class KITTI360UnsyncDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_dir = cfg['dataset']['root']
        self.rgb_dir = os.path.join(self.dataset_dir, 'image_00', 'data_rgb')
        self.preload_rgbinfo()
        self.c2i = np.loadtxt(os.path.join(self.dataset_dir, 'metadata', 'c2i.txt'))
        self.tqdm = tqdm(total=self.__len__())
        
    def __len__(self):
        return len(self.rgbinfo_dict['timestamp'])
    
    def preload_camtimestamp(self):
        return (np.loadtxt(os.path.join(self.dataset_dir, 'metadata', 'camstamp.txt'), str)[:,0]).astype(np.float64)[None].transpose(1,0)
    
    def preload_imu(self):
        all_imu = np.loadtxt(os.path.join(self.cfg['dataset']['root'], 'metadata', 'imu.txt'))
        all_imu[:, 0] -= 0.04
        return all_imu
    
    def preload_rgbinfo(self):
        timstamp_filename = np.loadtxt(os.path.join(self.dataset_dir, 'metadata', 'camstamp.txt'), dtype=str)
        rgbinfo_dict = {}
        rgbinfo_dict['timestamp'] = list(map(lambda x: float(x), timstamp_filename[:, 0].tolist())) # (N, ), list
        rgbinfo_dict['filepath']  = list(map(lambda x: os.path.join(self.rgb_dir, x), timstamp_filename[:, 1].tolist())) # (N, ), list
        self.rgbinfo_dict = rgbinfo_dict
    
    def __getitem__(self, idx):
        resized_h, resized_w = int(self.cfg['frontend']['image_size'][0]), int(self.cfg['frontend']['image_size'][1])
        rgb_raw = cv2.imread(self.rgbinfo_dict['filepath'][idx])
        
        # Distort.
        K = np.eye(3)
        K[0,0] = self.cfg['intrinsic']['fv']
        K[0,2] = self.cfg['intrinsic']['cv']
        K[1,1] = self.cfg['intrinsic']['fu']
        K[1,2] = self.cfg['intrinsic']['cu']
        
        rgb_raw = cv2.undistort(rgb_raw, K, np.array(self.cfg['intrinsic']['distortion_coeffs']))
        rgb_raw = rgb_raw[:, :, :]
        
        
        rgb = (torch.tensor(cv2.resize(rgb_raw, (resized_w, resized_h)))[...,[2,1,0]]).permute(2,0,1).unsqueeze(0).to(self.cfg['device']['tracker'])
        u_scale, v_scale = resized_h/self.cfg['intrinsic']['H'], resized_w/self.cfg['intrinsic']['W']
        intrinsic = torch.tensor([self.cfg['intrinsic']['fv']*v_scale, self.cfg['intrinsic']['fu']*u_scale, \
                                  self.cfg['intrinsic']['cv']*v_scale, self.cfg['intrinsic']['cu']*u_scale], dtype=torch.float32, device=self.cfg['device']['tracker'])
        data_packet = {}
        data_packet['timestamp'] = self.rgbinfo_dict['timestamp'][idx] # float 
        data_packet['rgb']       = rgb                                 # (1, 3, H, W)
        data_packet['intrinsic'] = intrinsic                           # (4, )
        self.tqdm.update(1)
        return data_packet

    def load_gt_dict(self):
        '''
        43287.85515 0.00000 0.00000 0.00000 0.01850189 0.02195314 0.29188570 0.95602222
        '''
        timestamp_tqs = np.loadtxt(os.path.join(self.dataset_dir, 'metadata', 'gt_local.txt'))
        
        timestamps = timestamp_tqs[:, 0]
        c2ws       = SE3(torch.tensor(timestamp_tqs[:, 1:])).matrix().numpy()
        
        gt_dict = {'timestamps': timestamps, 'c2ws': c2ws}
        return gt_dict
    
def get_dataset(config):
    return KITTI360UnsyncDataset(config)
