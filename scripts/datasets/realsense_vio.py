import os
import numpy as np
import torch
import cv2
from datetime import datetime
from tqdm import tqdm
import glob

class RealSenseDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_dir = cfg['dataset']['root']
        self.rgb_dir   = os.path.join(self.dataset_dir, 'image_00', 'data_nodyn')
        self.depth_dir = os.path.join(self.dataset_dir, 'image_00', 'depth')
        self.preload_rgbinfo()
        self.c2i = np.loadtxt(os.path.join(self.dataset_dir, 'DBAF_format', 'c2i.txt'))
        # self.tqdm = tqdm(total=self.__len__())
        
    def __len__(self):
        return len(self.rgbinfo_dict['timestamp'])
    
    def preload_camtimestamp(self):
        # print((np.loadtxt(os.path.join(self.dataset_dir, 'DBAF_format', 'camstamp.txt'), str)[:,0]).astype(np.float64)[None].transpose(1,0).shape)
        return (np.loadtxt(os.path.join(self.dataset_dir, 'DBAF_format', 'camstamp.txt'), str)[:,0]).astype(np.float64)[None].transpose(1,0)
    
    def preload_imu(self):
        all_imu = np.loadtxt(os.path.join(self.cfg['dataset']['root'], 'DBAF_format', 'imu.txt'))
        # TTD 2024//11/15
        # all_imu[:,0] -= 0.05006
        # all_imu[:, 0] -= 0.07 # 0.005006
        all_imu[:, 0] -= self.cfg['dataset']['imu_delay']
        return all_imu
    
    def preload_rgbinfo(self):
        timstamp_filename = np.loadtxt(os.path.join(self.dataset_dir, 'DBAF_format', 'camstamp.txt'), dtype=str)
        rgbinfo_dict = {}
        rgbinfo_dict['timestamp'] = list(map(lambda x: float(x), timstamp_filename[:, 0].tolist())) # (N, ), list
        rgbinfo_dict['filepath']  = list(map(lambda x: os.path.join(self.rgb_dir, x), timstamp_filename[:, 1].tolist())) # (N, ), list
        self.rgbinfo_dict = rgbinfo_dict
        depthinfo_dict = {}
        depthinfo_dict['timestamp'] = rgbinfo_dict['timestamp'] # (N, ), list
        depthinfo_dict['filepath']  = sorted(glob.glob(os.path.join(self.depth_dir, '*.npy'))) # (N, ), list
        self.depthinfo_dict = depthinfo_dict
    
    def __getitem__(self, idx):
        resized_h, resized_w = int(self.cfg['frontend']['image_size'][0]), int(self.cfg['frontend']['image_size'][1])
        rgb_raw = cv2.imread(self.rgbinfo_dict['filepath'][idx])
        rgb = (torch.tensor(cv2.resize(rgb_raw, (resized_w, resized_h)))[...,[2,1,0]]).permute(2,0,1).unsqueeze(0).to(self.cfg['device']['tracker'])
        u_scale, v_scale = resized_h/self.cfg['intrinsic']['H'], resized_w/self.cfg['intrinsic']['W']
        intrinsic = torch.tensor([self.cfg['intrinsic']['fv']*v_scale, self.cfg['intrinsic']['fu']*u_scale, \
                                  self.cfg['intrinsic']['cv']*v_scale, self.cfg['intrinsic']['cu']*u_scale], dtype=torch.float32, device=self.cfg['device']['tracker'])
        data_packet = {}
        data_packet['timestamp'] = self.rgbinfo_dict['timestamp'][idx] # float 
        data_packet['rgb']       = rgb                                 # (1, 3, H, W)
        data_packet['intrinsic'] = intrinsic                           # (4, )
        # self.tqdm.update(1)
        depth = np.load(self.depthinfo_dict['filepath'][idx])
        data_packet['depth'] = torch.tensor(depth)
        
        # self.tqdm.update(1)
        
        return data_packet
    
    
    def load_gt_dict(self):
        c2w_files    = os.listdir(os.path.join(self.dataset_dir, 'pose'))
        c2ws         = np.array([np.loadtxt(os.path.join(self.dataset_dir, 'pose', c2w_file)) for c2w_file in c2w_files])
        timestamps   = np.array(list(map(lambda x: float(x.strip('.txt')), c2w_files)))
        sorted_order = np.argsort(timestamps)
        timestamps   = timestamps[sorted_order]
        c2ws         = c2ws[sorted_order]
        gt_dict = {'timestamps': timestamps, 'c2ws': c2ws}
        return gt_dict
    

def get_dataset(config):
    return RealSenseDataset(config)
