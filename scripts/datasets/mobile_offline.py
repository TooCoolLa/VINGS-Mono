import os
import numpy as np
import torch
import cv2
from datetime import datetime
from tqdm import tqdm


class MobileOfflineDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_dir = cfg['dataset']['root']
        self.rgb_dir = os.path.join(self.dataset_dir, 'pic')
        self.preload_rgbinfo()
        self.c2i = np.loadtxt(os.path.join(self.dataset_dir, 'c2i.txt')) #todo: add c2i.txt
        # self.tqdm = tqdm(total=self.__len__())
        
    def __len__(self):
        return len(self.rgbinfo_dict['timestamp'])
    
    def preload_camtimestamp(self):
        filenames = os.listdir(self.rgb_dir)
        stamps_s = []
        for filename in filenames:
            timestamp_ns = filename.split('.')[0]  # 假设文件名的格式为 "timestamp.ext"
            timestamp_s = float(timestamp_ns[:-9] + '.' + timestamp_ns[-9:])
            stamps_s.append(timestamp_s)
        stamps_s = sorted(stamps_s)
        time_stamps_s = np.array(stamps_s, dtype=np.float64) 
        # print(time_stamps_s[:5])
        return time_stamps_s.reshape(-1,1)
    
    def preload_imu(self):
        
        all_imu = np.loadtxt(os.path.join(self.cfg['dataset']['root'], 'imu.txt'),delimiter=',',skiprows=1)
        all_imu[:, 0] -= self.cfg['dataset']['imu_delay']
        # all_imu[:,1:4] /= 180/3.1415926
        all_imu[:, [1, 2]] = all_imu[:, [2, 1]]
        all_imu[:, [4, 5]] = all_imu[:, [5, 4]]
        # print(all_imu[:4,0])
        return all_imu
    
    def preload_rgbinfo(self):
        filenames = os.listdir(self.rgb_dir)
        filenames = sorted(filenames)
        stamps_s = []
        for filename in filenames:
            timestamp_ns = filename.split('.')[0]  # 假设文件名的格式为 "timestamp.ext"
            timestamp_s = float(timestamp_ns[:-9] + '.' + timestamp_ns[-9:])
            stamps_s.append(timestamp_s)
        rgbinfo_dict = {}
        rgbinfo_dict['timestamp'] = stamps_s
        rgbinfo_dict['filepath']  = [os.path.join(self.rgb_dir,x) for x in filenames]
        self.rgbinfo_dict = rgbinfo_dict
    
    def __getitem__(self, idx):
        resized_h, resized_w = int(self.cfg['frontend']['image_size'][0]), int(self.cfg['frontend']['image_size'][1])
        rgb_raw = cv2.imread(self.rgbinfo_dict['filepath'][idx])
        rgb_raw =cv2.rotate(rgb_raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rgb = (torch.tensor(cv2.resize(rgb_raw, (resized_w, resized_h)))[...,[2,1,0]]).permute(2,0,1).unsqueeze(0).to(self.cfg['device']['tracker'])
        # rgb = torch.rot90(rgb, k=1, dims=(2, 3))
        u_scale, v_scale = resized_h/self.cfg['intrinsic']['H'], resized_w/self.cfg['intrinsic']['W']
        intrinsic = torch.tensor([self.cfg['intrinsic']['fv']*v_scale, self.cfg['intrinsic']['fu']*u_scale, \
                                  self.cfg['intrinsic']['cv']*v_scale, self.cfg['intrinsic']['cu']*u_scale], dtype=torch.float32, device=self.cfg['device']['tracker'])
        data_packet = {}
        data_packet['timestamp'] = self.rgbinfo_dict['timestamp'][idx] # float 
        data_packet['rgb']       = rgb                                 # (1, 3, H, W)
        data_packet['intrinsic'] = intrinsic                           # (4, )
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
    return MobileOfflineDataset(config)
