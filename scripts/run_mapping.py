import numpy as np
import shutil
import torch
from lietorch import SE3
import os
from frontend.dbaf import DBAFusion
from gaussian.gaussian_model import GaussianModel
from gaussian.vis_utils import save_ply, vis_map
import argparse
parser = argparse.ArgumentParser(description="Add config path.")
parser.add_argument("config")
args = parser.parse_args()
config_path = args.config
from gaussian.general_utils import load_config, get_name
config = load_config(config_path)
import importlib
get_dataset = importlib.import_module(config["dataset"]["module"]).get_dataset
from vings_utils.middleware_utils import judge_and_package, retrieve_to_tracker, datapacket_to_nerfslam
from storage.storage_manage import StorageManager
from loop.loop_model import LoopModel
from metric.metric_model import Metric_Model
import time
if config['mode'] == 'vo_nerfslam': from frontend_vo.vio_slam import VioSLAM
from tqdm import tqdm
from datasets.pth import Pth_Loader
# -  -  -  -  -  -  -  -  -  -  -
# PTH_DIR = '/data/wuke/workspace/Droid2DAcc/output/11-06-19-05-kintinuous/debug_dict'
PTH_DIR = '/data/wuke/workspace/VINGS-Mono/output/12-22-15-28-hierarchical-smallcit-/vizout_dict'
# -  -  -  -  -  -  -  -  -  -  -




class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset  = get_dataset(cfg)
        cfg['frontend']['c2i'] = self.dataset.c2i # (4, 4), ndarray
        
        self.mapper = GaussianModel(cfg)
        
        self.pth_loader = Pth_Loader(cfg, PTH_DIR)
        
        self.tracker = None
        
        if 'use_storage_manager' in cfg.keys() and cfg['use_storage_manager']:
            self.use_storage_manager = True
            self.storage_manager = StorageManager(cfg)
            if cfg['dataset']['module'] != 'phone':
                self.storage_manager.dataset_length = self.dataset.rgbinfo_dict['timestamp'][-1] - self.dataset.rgbinfo_dict['timestamp'][0] 
        else:
            self.use_storage_manager = False
    
    
    def vizout_to_mapperinput(self, viz_out):
        rgbs       = viz_out['cam0_images'].permute(0,2,3,1) / 255.0
        depths     = (1.0/viz_out['cam0_idepths_up'][..., None])
        depths_cov = viz_out['cam0_depths_cov_up'].unsqueeze(-1)
        c2ws       = torch.linalg.inv(SE3(viz_out['cam0_poses']).matrix())
        tstamps    = viz_out['viz_out_idx_to_f_idx']
        calibs     = viz_out["calibs"]
        DEVICE = self.cfg['device']['mapper']
        
        N_frames = depths.shape[0]
        cov_median = torch.tensor(np.median(depths_cov.reshape(N_frames, -1), axis=1)[:, None, None, None], device=depths.device) # (N, 1, 1, 1)
        zero_mask = torch.bitwise_or(depths_cov>(cov_median*self.cfg['middleware']['cov_times']), depths>self.cfg['middleware']['max_depth'])
        depths[zero_mask] = 0
        rgbs[zero_mask.squeeze(-1)] = 0
        
        camera_model = calibs[0].camera_model
        intrinsic    = {'fv': camera_model[0], 'fu': camera_model[1], 'cv': camera_model[2], 'cu': camera_model[3], 
                         'H': depths.shape[1], 'W': depths.shape[2]}
        viz_out = {'images': rgbs.to(DEVICE), 'depths': depths.to(DEVICE), 'depths_cov': depths_cov.to(DEVICE), 
                   'poses': c2ws.to(DEVICE), 'viz_out_idx_to_f_idx': tstamps, 'intrinsic': intrinsic, 'global_kf_id': viz_out['viz_out_idx_to_f_idx']}
        return viz_out


    def run(self):
        for idx in tqdm(range(0, len(self.pth_loader))):
            viz_out = self.pth_loader.load_data(idx) 
            torch.cuda.empty_cache()
            # self.mapper.run(self.vizout_to_mapperinput(viz_out))
            # print(viz_out.keys())
            self.mapper.run(viz_out)
            
            if self.use_storage_manager and (idx+1) % 10 == 0:
                self.storage_manager.run(self.tracker, self.mapper, viz_out)
                torch.cuda.empty_cache()
            
            if ((idx+1) % 500 == 0 or (idx == len(self.dataset) - 1)) and self.mapper._xyz.shape[0] > 0:
                save_ply(self.mapper, idx, save_mode='2dgs')
           
             
if __name__ == '__main__':
    config['output']['save_dir'] = os.path.join(config['output']['save_dir'], get_name(config))
    os.makedirs(config['output']['save_dir']+'/droid_c2w', exist_ok=True)
    os.makedirs(config['output']['save_dir']+'/rgbdnua', exist_ok=True)
    os.makedirs(config['output']['save_dir']+'/ply', exist_ok=True)
    if 'debug_mode' in list(config.keys()) and config['debug_mode']:
        os.makedirs(config['output']['save_dir']+'/debug_dict', exist_ok=True)
    shutil.copy(config_path, config['output']['save_dir']+'/config.yaml')
    
    runner = Runner(config)
    torch.backends.cudnn.benchmark = True
    
    runner.run()

    