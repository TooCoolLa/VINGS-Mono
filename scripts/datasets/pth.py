# Only use in train_gs_only.py;
import os
import torch



class Pth_Loader:
    def __init__(self, cfg, pth_dir):
        self.cfg = cfg
        self.pth_dir = pth_dir
        self.files = sorted(os.listdir(self.pth_dir))
    
    def __len__(self):
        return len(os.listdir(self.pth_dir))
    
    def postprocess(self, process_dict):
        if 'depths_cov' not in process_dict.keys():
            process_dict['depths_cov'] = torch.ones_like(process_dict['depths'])
        
        invalid_mask = torch.bitwise_or(process_dict['depths_cov'] > 1e8, process_dict['depths'] > 30)
        process_dict['depths'][invalid_mask] = 0
        process_dict['pixel_mask'] = torch.ones_like(process_dict['depths'].squeeze(-1), dtype=torch.bool).cuda()
        for key in list(process_dict.keys()):
            if key != 'intrinsic':
                process_dict[key] = process_dict[key][:-2].cuda()
        return process_dict

    def load_data(self, idx):
        file_path = os.path.join(self.pth_dir, self.files[idx])
        # process_dict = self.postprocess(torch.load(file_path))
        viz_out = torch.load(file_path)
        return viz_out
