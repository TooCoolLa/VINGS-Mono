import torch 
from lietorch import SE3

def get_xyz_bias_multi(gaussian_model, poses, optimize_c2c_tqs, localbatch_globalkf_id_mask, old_xyz):
    '''
    poses:            (B, 4, 4)
    optimize_c2c_tqs: (B, 7), cold_2_cnew.
    old_xyz:          (p, 3), p<P
    '''
    xyz_bias = torch.zeros_like(gaussian_model._xyz)
    # STEP 1 找到
    for i in range(optimize_c2c_tqs.shape[0]):
        curr_mask  = localbatch_globalkf_id_mask[i]
        c2c_matrix = SE3(optimize_c2c_tqs[i]).matrix()
        left_part  = poses[i] @ c2c_matrix @ torch.inverse(poses[i])
        xyz_bias[curr_mask] = (left_part[:3, :3]@gaussian_model._xyz[curr_mask].T).T + left_part[:3, 3].unsqueeze(0) - gaussian_model._xyz[curr_mask]
    
    return xyz_bias

def get_new_xyz_single(new_xyz, curr_c2w, curr_c2c_tq):
    left_part = curr_c2w @ SE3(curr_c2c_tq).matrix() @ torch.inverse(curr_c2w)
    new_xyz = (left_part[:3, :3]@new_xyz.T).T + left_part[:3, 3].unsqueeze(0)
    return new_xyz


def get_new_xyz(gaussian_model, poses, optimize_c2c_tqs, localbatch_globalkf_id_mask, curr_id):
    xyz_bias = get_xyz_bias_multi(gaussian_model, poses, optimize_c2c_tqs, localbatch_globalkf_id_mask, gaussian_model._xyz)
    new_xyz = gaussian_model._xyz + xyz_bias
    
    newnew_xyz = get_new_xyz_single(new_xyz, poses[curr_id], optimize_c2c_tqs[curr_id])
    
    return newnew_xyz


# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
# 2024/11/30
