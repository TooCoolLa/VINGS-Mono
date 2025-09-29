import torch
import kornia.feature as KF

def get_matching_keypoints(kp1, kp2, idxs):
    mkpts1 = kp1[idxs[:, 0]]
    mkpts2 = kp2[idxs[:, 1]]
    return mkpts1, mkpts2

def get_relative_points(loop_model, img_prev, img_curr):
    '''
    img_prev, img_curr: [3, H, W]
    '''
    with torch.inference_mode():
        inp = torch.concat([img_prev.unsqueeze(0), img_curr.unsqueeze(0)], dim=0) # (2, 3, H, W)
        hw = torch.tensor(img_prev.shape[1:], device=loop_model.device)
        features1, features2 = loop_model.disk(inp, loop_model.num_features, pad_if_not_divisible=True)
        kps1, descs1 = features1.keypoints, features1.descriptors
        kps2, descs2 = features2.keypoints, features2.descriptors
        lafs1 = KF.laf_from_center_scale_ori(kps1[None], torch.ones(1, len(kps1), 1, 1, device=loop_model.device))
        lafs2 = KF.laf_from_center_scale_ori(kps2[None], torch.ones(1, len(kps2), 1, 1, device=loop_model.device))
        dists, idxs = loop_model.lg_matcher(descs1, descs2, lafs1, lafs2, hw1=hw, hw2=hw)

    mkpts1, mkpts2 = get_matching_keypoints(kps1, kps2, idxs) # (N, 2), (N, 2)
    # Filter out outliers.
    mkpts1_filtered, mkpts2_filtered = mkpts1, mkpts2
    
    return mkpts1_filtered, mkpts2_filtered

# TODO: 计算相对位姿
def get_relative_pose(c2w_prev, depth_prev, mkpts1_filtered, mkpts2_filtered):
    '''
    Input:
        c2w_prev: (4, 4)
        depth_prev: (1, H, W)
        mkpts1_filtered, mkpts2_filtered: (n, 2)
    Output:
        c2w_curr: (4, 4)
    Description:
        注意这里由于scale可能飘了,第二张图的深度不能用;
    '''
    pass






