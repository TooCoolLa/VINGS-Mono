import torch
# import pytorch3d.transforms as pt3dtrans

# def q2R(q):
#     '''
#     Input:  q: (N, 4)
#     Output: R: (N, 3, 3)
#     '''
#     return pt3dtrans.quaternion_to_matrix(q)


# def R2q(R):
#     '''
#     Input:  R: (N, 3, 3)
#     Output: q: (N, 4)
#     '''
#     return pt3dtrans.matrix_to_quaternion(R)

from lietorch import SO3
from scipy.spatial.transform import Rotation

def q2R(q):
    '''
    Input:  q: (N, 4)
    Output: R: (N, 3, 3)
    '''
    R = SO3.InitFromVec(q).matrix()[..., :3, :3]
    return R

def R2q(R):
    '''
    Input:  R: (N, 3, 3)
    Output: q: (N, 4)
    '''
    q = torch.tensor(Rotation.from_matrix(R.detach().cpu().numpy()).as_quat(), device=R.device) # (N, 4)
    
    return q