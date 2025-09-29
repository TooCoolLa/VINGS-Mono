#!/usr/bin/env python3

from abc import abstractclassmethod

from icecream import ic
import torch
from torch import nn
from frontend_vo.vo_factor_graph.variables import Variable, Variables
from frontend_vo.vo_factor_graph.factor_graph import TorchFactorGraph

from gtsam import Values
from gtsam import NonlinearFactorGraph
from gtsam import GaussianFactorGraph

from frontend_vo.slam.meta_slam import SLAM
from frontend_vo.slam.visual_frontends.visual_frontend import RaftVisualFrontend
from frontend_vo.solvers.nonlinear_solver import iSAM2, LevenbergMarquardt, Solver

########################### REMOVE ############################
import numpy as np
import gtsam
from gtsam import (ImuFactor, Pose3, Rot3, Point3)
from gtsam import PriorFactorPose3, PriorFactorConstantBias, PriorFactorVector
from gtsam.symbol_shorthand import B, V, X


# Send IMU priors
def initial_priors_and_values(k, initial_state):
    pose_key = X(k)
    vel_key = V(k)
    bias_key = B(k)

    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.000001, 0.000001, 0.000001, 0.00001, 0.00001, 0.00001]))
    vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.000001)
    bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.00001)

    initial_pose, initial_vel, initial_bias = initial_state

    # Get inertial factors 
    pose_prior = PriorFactorPose3(pose_key, initial_pose, pose_noise)
    vel_prior = PriorFactorVector(vel_key, initial_vel, vel_noise)
    bias_prior = PriorFactorConstantBias(bias_key, initial_bias, bias_noise)

    # Add factors to inertial graph
    graph = NonlinearFactorGraph()
    graph.push_back(pose_prior)
    graph.push_back(vel_prior)
    graph.push_back(bias_prior)

    # Get guessed values
    x0 = Values()
    x0.insert(pose_key, initial_pose)
    x0.insert(vel_key, initial_vel)
    x0.insert(bias_key, initial_bias)

    return x0, graph

def initial_state():
    true_world_T_imu_t0 = gtsam.Pose3(gtsam.Rot3(0.060514, -0.828459, -0.058956, -0.553641),  # qw, qx, qy, qz
                                      gtsam.Point3(0.878612, 2.142470, 0.947262))
    true_vel = np.array([0.009474,-0.014009,-0.002145])
    true_bias = gtsam.imuBias.ConstantBias(np.array([-0.012492,0.547666,0.069073]), np.array([-0.002229,0.020700,0.076350]))
    # naive_pose = gtsam.Pose3.identity()
    # TTD 2024/04/21
    naive_pose = gtsam.Pose3.Identity()
    
    naive_vel = np.zeros(3)
    naive_bias = gtsam.imuBias.ConstantBias()
    initial_pose = true_world_T_imu_t0
    initial_vel = true_vel
    initial_bias = true_bias
    initial_pose = naive_pose
    initial_vel = naive_vel
    initial_bias = naive_bias
    return initial_pose, initial_vel, initial_bias
###############################################################


class NoUseClass:
    def __init__(self):
        pass


class VioSLAM(SLAM):
    def __init__(self, cfg):
        
        self.cfg = cfg
        
        device = cfg['device']['tracker']
        super().__init__('tracker', device)
        world_T_imu_t0, _, _ = initial_state()
        

        imu_T_cam0 = Pose3(np.eye(4))
        
        
        # ------------------------------------------------------------------------------
        world_T_imu_t0 = Pose3(np.eye(4))
        # TTD 2024/05/02
        # world_T_imu_t0 = Pose3(np.eye(4))
        # ------------------------------------------------------------------------------
        
        # self.visual_frontend = RaftVisualFrontend(world_T_imu_t0, imu_T_cam0, cfg, device=device)
        self.visual_frontend = RaftVisualFrontend(cfg, device=device)
        
        # No use, just acceptor.
        self.frontend = NoUseClass()
        
        self.backend = iSAM2()

        self.values = Values()
        self.inertial_factors = NonlinearFactorGraph()
        self.visual_factors = NonlinearFactorGraph()

        self.last_state = None
        
        # TTD 2024/04/24
        self.is_initialized = True
        
        self.new_frame_added = False
    
    def stop_condition(self):
        return self.visual_frontend.stop_condition()

    # Converts sensory inputs to measurements and initial guess
    def _frontend(self, batch, last_state, last_delta):
        '''
        batch = patch['data'], 就是datasets.__getitem__的字典哈
        '''
        # Compute optical flow
        '''
        batch['images'].shape: (1, 344, 616, 4)
        batch['depths'].shape: (1, 344, 616, 1)
        '''
        x0_visual, visual_factors, viz_out = self.visual_frontend(batch)  # TODO: currently also calls BA, and global BA
        self.last_state = x0_visual

        if x0_visual is None:
            return False
        
        # Wrap guesses
        x0 = Values()
        factors = NonlinearFactorGraph()

        return x0, factors, viz_out

    def _backend(self, factor_graph, x0):
        return self.backend.solve(factor_graph, x0)
    
    # TTD 2024/04/24
    def run(self, input_data):
        '''
        input_data就一个字典
            images
            poses
            depths
            t_cams
            calibs
        '''
        state, viz_out = self.forward(input_data)
        
        if viz_out is not None: self.new_frame_added = True
        else: self.new_frame_added = False 
        
        return state, viz_out
    
    def track(self, input_data):
        with torch.no_grad():
            state, viz_out = self.forward(input_data)
            if viz_out is not None: self.viz_out = viz_out
            else: self.viz_out = None
        
        
    # Tailored for debug Looper.
    def save_pt_ckpt(self, save_path):
        # Only save video's attrributes.
        save_dict = {'visual_frontend': {'cam0_images': None, 'cam0_intrinsics': None, 'cam0_T_world': None, 'world_T_body': None}}
        
        save_dict['visual_frontend']['cam0_T_world']       = self.visual_frontend.cam0_T_world
        save_dict['visual_frontend']['world_T_body']       = self.visual_frontend.world_T_body
        save_dict['visual_frontend']['world_T_body_cov']   = self.visual_frontend.world_T_body_cov
        save_dict['visual_frontend']['cam0_idepths']       = self.visual_frontend.cam0_idepths
        save_dict['visual_frontend']['cam0_idepths_cov']   = self.visual_frontend.cam0_idepths_cov
        save_dict['visual_frontend']['cam0_depths_cov']    = self.visual_frontend.cam0_depths_cov
        save_dict['visual_frontend']['cam0_idepths_up']    = self.visual_frontend.cam0_idepths_up
        save_dict['visual_frontend']['cam0_depths_cov_up'] = self.visual_frontend.cam0_depths_cov_up
        save_dict['visual_frontend']['cam0_images']        = self.visual_frontend.cam0_images
        save_dict['visual_frontend']['cam0_timestamps']    = self.visual_frontend.cam0_timestamps
        save_dict['visual_frontend']['cam0_intrinsics']    = self.visual_frontend.cam0_intrinsics
        save_dict['visual_frontend']['kf_idx_to_f_idx']    = self.visual_frontend.kf_idx_to_f_idx
        save_dict['visual_frontend']['f_idx_to_kf_idx']    = self.visual_frontend.f_idx_to_kf_idx
        
        torch.save(save_dict, save_path)
    
    def load_pt_ckpt(self, load_path):
        
        # 请你把下面的等式左右调换
        load_dict = torch.load(load_path)
        
        self.visual_frontend.cam0_T_world          = load_dict['visual_frontend']['cam0_T_world']
        self.visual_frontend.world_T_body          = load_dict['visual_frontend']['world_T_body']
        self.visual_frontend.world_T_body_cov      = load_dict['visual_frontend']['world_T_body_cov']
        self.visual_frontend.cam0_idepths          = load_dict['visual_frontend']['cam0_idepths']
        self.visual_frontend.cam0_idepths_cov      = load_dict['visual_frontend']['cam0_idepths_cov']
        self.visual_frontend.cam0_depths_cov       = load_dict['visual_frontend']['cam0_depths_cov']
        self.visual_frontend.cam0_idepths_up       = load_dict['visual_frontend']['cam0_idepths_up']
        self.visual_frontend.cam0_depths_cov_up    = load_dict['visual_frontend']['cam0_depths_cov_up']
        self.visual_frontend.cam0_images           = load_dict['visual_frontend']['cam0_images']
        self.visual_frontend.cam0_timestamps       = load_dict['visual_frontend']['cam0_timestamps'] 
        self.visual_frontend.cam0_intrinsics       = load_dict['visual_frontend']['cam0_intrinsics']
        self.visual_frontend.kf_idx_to_f_idx       = load_dict['visual_frontend']['kf_idx_to_f_idx'] 
        self.visual_frontend.f_idx_to_kf_idx       = load_dict['visual_frontend']['f_idx_to_kf_idx'] 