import os
import sys
sys.path.append("../ddn/")
sys.path.append("./")
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import scipy.special
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from scipy.linalg import block_diag
from torch.utils.data import Dataset, DataLoader
#from bernstein import bernstesin_coeff_order10_new

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.visualization.visualize_sequences import viz_sequence
avm = ArgoverseMap()

class TrajectoryDataset(Dataset):
    def __init__(self, root_dir, t_obs=20, dt=0.1, centerline_dir=None):
        self.root_dir = root_dir
        self.t_obs = t_obs
        self.dt = dt
        self.data = []
        data = np.load(self.root_dir)
        print(data.shape)
        print("AB")
        self.data = data
        self.centerline_dir = centerline_dir
    
    def __len__(self):
        return len(self.data)
    
    def get_vel(self, pos):
        return (pos[-1] - pos[-2]) / self.dt
    
    def get_acc(self, vel):
        return (vel[-1] - vel[-2]) / self.dt
    
    def __getitem__(self, idx):
#         file_name = "tests/test.npy"
#         file_path = os.path.join(self.root_dir, file_name)
        
        data = np.load(self.root_dir)
        self.data = data
#         print(data[idx].shape)
        x_traj = data[idx][:, 0]
        x_traj -= x_traj[0] + 2
        y_traj = data[idx][:, 1]
        y_traj -= y_traj[0] + 2
        
        x_inp = x_traj[:self.t_obs]
        y_inp = y_traj[:self.t_obs]
        x_fut = x_traj[self.t_obs:]
        y_fut = y_traj[self.t_obs:]
        
        vx_beg = (x_inp[-1] - x_inp[-2]) / self.dt
        vy_beg = (y_inp[-1] - y_inp[-2]) / self.dt
        
        vx_beg_prev = (x_inp[-2] - x_inp[-3]) / self.dt
        vy_beg_prev = (y_inp[-2] - y_inp[-3]) / self.dt
        
        ax_beg = (vx_beg - vx_beg_prev) / self.dt
        ay_beg = (vy_beg - vy_beg_prev) / self.dt
        
        vel_acc_inp = np.array([vx_beg, vy_beg, ax_beg, ay_beg])
        
        traj_inp = np.dstack((x_inp, y_inp)).flatten()
        if self.centerline_dir is None:
            traj_inp = np.hstack((traj_inp, vel_acc_inp))
        else:
            cs = np.load(self.centerline_dir)[idx]
            data = np.load(self.root_dir)
            c_x = cs[:, 0]
            c_x -= data[idx][0,0] + 2
            c_y = cs[:, 1]
            c_y -= data[idx][0,1] + 2
            c_inp = np.dstack((c_x, c_y)).flatten()
            traj_inp = np.hstack((traj_inp, c_inp, vel_acc_inp))
        traj_out = np.hstack((x_fut, y_fut)).flatten()
        b_inp = np.array([x_inp[-1], vx_beg, ax_beg, 0, 0, 0, y_inp[-1], vy_beg, ay_beg, 0, 0, 0])
        
        return torch.tensor(traj_inp), torch.tensor(traj_out), torch.tensor(b_inp)
    
train_dataset = TrajectoryDataset("/datasets/argoverse/val_data.npy", centerline_dir="/datasets/argoverse/val_centerlines.npy")
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=False, num_workers=0)
