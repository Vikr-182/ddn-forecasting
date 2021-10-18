import os
import pandas as pd
import glob
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
 
def denoise(gt_x, gt_y, w = 7):
    # denoising
    w = w
    gt_x_t = []
    gt_y_t = []
    for iq in range(len(gt_x)):
        if iq >= w and iq + w <= len(gt_x):
            gt_x_t.append(np.mean(gt_x[iq: iq + w]))
            gt_y_t.append(np.mean(gt_y[iq: iq + w]))
        elif iq < w:
            okx = np.mean(gt_x[w: w + w])
            gt_x_t.append(gt_x[0] + (okx - gt_x[0]) * (iq) / w)
            oky = np.mean(gt_y[w: w + w])
            gt_y_t.append(gt_y[0] + (oky - gt_y[0]) * (iq) / w)
        else:
            okx = np.mean(gt_x[len(gt_x) - w:len(gt_x) - w  + w])
            oky = np.mean(gt_y[len(gt_x) - w: len(gt_x) - w + w])
            gt_x_t.append(okx + (gt_x[-1] - okx) * (w - (len(gt_x) - iq)) / w)
            gt_y_t.append(oky + (gt_y[-1] - oky) * (w - (len(gt_y) - iq)) / w)                   

    gt_x = gt_x_t
    gt_y = gt_y_t
    return gt_x, gt_y

def rotate(gt_x, gt_y,theta):
    gt_x_x = [ (gt_x[k] * np.cos(theta) - gt_y[k] * np.sin(theta))  for k in range(len(gt_x))]
    gt_y_y = [ (gt_x[k] * np.sin(theta) + gt_y[k] * np.cos(theta))  for k in range(len(gt_x))]
    gt_x = gt_x_x
    gt_y = gt_y_y
    return gt_x, gt_y

def transform(x_traj, y_traj, dt = 0.3, t_obs = 20, theta=None, offsets=None):
    if offsets == None:
        x_traj -= x_traj[0]
        y_traj -= y_traj[0]
    else:
        x_traj -= offsets[0]
        y_traj -= offsets[1]
        
    gt_x = x_traj
    gt_y = y_traj
    
    gt_x, gt_y = denoise(gt_x, gt_y)
    v_x = [ (gt_x[k + 1] - gt_x[k])/dt  for k in range(len(gt_x) - 1)]
    v_y = [ (gt_y[k + 1] - gt_y[k])/dt  for k in range(len(gt_y) - 1)]
    psi = [ np.arctan2(v_y[k], v_x[k]) for k in range(len(v_x))]  

    # till here, gt-> (50, 1), v -> (49, 1), psi -> (31, 1)

    # obtain this -psi
    if theta == None:
        theta = -psi[t_obs - 1]

    # rotate by theta
    gt_x, gt_y = rotate(gt_x, gt_y, theta)
    v_x = [ (gt_x[k + 1] - gt_x[k])/dt  for k in range(len(gt_x) - 1)]
    v_y = [ (gt_y[k + 1] - gt_y[k])/dt  for k in range(len(gt_y) - 1)]
    psi = [ np.arctan2(v_y[k], v_x[k]) for k in range(len(v_x))]
    psidot = [ (psi[k + 1] - psi[k])/dt for k in range(len(psi) - 1) ]
    psi_traj = [i.item() for i in psi]
    psidot_traj = [i.item() for i in psidot]
    
    return gt_x, gt_y, v_x, v_y, psi, psidot, psi_traj, psidot_traj, theta

class ArgoverseDataset(Dataset):
    def __init__(self, data_path, t_obs=16, dt=0.125,centerline_dir=None, include_centerline = False, flatten=True, end_point = False):
        self.paths = glob.glob(os.path.join(data_path, "*.csv"))
        self.data_path = data_path
        self.t_obs = t_obs
        self.dt = dt
        self.end_point = end_point
        self.include_centerline = include_centerline
        self.centerline_dir = centerline_dir
        self.flatten = flatten
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        dt = self.dt
        path = self.paths[idx]
        df = pd.read_csv(path)
        agent_df = df[df['OBJECT_TYPE'] == 'AGENT']
        others_df = df[df['OBJECT_TYPE'] == 'OTHERS']
        others_dfs = np.array([v for k, v in others_df.groupby('TRACK_ID')], dtype=object)
        av_df = df[df['OBJECT_TYPE'] == 'AV']        

        x_traj = agent_df['X'].values
        y_traj = agent_df['Y'].values                
        
        x_traj -= x_traj[0]
        y_traj -= y_traj[0]

        x_traj, y_traj, v_x, v_y, psi, psi_dot, psi_traj, psidot_traj, theta = transform(x_traj, y_traj, dt = self.dt, t_obs = self.t_obs)                                

        x_inp = x_traj[:self.t_obs]
        y_inp = y_traj[:self.t_obs]
        x_fut = x_traj[self.t_obs:]
        y_fut = y_traj[self.t_obs:]

        # till here, gt-> (32, 1), v -> (31, 1), psi -> (31, 1), psidot -> (30, 1)
        psi_fut = psi_traj[self.t_obs - 1:]
        psidot_fut = psi_traj[self.t_obs - 2:]
        
        vx_traj = v_x
        vy_traj = v_y
        
        vx_beg = vx_traj[self.t_obs]
        vy_beg = vy_traj[self.t_obs]
        
        vx_beg_prev = vx_traj[self.t_obs - 1]
        vy_beg_prev = vy_traj[self.t_obs - 1]
        
        ax_beg = (vx_beg - vx_beg_prev) / self.dt
        ay_beg = (vy_beg - vy_beg_prev) / self.dt

        vx_fin = v_x[-1]
        vy_fin = v_y[-1]
        
        vx_fin_prev = v_x[-2]
        vy_fin_prev = v_y[-2]

        ax_fin = (vx_fin - vx_fin_prev) / self.dt
        ay_fin = (vy_fin - vy_fin_prev) / self.dt

        x_fut = x_traj[self.t_obs:]
        y_fut = y_traj[self.t_obs:]
        
        if self.flatten:
            traj_inp = np.dstack((x_inp, y_inp)).flatten()  
        else:
            traj_inp = np.vstack((x_inp, y_inp))
            traj_inp = np.swapaxes(traj_inp, 0, 1)
            
        traj_out = np.hstack((x_fut, y_fut)).flatten()

        fixed_params = np.array([x_fut[0], y_fut[0], 0, psi_fut[0], psidot_fut[0]])
        var_inp = np.array([x_inp[-1], y_inp[-1], psi_fut[-1]])

        if self.end_point:
            return torch.tensor(traj_inp), torch.tensor([psi[-1], x_traj[-1], y_traj[-1]]), torch.tensor(fixed_params), torch.tensor(var_inp)
        else:
            return torch.tensor(traj_inp), torch.tensor(traj_out), torch.tensor(fixed_params), torch.tensor(var_inp)
        
class ArgoverseDataset_Old(Dataset):
    def __init__(self, data_path, t_obs=16, dt=0.125,centerline_dir=None, include_centerline = False, flatten=True, end_point = False):
        self.data = np.load(data_path, allow_pickle=True)
        self.data_path = data_path
        self.t_obs = t_obs
        self.dt = dt
        self.include_centerline = include_centerline
        self.centerline_dir = centerline_dir
        self.flatten = flatten
        self.end_point = end_point
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        dt = self.dt
        traj = self.data[idx]
        x_traj = traj[:, 0]
        y_traj = traj[:, 1]
        
        x_traj -= x_traj[0]
        y_traj -= y_traj[0]
                
        gt_x = x_traj
        gt_y = y_traj
                
        gt_x, gt_y = denoise(gt_x, gt_y)
        v_x = [ (gt_x[k + 1] - gt_x[k])/dt  for k in range(len(gt_x) - 1)]
        v_y = [ (gt_y[k + 1] - gt_y[k])/dt  for k in range(len(gt_y) - 1)]
        psi = [ np.arctan2(v_y[k], v_x[k]) for k in range(len(v_x))]  
        
        # till here, gt-> (50, 1), v -> (49, 1), psi -> (31, 1)
        
        # obtain this -psi
        theta = -psi[self.t_obs - 1]
        
        # rotate by theta
        gt_x, gt_y = rotate(gt_x, gt_y, theta)
        v_x = [ (gt_x[k + 1] - gt_x[k])/dt  for k in range(len(gt_x) - 1)]
        v_y = [ (gt_y[k + 1] - gt_y[k])/dt  for k in range(len(gt_y) - 1)]
        psi = [ np.arctan2(v_y[k], v_x[k]) for k in range(len(v_x))]
        psidot = [ (psi[k + 1] - psi[k])/dt for k in range(len(psi) - 1) ]
        psi_traj = [i.item() for i in psi]
        psidot_traj = [i.item() for i in psidot]
        
        x_traj = gt_x
        y_traj = gt_y

        x_inp = x_traj[:self.t_obs]
        y_inp = y_traj[:self.t_obs]
        x_fut = x_traj[self.t_obs:]
        y_fut = y_traj[self.t_obs:]

        # till here, gt-> (32, 1), v -> (31, 1), psi -> (31, 1), psidot -> (30, 1)
        psi_fut = psi_traj[self.t_obs - 1:]
        psidot_fut = psi_traj[self.t_obs - 2:]
        
        vx_traj = v_x
        vy_traj = v_y
        
        vx_beg = vx_traj[self.t_obs]
        vy_beg = vy_traj[self.t_obs]
        
        vx_beg_prev = vx_traj[self.t_obs - 1]
        vy_beg_prev = vy_traj[self.t_obs - 1]
        
        ax_beg = (vx_beg - vx_beg_prev) / self.dt
        ay_beg = (vy_beg - vy_beg_prev) / self.dt

        vx_fin = v_x[-1]
        vy_fin = v_y[-1]
        
        vx_fin_prev = v_x[-2]
        vy_fin_prev = v_y[-2]

        ax_fin = (vx_fin - vx_fin_prev) / self.dt
        ay_fin = (vy_fin - vy_fin_prev) / self.dt

        x_fut = x_traj[self.t_obs:]
        y_fut = y_traj[self.t_obs:]
        
        if self.flatten:
            traj_inp = np.dstack((x_inp, y_inp)).flatten()  
        else:
            traj_inp = np.vstack((x_inp, y_inp))
            traj_inp = np.swapaxes(traj_inp, 0, 1)
        
        if self.include_centerline:
            cs = np.load(self.centerline_dir)[idx]
            data = np.load(self.data_path)

            c_x = cs[:, 0]            
            c_y = cs[:, 1]
            c_x -= data[idx][0,0]
            c_y -= data[idx][0,1]
            c_x, c_y = denoise(c_x, c_y)
    
            # rotate by theta
            c_x, c_y = rotate(c_x, c_y, theta)
            c_x -= c_x[0]
            c_y -= c_y[0]
            c_x += x_inp[-1]
            c_y += y_inp[-1]
        
            c_inp = np.dstack((c_x, c_y)).flatten()
            traj_inp = np.hstack((traj_inp, c_inp))
            
        vx_fut = vx_traj[self.t_obs:]
        vy_fut = vy_traj[self.t_obs:]
#         traj_out = np.vstack((x_fut, y_fut))#.flatten()
#         traj_out = np.swapaxes(traj_out, 0, 1)
        traj_out = np.hstack((x_fut, y_fut)).flatten()
#         traj_out = np.hstack((x_fut, y_fut)).flatten()

        fixed_params = np.array([x_fut[0], y_fut[0], 0, psi_fut[0], psidot_fut[0]])
        var_inp = np.array([x_inp[-1], y_inp[-1], psi_fut[-1]])
#         var_inp = np.array([x_inp[-1], y_inp[-1], psi_fut[-1], x_fut[10], y_fut[10], x_fut[20], y_fut[20]])

        if self.end_point:
            return torch.tensor(traj_inp), torch.tensor([psi[-1], x_traj[-1], y_traj[-1]]), torch.tensor(fixed_params), torch.tensor(var_inp)
        else:
            return torch.tensor(traj_inp), torch.tensor(traj_out), torch.tensor(fixed_params), torch.tensor(var_inp)        
