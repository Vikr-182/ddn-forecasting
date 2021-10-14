import argparse
import gc
import logging
import os
import glob
import pandas as pd

import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')


import time

from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
torch.backends.cudnn.benchmark = True

from matplotlib import pyplot as plt

from matplotlib import pyplot as plt
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.visualization.visualize_sequences import viz_sequence
avm = ArgoverseMap()

num = 10

def vis_trajectories(data_path, output_dir="results/", dt = 0.3, t_obs=20, pred=False, pred_array=None, batch_size = 512):
    paths = glob.glob(os.path.join(data_path, "*.csv"))
    for idx in range(len(paths)):
        path = paths[idx]
        df = pd.read_csv(path)
        agent_df = df[df['OBJECT_TYPE'] == 'AGENT']
        others_df = df[df['OBJECT_TYPE'] == 'OTHERS']
        others_dfs = np.array([v for k, v in others_df.groupby('TRACK_ID')], dtype=object)
        av_df = df[df['OBJECT_TYPE'] == 'AV']
        city = df['CITY_NAME'].values[0]

        # agent
        x_traj = agent_df['X'].values
        y_traj = agent_df['Y'].values
        offsets = [x_traj[0], y_traj[0]] # offsets for other agents
#         x_traj, y_traj, v_x, v_y, psi, psi_dot, psi_traj, psidot_traj, theta_agent = transform(x_traj, y_traj)
        plt.figure(figsize=(15,15))

        plt.plot(x_traj[:t_obs], y_traj[:t_obs], color='blue', label='observed')
        plt.scatter(x_traj[t_obs], y_traj[t_obs], color='blue', label='end observed')
        plt.plot(x_traj[t_obs:], y_traj[t_obs:], color='orange', label='gt')
        plt.scatter(x_traj[-1], y_traj[-1], color='orange', label='gt end point')
        if pred:
            i1 = idx//batch_size
            i2 = idx % batch_size
            x_traj, y_traj, v_x, v_y, psi, psi_dot, psi_traj, psidot_traj, theta_agent = transform(x_traj, y_traj)
            pred_x = pred[i1, i2, :, 0]
            pred_y = pred[i1, i2, :, 1]
            pred_x, pred_y = rotate(pred_x, pred_y, -theta)
            pred_x += offsets[0]
            pred_y += offsets[1]
            plt.plot(pred_x, pred_y, color='orange', label='predicted')
            plt.scatter(pred_x[-1], pred_y[-1], color='orange', label='predicted goal point')    
    
        # av
        x_traj = av_df['X'].values
        y_traj = av_df['Y'].values
        plt.plot(x_traj, y_traj, color='red', label='AV')
        plt.scatter(x_traj[-1], y_traj[-1], color='red')
        
        # others
        for other in others_dfs:
            x_traj = other['X'].values
            y_traj = other['Y'].values
#             x_traj, y_traj, v_x, v_y, psi, psi_dot, psi_traj, psidot_traj, theta_others = transform(x_traj, y_traj, theta = theta_agent, offsets = offsets)
            plt.plot(x_traj, y_traj, color='grey')
            plt.scatter(x_traj[-1], y_traj[-1], color='grey')
        
        # centerlines
        lane_centerlines = []    
        # Get lane centerlines which lie within the range of trajectories
        agent_df = df[df['OBJECT_TYPE'] == 'AGENT']
        gt_x = agent_df['X'].values
        gt_y = agent_df['Y'].values
        
        x_max, y_max = np.max(gt_x) + 30, np.max(gt_y) + 30
        x_min, y_min = np.min(gt_x) - 30, np.min(gt_y) - 30
        
        print(x_max, x_min)
        print(y_max, y_min)

        avm = ArgoverseMap()
        seq_lane_props = avm.city_lane_centerlines_dict[city]
        for lane_id, lane_props in seq_lane_props.items():
            lane_cl = lane_props.centerline
            if (np.min(lane_cl[:, 0]) < x_max and np.min(lane_cl[:, 1]) < y_max and np.max(lane_cl[:, 0]) > x_min and np.max(lane_cl[:, 1]) > y_min):
                lane_centerlines.append(lane_cl)

        for lane_cl in lane_centerlines:
            plt.plot(lane_cl[:, 0], lane_cl[:, 1], "--", color="grey", alpha=1, linewidth=1, zorder=0)

        plt.legend()
        plt.axis('equal')
        plt.savefig(output_dir + str(idx) + ".png")
        plt.clf()


