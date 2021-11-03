import argparse
import gc
import logging
import os
import glob
import pandas as pd

import sys
sys.path.append("../ddn/")
sys.path.append("./")

from collections import defaultdict

import torch
import warnings
warnings.filterwarnings('ignore')

import numpy as np
torch.backends.cudnn.benchmark = True

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from scipy.linalg import block_diag
from torch.utils.data import Dataset, DataLoader
#from bernstein import bernstesin_coeff_order10_new

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.visualization.visualize_sequences import viz_sequence
avm = ArgoverseMap()

num = 10

data_path="/datasets/argoverse/val/data"
output_dir="../results/"
t_obs=20
dt=0.3
t_obs=20
pred=False
pred_array=None
batch_size = 512
dpi=50
w,h=200,200
paths = glob.glob(os.path.join(data_path, "*.csv"))
color = {
    'polygon': '#e6cf93',
    'polygon-outline': '#e6cf93',
    'centerline': '#fceec7',
    'agent': 'blue',
    'av': 'grey',
    'other': 'grey',
    'outline': 'black'
}

avm = ArgoverseMap()

def denoise(gt_x, gt_y, w = 7):
    # denoising
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


for idx in range(len(paths)):
    path = paths[idx]
    dff = pd.read_csv(path)
    
    city = dff['CITY_NAME'].values[0]    
    
    agent_df = dff[dff['OBJECT_TYPE'] == 'AGENT']
    x_a = agent_df['X'].values
    y_a = agent_df['Y'].values    
    x_a, y_a = denoise(x_a, y_a)    
    
    av_df = dff[dff['OBJECT_TYPE'] == 'AV']
    x_av = av_df['X'].values
    y_av = av_df['Y'].values
    x_av, y_av = denoise(x_av, y_av)
    
    others_df = dff[dff['OBJECT_TYPE'] == 'OTHERS']
    others_dfs = np.array([v for k, v in others_df.groupby('TRACK_ID')], dtype=object)
    x_o = {}
    y_o = {}
    
    for other_df in others_dfs:
        x_other, y_other = other_df['X'].values, other_df['Y'].values
        x_other, y_other = denoise(x_other, y_other)
        x_o[other_df['TRACK_ID'].values[0]] = x_other
        y_o[other_df['TRACK_ID'].values[0]] = other_df['Y'].values  
    
    # group by timestamp
    dfs = [x for _, x in dff.groupby('TIMESTAMP')]    
    
    for ind, df in enumerate(dfs):
        agent_df = df[df['OBJECT_TYPE'] == 'AGENT']
        others_df = df[df['OBJECT_TYPE'] == 'OTHERS']
        others_dfs = [x for _, x in others_df.groupby('TRACK_ID')]
#         others_dfs = np.array([v for k, v in others_df.groupby('TRACK_ID')], dtype=object)
        av_df = df[df['OBJECT_TYPE'] == 'AV']

        # agent
        x_traj = agent_df['X'].values
        y_traj = agent_df['Y'].values
        
        offsets = [x_a[0], y_a[0]] # offsets for other agents
        
        fig = plt.figure(figsize=(200/dpi,200/dpi), dpi=dpi)
        # fig = plt.figure(figsize=(10, 10), dpi=dpi)
        
        x_off = 75
        y_off = 75   
        points = np.array([[x_a[20] - x_off, y_a[20] + y_off],[x_a[20] + x_off, y_a[20] + y_off], [x_a[20] + x_off, y_a[20] - y_off],[x_a[20] - x_off, y_a[20] - y_off],[x_a[20] - x_off, y_a[20] + y_off]])
        plt.fill(points[:, 0], points[:, 1], color=color['outline'], zorder=0)
        if ind < len(dfs) - 1:
            x_off = 0.75
            y_off = 1.25
            points = np.array([[x_traj[0] - x_off, y_traj + y_off],[x_traj[0] + x_off, y_traj + y_off], [x_traj[0] + x_off, y_traj - y_off],[x_traj[0] - x_off, y_traj - y_off],[x_traj[0] - x_off, y_traj + y_off]])
            
            
            theta = np.arctan2((y_a[ind + 1] - y_a[ind]) , (x_a[ind + 1] - x_a[ind])) - np.pi/2
            w = np.zeros(points.shape)
            A = np.matrix([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            points = points - np.array([x_traj[0], y_traj[0]])
            for i,v in enumerate(points): w[i] = A @ points[i]
            plt.fill(w[:, 0] + x_traj[0], w[:, 1] + y_traj[0], color=color['agent'], zorder=5)
        plt.scatter(x_traj[0], y_traj[0], color=color['agent'], label='end observed', zorder=5)

        # av
        x_traj = av_df['X'].values
        y_traj = av_df['Y'].values
        x_max, y_max = np.max(x_traj), np.max(y_traj)
        x_min, y_min = np.min(x_traj), np.min(y_traj)

        if ind < len(dfs) - 1:
            x_off = 0.75
            y_off = 1.25
            points = np.array([[x_traj[0] - x_off, y_traj + y_off],[x_traj[0] + x_off, y_traj + y_off], [x_traj[0] + x_off, y_traj - y_off],[x_traj[0] - x_off, y_traj - y_off],[x_traj[0] - x_off, y_traj + y_off]])
            theta = np.arctan2((y_av[ind + 1] - y_av[ind]) , (x_av[ind + 1] - x_av[ind])) - np.pi/2
            w = np.zeros(points.shape)
            A = np.matrix([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            points = points - np.array([x_traj[0], y_traj[0]])
            for i,v in enumerate(points): w[i] = A @ points[i]
            plt.fill(w[:, 0] + x_traj[0], w[:, 1] + y_traj[0], color=color['av'], zorder=4)
        
        plt.scatter(x_traj[-1], y_traj[-1], color=color['av'], zorder=4)

#         # others
        for indoo, other in enumerate(others_dfs):
            x_traj = other['X'].values
            y_traj = other['Y'].values
            indo = other['TRACK_ID'].values[0]
            if ind < len(dfs) - 1 and ind < len(x_o[indo]) - 1 and ind < len(y_o[indo]) - 1:
                x_off = 0.75
                y_off = 1.25
                points = np.array([[x_traj[0] - x_off, y_traj + y_off],[x_traj[0] + x_off, y_traj + y_off], [x_traj[0] + x_off, y_traj - y_off],[x_traj[0] - x_off, y_traj - y_off],[x_traj[0] - x_off, y_traj + y_off]])
                
                theta = np.arctan2((y_o[indo][ind + 1] - y_o[indo][ind]) , (x_o[indo][ind + 1] - x_o[indo][ind])) - np.pi/2
                w = np.zeros(points.shape)
                A = np.matrix([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
                points = points - np.array([x_traj[0], y_traj[0]])
                for i,v in enumerate(points): w[i] = A @ points[i]
                plt.fill(w[:, 0] + x_traj[0], w[:, 1] + y_traj[0], color=color['other'], zorder=4)
            

        # centerlines
        lane_centerlines = []    
        # Get lane centerlines which lie within the range of trajectories
        agent_df = df[df['OBJECT_TYPE'] == 'AGENT']
        gt_x = agent_df['X'].values
        gt_y = agent_df['Y'].values

        x_max, y_max = np.max(x_a) + 50, np.max(y_a) + 50
        x_min, y_min = np.min(x_a) - 50, np.min(y_a) - 50

        # print(x_max, x_min)
        # print(y_max, y_min)
        for arr in avm.find_local_lane_polygons([x_min, x_max, y_min, y_max], city):
            plt.fill(arr[:, 0], arr[:, 1], color=color['polygon'],zorder=0)

        for arr in avm.find_local_lane_polygons([x_min, x_max, y_min, y_max], city):
            plt.plot(arr[:, 0], arr[:, 1], color=color['polygon-outline'],zorder=1)

        seq_lane_props = avm.city_lane_centerlines_dict[city]
        for lane_id, lane_props in seq_lane_props.items():
            lane_cl = lane_props.centerline
            if (np.min(lane_cl[:, 0]) < x_max and np.min(lane_cl[:, 1]) < y_max and np.max(lane_cl[:, 0]) > x_min and np.max(lane_cl[:, 1]) > y_min):
                lane_centerlines.append(lane_cl)

        for lane_cl in lane_centerlines:
            plt.plot(lane_cl[:, 0], lane_cl[:, 1], color=color['centerline'], alpha=1, linewidth=1, zorder=2)

#         plt.legend()
        plt.xlim([x_a[20] - 50, x_a[20] + 50])
        plt.ylim([y_a[20] - 50, y_a[20] + 50])
        import os
        try:
            os.mkdir('./results/{}'.format(idx))
        except:
            pass
#         plt.set_facecolor('red')
        plt.axis('off')
        plt.savefig('./results/{}/{}.png'.format(idx,ind), dpi=dpi, bbox_inches='tight')
        # fig.canvas.draw()
        # data_image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        # data_image = data_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # np.save('./results/{}/{}.npy'.format(idx,ind),  data_image)
        # print(data_image.shape)
        plt.clf()