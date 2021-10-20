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
from math import exp

import numpy as np
torch.backends.cudnn.benchmark = True

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.visualization.visualize_sequences import viz_sequence
avm = ArgoverseMap()

num = 10

data_path="/datasets/argoverse/val/data"
infer_path="../../inn"


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

from shapely.geometry.polygon import Polygon, Point

output_dir="../results/"
t_obs=20
dt=0.3
t_obs=20
pred=False
pred_array=None
batch_size = 512
dpi=100
w,h=512,512
res=0.5
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
color = {
    'polygon': 'white',
    'polygon-outline': 'white',
    'centerline': 'white',
    'agent': 'white',
    'av': 'white',
    'other': 'white',
    'outline': 'black'
}

from tqdm import tqdm
for idx in tqdm(range(len(paths))):
    if idx < 19:
        continue
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

    
    grids_lanes = np.zeros((20, h, w))
    grids_obstacles = np.zeros((20, h, w))
    grids_centerlines = np.zeros((20, h, w))
    grids_agent = np.zeros((20, h, w))
    
    
    total_successors = []
    current = []
    das_polygons = []
    das_polygons_mp = []
    das_ids = []

    agent_polygons = []
    
    others_polygons = []
    
    for indd in range(0, 20):
        lane_id = avm.get_nearest_centerline(np.array([x_a[indd],y_a[indd]]), city_name=city)[0].id
        current.append(lane_id)
        successors = avm.get_lane_segment_successor_ids(lane_id, city)
        if successors == None:
            continue
        for successor in successors:
            total_successors.append(successor)
            successors_2d = avm.get_lane_segment_successor_ids(successor, city)
            for successorr in successors_2d:
                if successors_2d == None:
                    continue                    
                total_successors.append(successorr)
        polygons = [ avm.get_lane_segment_polygon(successor, city) for successor in successors]
    current = np.unique(np.array(current))
    total_successors = np.unique(np.array(total_successors))    
    for curr in current:
        current_polygon = avm.get_lane_segment_polygon(curr, city)
        das_polygons.append(current_polygon)
        das_polygons_mp.append(avm.get_lane_segment_polygon(curr, city))
        das_ids.append(curr)
#         plt.fill(current_polygon[:, 0], current_polygon[:, 1], color='white', zorder=4)
    for successor in total_successors : 
        polygon = avm.get_lane_segment_polygon(successor, city)
        das_polygons.append(polygon)
        das_polygons_mp.append(avm.get_lane_segment_polygon(successor, city))
        das_ids.append(successor)
#         plt.fill(polygon[:, 0], polygon[:, 1], color='white', zorder=4)
    das_polygons_mp = np.array(das_polygons_mp)
    x_off = 75
    y_off = 75
    points = np.array([[x_a[20] - x_off, y_a[20] + y_off],[x_a[20] + x_off, y_a[20] + y_off], [x_a[20] + x_off, y_a[20] - y_off],[x_a[20] - x_off, y_a[20] - y_off],[x_a[20] - x_off, y_a[20] + y_off]])

    
    for ind, df in enumerate(dfs):
        agent_df = df[df['OBJECT_TYPE'] == 'AGENT']
        others_df = df[df['OBJECT_TYPE'] == 'OTHERS']
        others_dfs = [x for _, x in others_df.groupby('TRACK_ID')]
        av_df = df[df['OBJECT_TYPE'] == 'AV']

        # agent
        x_traj = agent_df['X'].values
        y_traj = agent_df['Y'].values
        offsets = [x_a[0], y_a[0]] # offsets for other agents
        others_polyon = []
        if ind < len(dfs) - 1:
            x_off = 2 #0.75
            y_off = 2.25 #1.25
            points = np.array([[x_traj[0] - x_off, y_traj + y_off],[x_traj[0] + x_off, y_traj + y_off], [x_traj[0] + x_off, y_traj - y_off],[x_traj[0] - x_off, y_traj - y_off],[x_traj[0] - x_off, y_traj + y_off]])
            theta = np.arctan2((y_a[ind + 1] - y_a[ind]) , (x_a[ind + 1] - x_a[ind])) - np.pi/2
            ww = np.zeros(points.shape)
            A = np.matrix([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            points = points - np.array([x_traj[0], y_traj[0]])
            for i,v in enumerate(points): ww[i] = A @ points[i]
            ww[:, 0] += x_traj[0]
            ww[:, 1] += y_traj[0]
            try:
                agent_polygons.append(Polygon(ww))
            except:
                print("AGENT problem")

        for indoo, other in enumerate(others_dfs):
            x_traj = other['X'].values
            y_traj = other['Y'].values
            indo = other['TRACK_ID'].values[0]
            if ind < len(dfs) - 1 and ind < len(x_o[indo]) - 1 and ind < len(y_o[indo]) - 1:
                x_off = 2
                y_off = 2.25
                points = np.array([[x_traj[0] - x_off, y_traj + y_off],[x_traj[0] + x_off, y_traj + y_off], [x_traj[0] + x_off, y_traj - y_off],[x_traj[0] - x_off, y_traj - y_off],[x_traj[0] - x_off, y_traj + y_off]])
                
                theta = np.arctan2((y_o[indo][ind + 1] - y_o[indo][ind]) , (x_o[indo][ind + 1] - x_o[indo][ind])) - np.pi/2
                ww = np.zeros(points.shape)
                A = np.matrix([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
                points = points - np.array([x_traj[0], y_traj[0]])
                for i,v in enumerate(points): ww[i] = A @ points[i]
                ww[:, 0] += x_traj[0]
                ww[:, 1] += y_traj[0]
                try:
                    others_polyon.append(Polygon(ww))
                except:
                    print("OTHERS")
        others_polygons.append(others_polyon)
        
    sample = np.zeros((h, w))
    lx = x_a[20] - res*(h/2)
    ly = y_a[20] - res*(w/2)
    
#     seq_lane_props = avm.city_lane_centerlines_dict[city]
#     for lane_id, lane_props in seq_lane_props.items():
#         lane_cl = lane_props.centerline
#         if (np.min(lane_cl[:, 0]) < x_max and np.min(lane_cl[:, 1]) < y_max and np.max(lane_cl[:, 0]) > x_min and np.max(lane_cl[:, 1]) > y_min):
#             lane_centerlines.append(lane_cl)

    
    for i in tqdm(range(h)):
        for j in range(w):
            px = lx + i * res
            py = ly + j * res
            point_xy = Point(px, py)
            flag = 0
            for k in range(len(das_polygons)):
                if Polygon(das_polygons[k]).contains(point_xy):
                    flag = 1
            sample[j,i] = flag
            
            for k in range(20):
                # get obstacle polygon
                for l in range(len(others_polygons[k])):
                    if others_polygons[k][l].contains(point_xy):
                        grids_obstacles[k, j, i] = 1
                        
                # get agent polygon
                if agent_polygons[k].contains(point_xy):
                    grids_agent[k, j, i] = 1

    print("DONE")

    print(grids_agent.shape)
    for i in range(20): grids_lanes[i] = sample
    print(str(infer_path) + "/das/{}.npy".format(idx))
    np.save(str(infer_path) + "/das/{}.npy".format(idx), grids_lanes)
    np.save(str(infer_path) + "/agents/{}.npy".format(idx), grids_agent)
    np.save(str(infer_path) + "/others/{}.npy".format(idx), grids_obstacles)

