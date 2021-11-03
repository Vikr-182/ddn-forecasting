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

from utils.viz_helpers import *

avm = ArgoverseMap()

num = 10

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

predictions = np.load("predictions.py.npy")
sequences = np.load("sequences.npy")
gt = np.load("/datasets/argoverse/val_data.npy", allow_pickle=True)
vis_trajectories(data_path = "/datasets/argoverse/val/data", output_dir="./results/", t_obs=20, pred=True, pred_array=predictions, sequences=sequences)
