#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
sys.path.append("./")
import warnings
warnings.filterwarnings('ignore')

import torch
import argparse
import numpy as np
import scipy.special
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.linalg import block_diag
from torch.utils.data import Dataset, DataLoader
from ddn.ddn.pytorch.node import AbstractDeclarativeNode

# from utils.nodes.OPTNode_waypoint import OPTNode_waypoint
from utils.nodes.OPTNode import OPTNode
from utils.dataloader import ArgoverseDataset

from utils.models.ddn import *

from utils.bernstein import bernstein_coeff_order10_new
from utils.viz_helpers import plot_traj, plot_trajj
from utils.metrics import get_ade, get_fde
from utils.args_parser import *

import pytorch_lightning as pl

use_cuda = torch.cuda.is_available()
device = torch.device('cpu')

model_dict = {
    "MLP": TrajNet,
    "LSTM": TrajNetLSTM,
    "LSTMSimple": TrajNetLSTMSimple,
    "LSTMSimpler": TrajNetLSTMSimpler,
    "LSTMPredHeading": TrajNetLSTMPredFinalHead,
    "LSTMEP": TrajNetLSTMEP
}

args = parse_arguments()
network_type = args.network
Model = model_dict[network_type]
num = args.pred_len
t_obs = args.obs_len
num_elems = args.num_elems
include_centerline = args.include_centerline
name = "final_without_ok" if include_centerline else "final_with"
lr = args.lr
shuffle = args.shuffle
num_waypoints = args.num_waypoints
flatten = args.flatten
num_epochs = args.end_epoch
batch_size = args.train_batch_size
num_workers = 10
saved_model = args.model_path

train_dir = args.train_dir
centerline_train_dir = args.train_centerlines_dir
if args.test:
    test_dir = args.test_dir
    centerline_test_dir = args.test_centerlines_dir
else:
    test_dir = args.train_dir
val_offsets_dir = args.val_offsets_dir

# In[2]:
train_dataset = ArgoverseDataset(train_dir, centerline_dir=centerline_train_dir, t_obs=t_obs, dt=0.3, include_centerline = include_centerline, flatten = flatten, end_point=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

test_dataset = ArgoverseDataset(test_dir, centerline_dir=centerline_test_dir, t_obs=t_obs, dt=0.3, include_centerline = include_centerline, flatten = flatten, end_point=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

offsets_train = np.load(val_offsets_dir)

problem = OPTNode(rho_eq=10, t_fin=9.0, num=num, bernstein_coeff_order10_new=bernstein_coeff_order10_new, device = device)
opt_layer = DeclarativeLayer(problem)

# model = TrajNetLSTM(opt_layer, problem.P, problem.Pdot)#, input_size=t_obs * 2 + include_centerline * num_elems * 2)
# model = TrajNet(opt_layer, problem.P, problem.Pdot)#, input_size=t_obs * 2 + include_centerline * num_elems * 2)
if flatten:
    model = Model(opt_layer, problem.P, problem.Pdot, input_size=t_obs * 2 + include_centerline * num_elems * 2, device=device)
else:
    model = Model(opt_layer, problem.P, problem.Pdot, device=device)
# model = TrajNet(opt_layer, problem.P, problem.Pdot, input_size=t_obs * 2 + include_centerline * num_elems * 2, output_size = num_waypoints * 2 + 2)
#model = torch.nn.DataParallel(model, device_ids=[0])
model = model.double()
model = model.to(device)

if args.model_path and args.test:
    model.load_state_dict(torch.load(args.model_path))

# model.load_state_dict(torch.load("./checkpoints/final.ckpt"))

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lr)

# In[5]:


epoch_train_loss = []

for epoch in range(num_epochs):
    train_loss = []
    mean_ade = []
    mean_fde = []  
    mean_head_loss = []   
    for batch_num, data in enumerate(tqdm(train_loader)):
        traj_inp, traj_out, fixed_params, var_inp = data
        traj_inp = traj_inp.to(device)
        traj_out = traj_out.to(device)
        fixed_params = fixed_params.to(device)
        var_inp = var_inp.to(device)

        ade = []
        fde = []
        head_loss = []        
#         out = model(traj_inp.float(), fixed_params.float(), var_inp.float())
        out = model(traj_inp, fixed_params, var_inp)
        loss = criterion(out, traj_out)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        for ii in range(traj_inp.size()[0]):
            fde.append(np.linalg.norm( out[ii][1:].detach().numpy() - traj_out[ii][1:].detach().numpy()))
            head_loss.append(np.linalg.norm(out[ii][0].detach().numpy() - traj_out[ii][0].detach().numpy()))
#            plot_trajj(ii, traj_inp[ii], traj_out[ii], out[ii], {"x": [], "y": []}, offsets=offsets_train, cities = [], avm=None, center=include_centerline, inp_len=t_obs * 2, c_len = t_obs * 2 + num_elems * 2, num=num, mode="test", batch_num=batch_num)
        if batch_num % 10 == 0:
            print("Epoch: {}, Batch: {}, Loss: {}".format(epoch, batch_num, loss.item()))
            print("Head loss: {}".format(np.mean(head_loss)), "FDE: {}".format(np.mean(fde)))

        mean_fde.append(np.mean(fde))
        mean_head_loss.append(np.mean(head_loss))
    mean_loss = np.mean(train_loss)
    epoch_train_loss.append(mean_loss)
    torch.save(model.state_dict(), "./checkpoints/{}.ckpt".format(name))
    print("Epoch: {}, Mean Loss: {}".format(epoch, mean_loss))
    print("Mean Heading Loss: {}".format(np.mean(mean_head_loss)))
    print("-"*100)


# In[ ]:
if args.test is False:
    exit()

with torch.no_grad():
    cnt = 0
    test_loss = []
    mean_ade = []
    mean_fde = []   
    mean_head_loss = []  
    for batch_num, data in enumerate(test_loader):
        traj_inp, traj_out, fixed_params, var_inp = data
        traj_inp = traj_inp.to(device)
        traj_out = traj_out.to(device)
        fixed_params = fixed_params.to(device)
        var_inp = var_inp.to(device)
        
        ade = []
        fde = []
        head_loss = []       
        
        out = model(traj_inp, fixed_params, var_inp)
        loss = criterion(out, traj_out)
        
        test_loss.append(loss.item())
        print("Batch: {}, Loss: {}".format(batch_num, loss.item()))
        
        for ii in range(traj_inp.size()[0]):
            fde.append(np.linalg.norm( out[ii][1:] - traj_out[ii][1:]))
            head_loss.append(np.linalg.norm(out[ii][0] - traj_out[ii][0]))

        mean_fde.append(np.mean(fde))  
        mean_head_loss.append(np.mean(head_loss))
mean_loss = np.mean(test_loss)
print("Epoch Mean Test Loss: {}".format(mean_loss))
print("Mean Heading Loss: {}".format(np.mean(mean_head_loss)))
