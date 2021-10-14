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
from utils.viz_helpers import vis_trajectories
from utils.metrics import get_ade, get_fde
from utils.args_parser import *

#import pytorch_lightning as pl

use_cuda = torch.cuda.is_available()
#device = torch.device('cpu')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_dict = {
    "MLP": TrajNet,
    "LSTM": TrajNetLSTM,
    "LSTMSimple": TrajNetLSTMSimple,
    "LSTMSimpler": TrajNetLSTMSimpler,
    "LSTMPredHeading": TrajNetLSTMPredFinalHead,
    "LSTMEP": TrajNetLSTMEP
}

args = parse_arguments()
Model = model_dict[args.network]

# In[2]:
train_dataset = ArgoverseDataset(args.train_dir, t_obs=args.obs_len, dt=0.3, include_centerline=args.include_centerline, flatten=args.flatten, end_point=args.end_point)
train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

problem = OPTNode(rho_eq=10, t_fin=9.0, num=args.pred_len, bernstein_coeff_order10_new=bernstein_coeff_order10_new, device = device)
opt_layer = DeclarativeLayer(problem)

if args.flatten:
    model = Model(opt_layer, problem.P, problem.Pdot, input_size=args.obs_len * 2 + args.include_centerline * args.num_elems * 2, device=device)
else:
    model = Model(opt_layer, problem.P, problem.Pdot, device=device)

model = model.double()
model = model.to(device)
model = torch.nn.DataParallel(model, device_ids=[0])

if args.model_path:
    model.load_state_dict(torch.load(args.model_path))

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)

# In[5]:

epoch_train_loss = []

for epoch in range(args.end_epoch):
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
            if args.end_point:
                fde.append(np.linalg.norm( out[ii][1:].detach().numpy() - traj_out[ii][1:].detach().numpy()))
                head_loss.append(np.linalg.norm(out[ii][0].detach().numpy() - traj_out[ii][0].detach().numpy()))
            else:  
                gt = [[out[ii][j].item(),out[ii][j + args.pred_len].item()] for j in range(len(out[ii])//2)]
                pred = [[traj_out[ii][j].item(),traj_out[ii][j + args.pred_len].item()] for j in range(len(out[ii])//2)] 
                fde.append(get_fde(np.array(pred), np.array(gt)))                                        
                ade.append(get_ade(np.array(pred), np.array(gt)))
        if batch_num % 10 == 0:
            print("Epoch: {}, Batch: {}, Loss: {}".format(epoch, batch_num, loss.item()))
            if args.end_point:
                print("Head loss: {}".format(np.mean(head_loss)), "FDE: {}".format(np.mean(fde)))
            else:
                print("ADE: {}".format(np.mean(ade)), "FDE: {}".format(np.mean(fde)))
        mean_ade.append(np.mean(ade))
        mean_fde.append(np.mean(fde))
        mean_head_loss.append(np.mean(head_loss))
    mean_loss = np.mean(train_loss)
    epoch_train_loss.append(mean_loss)
    torch.save(model.state_dict(), "./checkpoints/final.ckpt")
    print("Epoch: {}, Mean Loss: {}".format(epoch, mean_loss))
    if args.end_point:
        print("Mean Heading Loss: {}".format(np.mean(mean_head_loss)))
    else:
        print("Mean ADE: {}".format(np.mean(mean_ade)), "Mean FDE: {}".format(np.mean(mean_fde)))
    print("-"*100)

# In[ ]:
if args.test is False:
    exit()

test_dataset = ArgoverseDataset(args.test_dir, t_obs=args.obs_len, dt=0.3, include_centerline=args.include_centerline, flatten=args.flatten, end_point=args.end_point)
test_loader = DataLoader(test_dataset, batch_size=args.train_batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

predictions = np.empty((1,args.pred_len, 2))
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
        x_out = out[:, :args.pred_len]
        y_out = out[:, args.pred_len:]
        predictions = np.append(predictions, np.dstack((x_out, y_out)),0)
        loss = criterion(out, traj_out)
        
        test_loss.append(loss.item())
        print("Batch: {}, Loss: {}".format(batch_num, loss.item()))
 
        for ii in range(traj_inp.size()[0]):
            if args.end_point:
                fde.append(np.linalg.norm( out[ii][1:] - traj_out[ii][1:]))
                head_loss.append(np.linalg.norm(out[ii][0].detach().numpy() - traj_out[ii][0].detach().numpy()))
            else:  
                gt = [[out[ii][j].item(),out[ii][j + args.pred_len].item()] for j in range(len(out[ii])//2)]
                pred = [[traj_out[ii][j].item(),traj_out[ii][j + args.pred_len].item()] for j in range(len(out[ii])//2)]
                fde.append(get_fde(np.array(pred), np.array(gt)))
                ade.append(get_ade(np.array(pred), np.array(gt)))            
        mean_ade.append(np.mean(ade))
        mean_fde.append(np.mean(fde))  
        mean_head_loss.append(np.mean(head_loss))
mean_loss = np.mean(test_loss)
predictions = predictions[1:]
np.save("predictions.npy", predictions)
print("Epoch Mean Test Loss: {}".format(mean_loss))
if args.end_point:
    print("Mean Heading Loss: {}".format(np.mean(mean_head_loss)))
else:
    print("Mean ADE: {}".format(np.mean(mean_ade)), "Mean FDE: {}".format(np.mean(mean_fde)))
vis_trajectories(args.test_dir, args.traj_save_path, pred=True, pred_array=predictions)
