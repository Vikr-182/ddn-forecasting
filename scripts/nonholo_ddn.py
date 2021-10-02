#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
# sys.path.append("/Users/shashanks./Downloads/Installations/ddn/")
sys.path.append("./ddn/")
sys.path.append("./")
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import scipy.special
import torch.nn as nn
import matplotlib.pyplot as plt

from scipy.linalg import block_diag
from torch.utils.data import Dataset, DataLoader
from bernstein import bernstein_coeff_order10_new
from ddn.pytorch.node import AbstractDeclarativeNode

from utils.viz_helpers import plot_traj
from utils.metrics import get_ade, get_fde


# In[2]:


from utils.dataloader import TrajectoryDataset

train_dataset = TrajectoryDataset("/datasets/argoverse/val_data.npy")
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=False, num_workers=0)

test_dataset = TrajectoryDataset("/datasets/argoverse/val_test_data.npy")
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=0)


# In[3]:


import json


# In[4]:


# print(np.array(data['x_init']).shape)
# print(np.array(data['y_init']).shape)
# print(np.array(data['v_init']).shape)

ind = [8, 12, 19, 21, 27, 29, 34, 48, 77, 94, 105, 116, 122, 145, 163, 165, 229, 252, 272, 326]
others = [386, 409, 411]


# In[5]:


np.arctan(1/-1)


# In[199]:


cnt = 0
x_s = []
y_s = []

x_init_s = []
y_init_s = []
vx_init_s = []
vy_init_s = []
psi_init_s = []
psidot_init_s = []

#['x_init', 'y_init', 'v_init', 'psi_init', 'psidot_init', 'x_fin', 'y_fin', 'psi_fin', 'psidot_fin'])

x_fin_s = []
y_fin_s = []
vx_fin_s = []
vy_fin_s = []
psi_fin_s = []
psidot_fin_s = []

dt = 0.1

xs = []
vs = []
psis = []
psidots = []
for i, data in enumerate(train_loader):
    traj_inp, traj_out, b_inp = data
    for ii in range(traj_inp.shape[0]):
#         gt = [[out[ii][j],out[ii][j + num]] for j in range(len(out[ii])//2)]
        if i * 20 + ii in ind:
            cnt = cnt + 1
            
            inp_x = traj_inp[ii,:40:2].detach()
            inp_y = traj_inp[ii,1:40:2].detach()
            gt_x = [traj_out[ii][j] for j in range(len(traj_out[ii])//2)]
            gt_y = [traj_out[ii][j + 30] for j in range(len(traj_out[ii])//2)]
            
            if cnt is 12:
                plt.plot(gt_x, gt_y, label='gt_act' + str(i * 20 + ii))
            
            v_x = [ (gt_x[k + 1] - gt_x[k])/dt  for k in range(len(gt_x) - 1)]
            v_y = [ (gt_y[k + 1] - gt_y[k])/dt  for k in range(len(gt_y) - 1)]
#             psi = [ np.arctan(v_y[k]/v_x[k]) if v_x[k] > 0 else (np.arctan(v_y[k]/v_x[k]) + np.pi) for k in range(len(v_x))]
            psi = [ np.arctan2(v_y[k], v_x[k]) for k in range(len(v_x))]
            
#             # denoising
            w = 7
            gt_x_t = []
            gt_y_t = []
            for iq in range(len(gt_x)):
                if iq >= w and iq + w <= len(gt_x):
                    gt_x_t.append(np.average(gt_x[iq: iq + w]))
                    gt_y_t.append(np.average(gt_y[iq: iq + w]))
                elif iq < w:
                    okx = np.average(gt_x[w: w + w])
                    gt_x_t.append(gt_x[0] + (okx - gt_x[0]) * (iq) / w)
                    oky = np.average(gt_y[w: w + w])
                    gt_y_t.append(gt_y[0] + (oky - gt_y[0]) * (iq) / w)
                else:
                    okx = np.average(gt_x[len(gt_x) - w:len(gt_x) - w  + w])
                    oky = np.average(gt_y[len(gt_x) - w: len(gt_x) - w + w])
                    gt_x_t.append(okx + (gt_x[-1] - okx) * (w - (len(gt_x) - iq)) / w)
                    gt_y_t.append(oky + (gt_y[-1] - oky) * (w - (len(gt_y) - iq)) / w)
                    if i * 20 + ii is 12:
                        print(okx, gt_x[-1])                    
    
            gt_x = gt_x_t
            gt_y = gt_y_t
            
            v_x = [ (gt_x[k + 1] - gt_x[k])/dt  for k in range(len(gt_x) - 1)]
            v_y = [ (gt_y[k + 1] - gt_y[k])/dt  for k in range(len(gt_y) - 1)]
#             psi = [ np.arctan(v_y[k]/v_x[k]) if v_x[k] > 0 else (np.arctan(v_y[k]/v_x[k]) + np.pi) for k in range(len(v_x))]
            psi = [ np.arctan2(v_y[k], v_x[k]) for k in range(len(v_x))]
        
            # rotate by -psi[0]
            theta = -psi[0]
            gt_x_x = [ (gt_x[k] * np.cos(theta) - gt_y[k] * np.sin(theta))  for k in range(len(gt_x))]
            gt_y_y = [ (gt_x[k] * np.sin(theta) + gt_y[k] * np.cos(theta))  for k in range(len(gt_x))]
            gt_x = gt_x_x
            gt_y = gt_y_y
            gt_x = [ gt_x[k] - gt_x[0] for k in range(len(gt_x)) ]
            gt_y = [ gt_y[k] - gt_y[0] for k in range(len(gt_x)) ]
            
            if cnt is 12:
                plt.plot(gt_x, gt_y, label='gt' + str(i * 20 + ii))                        
            
            
            v_x = [ (gt_x[k + 1] - gt_x[k])/dt  for k in range(len(gt_x) - 1)]
            v_y = [ (gt_y[k + 1] - gt_y[k])/dt  for k in range(len(gt_y) - 1)]    
      
            psi = [ np.arctan2(v_y[k], v_x[k]) for k in range(len(v_x))]
            psidot = [ (psi[k + 1] - psi[k])/dt for k in range(len(psi) - 1) ]
            psi = [i.item() for i in psi]
            psidot = [i.item() for i in psidot]
            
            x_init = gt_x[0]
            y_init = gt_y[0]
            
            vx_init = v_x[0]
            vy_init = v_y[0]
            
            vx_fin = v_x[len(v_x) - 1]
            vy_fin = v_y[len(v_y) - 1]
            
            psi_init = psi[0]
            psi_fin = psi[len(psi) - 1]
            
            psidot_init = psi[0]
            psidot_fin = psidot[len(psidot) - 1]
            
            x_fin = gt_x[29]
            y_fin = gt_y[29]
            
            
            x_init_s.append([x_init])
            y_init_s.append([y_init])
            
            x_fin_s.append([x_fin])
            y_fin_s.append([y_fin])
            
            vx_init_s.append([vx_init])
            vy_init_s.append([vy_init])
            
            vx_fin_s.append([vx_fin])
            vy_fin_s.append([vy_fin])
            
            psi_init_s.append([psi_init])
            psi_fin_s.append([psi_fin])

            psidot_init_s.append([psidot_init])
            psidot_fin_s.append([psidot_fin])
            
            print(i * 20 + ii)
#             if i * 20 + ii is 19:

#                 print(psi[0])
#                 print(gt_x);print(gt_y)
#             plt.legend()
            
            xs.append([gt_x, gt_y])
            vs.append([v_x, v_y])
            psis.append([psi])
            psidots.append([psidot])
#             print(arr)
#         plot_traj(ii, traj_inp[ii], traj_out[ii], out[ii], {"x": [], "y": []}, offsets=offsets_train, cities = cities, avm=None, center=False, inp_len=20 * 2, c_len = 20 * 2 + num_elems * 2, num=30, mode="test", batch_num = i)

x_init_s = [[i[0].item()] for i in x_init_s]
y_init_s = [[i[0].item()] for i in y_init_s]
vx_init_s = [[i[0].item()] for i in vx_init_s]
vy_init_s = [[i[0].item()] for i in vy_init_s]

x_fin_s = [[i[0].item()] for i in x_fin_s]
y_fin_s = [[i[0].item()] for i in y_fin_s]
vx_fin_s = [[i[0].item()] for i in vx_fin_s]
vy_fin_s = [[i[0].item()] for i in vy_fin_s]


dic = {
    'x_init': (x_init_s),
    'y_init': (y_init_s),
    'vx_init': (vx_init_s),
    'vy_init': (vy_init_s),
    'psi_init': (psi_init_s), 
    'psidot_init': (psidot_init_s),
    'x_fin': (x_fin_s),
    'y_fin': (y_fin_s), 
    'psi_fin': (psi_fin_s), 
    'psidot_fin': (psidot_fin_s) 
}

plt.axis('equal')

with open("inp.json", "w") as f:
    json.dump(dic, f)
# #['x_init', 'y_init', 'v_init', 'psi_init', 'psidot_init', 'x_fin', 'y_fin', 'psi_fin', 'psidot_fin'])


# In[192]:


xs = np.array(xs)
vs = np.array(vs)
psis = np.array(psis)
psidots = np.array(psidots)
print(np.max(psis))

print(xs.shape, vs.shape, psis.shape, psidots.shape)


# In[193]:


np.save("data/pos_data.npy", np.array(xs))
np.save("data/vel_data.npy", np.array(xs))
np.save("data/psi_data.npy", np.array(psis))
np.save("data/psidot_data.npy", np.array(psidots))
print(np.array(psis).shape)
# plt.plot(arr[:, 0], arr[ :, 1])

# np.save("data/pos_data_.npy", np.array(xs))
# np.save("data/vel_data_.npy", np.array(xs))
# np.save("data/psi_data_.npy", np.array(psis))
# np.save("data/psidot_data_.npy", np.array(psidots))
# print(np.array(psis).shape)
# plt.plot(arr[:, 0], arr[ :, 1])


# In[39]:


with open("./tempinp.json") as f:
    rdata = json.load(f)
    
print(rdata.keys())
print(np.array(rdata['psidot_init']).shape)
print(dic.keys())
print(np.array(dic['psidot_init']).shape)


# In[40]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


# In[12]:


class OPTNode(AbstractDeclarativeNode):
    def __init__(self, rho_eq=1.0, rho_goal=1.0, rho_nonhol=1.0, rho_psi=1.0, maxiter=500, weight_smoothness=5.0, weight_smoothness_psi=5.0, t_fin=8.0, num=16):
        super().__init__()
        self.rho_eq = rho_eq
        self.rho_goal = rho_goal
        self.rho_nonhol = rho_nonhol
        self.rho_psi = rho_psi
        self.maxiter = maxiter
        self.weight_smoothness = weight_smoothness
        self.weight_smoothness_psi = weight_smoothness_psi

        self.t_fin = t_fin
        self.num = num
        self.t = self.t_fin / self.num

        #self.num_batch = 10
        
        tot_time = np.linspace(0.0, self.t_fin, self.num)
        tot_time_copy = tot_time.reshape(self.num, 1)
        self.P, self.Pdot, self.Pddot = bernstein_coeff_order10_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)
        self.nvar = np.shape(self.P)[1]
        
        self.cost_smoothness = self.weight_smoothness * np.dot(self.Pddot.T, self.Pddot)
        self.cost_smoothness_psi = self.weight_smoothness_psi * np.dot(self.Pddot.T, self.Pddot)
        self.lincost_smoothness_psi = np.zeros(self.nvar)

        self.A_eq = np.vstack((self.P[0], self.P[-1]))
        self.A_eq_psi = np.vstack((self.P[0], self.Pdot[0], self.P[-1], self.Pdot[-1]))
        
        self.P = torch.tensor(self.P, dtype=torch.double).to(device)
        self.Pdot = torch.tensor(self.Pdot, dtype=torch.double).to(device)
        self.Pddot = torch.tensor(self.Pddot, dtype=torch.double).to(device)
        self.A_eq = torch.tensor(self.A_eq, dtype=torch.double).to(device)        
        self.A_eq_psi = torch.tensor(self.A_eq_psi, dtype=torch.double).to(device)
        self.cost_smoothness = torch.tensor(self.cost_smoothness, dtype=torch.double).to(device)
        self.cost_smoothness_psi = torch.tensor(self.cost_smoothness_psi, dtype=torch.double).to(device)
        self.lincost_smoothness_psi = torch.tensor(self.lincost_smoothness_psi, dtype=torch.double).to(device)
        
        self.A_nonhol = self.Pdot
        self.A_psi = self.P
    
    def compute_x(self, lamda_x, lamda_y, v, psi, b_eq_x, b_eq_y):
        b_nonhol_x = v * torch.cos(psi)
        b_nonhol_y = v * torch.sin(psi)
    
        cost = self.cost_smoothness + self.rho_nonhol * torch.matmul(self.A_nonhol.T, self.A_nonhol) + self.rho_eq * torch.matmul(self.A_eq.T, self.A_eq)
        lincost_x = -lamda_x - self.rho_nonhol * torch.matmul(self.A_nonhol.T, b_nonhol_x.T).T - self.rho_eq * torch.matmul(self.A_eq.T, b_eq_x.T).T
        lincost_y = -lamda_y - self.rho_nonhol * torch.matmul(self.A_nonhol.T, b_nonhol_y.T).T - self.rho_eq * torch.matmul(self.A_eq.T, b_eq_y.T).T

        cost_inv = torch.linalg.inv(cost)

        sol_x = torch.matmul(-cost_inv, lincost_x.T).T
        sol_y = torch.matmul(-cost_inv, lincost_y.T).T

        x = torch.matmul(self.P, sol_x.T).T
        xdot = torch.matmul(self.Pdot, sol_x.T).T

        y = torch.matmul(self.P, sol_y.T).T
        ydot = torch.matmul(self.Pdot, sol_y.T).T
         
        return sol_x, sol_y, x, xdot, y, ydot
    
    def compute_psi(self, psi, lamda_psi, psi_temp, b_eq_psi):
        cost = self.cost_smoothness_psi + self.rho_psi * torch.matmul(self.A_psi.T, self.A_psi) + self.rho_eq * torch.matmul(self.A_eq_psi.T, self.A_eq_psi)
        lincost_psi = -lamda_psi - self.rho_psi * torch.matmul(self.A_psi.T, psi_temp.T).T - self.rho_eq * torch.matmul(self.A_eq_psi.T, b_eq_psi.T).T

        cost_inv = torch.linalg.inv(cost)

        sol_psi = torch.matmul(-cost_inv, lincost_psi.T).T

        psi = torch.matmul(self.P, sol_psi.T).T

        res_psi = torch.matmul(self.A_psi, sol_psi.T).T - psi_temp
        res_eq_psi = torch.matmul(self.A_eq_psi, sol_psi.T).T - b_eq_psi

        lamda_psi = lamda_psi - self.rho_psi * torch.matmul(self.A_psi.T, res_psi.T).T - self.rho_eq * torch.matmul(self.A_eq_psi.T, res_eq_psi.T).T

        return sol_psi, np.linalg.norm(res_psi), np.linalg.norm(res_eq_psi), psi, lamda_psi

    
    def solve(self, fixed_params, variable_params):
        batch_size, _ = fixed_params.size()
        x_init, y_init, v_init, psi_init, psidot_init = torch.chunk(fixed_params, 5, dim=1)
        x_fin, y_fin, psi_fin, psidot_fin = torch.chunk(variable_params, 4, dim=1)
        
        b_eq_x = torch.cat((x_init, x_fin), dim=1)
        b_eq_y = torch.cat((y_init, y_fin), dim=1)
        b_eq_psi = torch.cat((psi_init, psidot_init, psi_fin, psidot_fin), dim=1)
        
        lamda_x = torch.zeros(batch_size, self.nvar, dtype=torch.double).to(device)
        lamda_y = torch.zeros(batch_size, self.nvar, dtype=torch.double).to(device)
        lamda_psi = torch.zeros(batch_size, self.nvar, dtype=torch.double).to(device)
        
        v = torch.ones(batch_size, self.num, dtype=torch.double).to(device) * v_init
        psi = torch.ones(batch_size, self.num, dtype=torch.double).to(device) * psi_init
        xdot = v * torch.cos(psi)
        ydot = v * torch.sin(psi)
        
        for i in range(0, self.maxiter):
            psi_temp = torch.atan2(ydot, xdot)
            c_psi, _, _, psi, lamda_psi = self.compute_psi(psi, lamda_psi, psi_temp, b_eq_psi)
            c_x, c_y, x, xdot, y, ydot = self.compute_x(lamda_x, lamda_y, v, psi, b_eq_x, b_eq_y)
            
            v = torch.sqrt(xdot ** 2 + ydot ** 2)
            #v[:, 0] = v_init[:, 0]

            res_eq_x = torch.matmul(self.A_eq, c_x.T).T - b_eq_x
            res_nonhol_x = xdot - v * torch.cos(psi)

            res_eq_y = torch.matmul(self.A_eq, c_y.T).T - b_eq_y
            res_nonhol_y = ydot - v * torch.sin(psi)

            lamda_x = lamda_x - self.rho_eq * torch.matmul(self.A_eq.T, res_eq_x.T).T - self.rho_nonhol * torch.matmul(self.A_nonhol.T, res_nonhol_x.T).T
            lamda_y = lamda_y - self.rho_eq * torch.matmul(self.A_eq.T, res_eq_y.T).T - self.rho_nonhol * torch.matmul(self.A_nonhol.T, res_nonhol_y.T).T
                    
        
        primal_sol = torch.hstack((c_x, c_y, c_psi, v))
        return primal_sol, None
    
    def objective(self, fixed_params, variable_params, y):
        c_x = y[:, :self.nvar]
        c_y = y[:, self.nvar:2*self.nvar]
        c_psi = y[:, 2*self.nvar:3*self.nvar]
        v = y[:, 3*self.nvar:]
        
        x_init, y_init, v_init, psi_init, psidot_init = torch.chunk(fixed_params, 5, dim=1)
        x_fin, y_fin, psi_fin, psidot_fin = torch.chunk(variable_params, 4, dim=1)
        
        x = torch.matmul(self.P, c_x.T).T
        y = torch.matmul(self.P, c_y.T).T
        psi = torch.matmul(self.P, c_psi.T).T
        xdot = torch.matmul(self.Pdot, c_x.T).T
        ydot = torch.matmul(self.Pdot, c_y.T).T
        psidot = torch.matmul(self.Pdot, c_psi.T).T
        xddot = torch.matmul(self.Pddot, c_x.T).T
        yddot = torch.matmul(self.Pddot, c_y.T).T
        psiddot = torch.matmul(self.Pddot, c_psi.T).T
        
        cost_nonhol = 0.5*self.rho_nonhol*torch.sum((xdot - v*torch.cos(psi)) ** 2, 1) + 0.5*self.rho_nonhol*torch.sum((ydot - v*torch.sin(psi)) ** 2, 1)
        cost_pos = 0.5*self.rho_eq*(torch.sum((x[:, -1] - x_fin) ** 2, 1) + torch.sum((y[:, -1] - y_fin) ** 2, 1) + torch.sum((x[:, 0] - x_init) ** 2, 1) + torch.sum((y[:, 0] - y_init) ** 2, 1))
        cost_psi = 0.5*self.rho_eq*(torch.sum((psi[:, -1] - psi_fin) ** 2, 1) + torch.sum((psi[:, 0] - psi_init) ** 2, 1) + 
                                    torch.sum((psidot[:, -1] - psidot_fin) ** 2, 1) + torch.sum((psidot[:, 0] - psidot_init) ** 2, 1))
        #cost_v = 0.5*self.rho_eq*torch.sum((v[:, 0] - v_init) ** 2, 1)
        
        cost_smoothness = 0.5*self.weight_smoothness*(torch.sum(xddot**2, 1) + torch.sum(yddot**2, 1)) + 0.5*self.weight_smoothness_psi*torch.sum(psiddot**2, 1)
        return cost_nonhol + cost_pos + cost_psi + cost_smoothness #+ cost_v


# In[14]:


class DeclarativeFunction(torch.autograd.Function):
    """Generic declarative autograd function.
    Defines the forward and backward functions. Saves all inputs and outputs,
    which may be memory-inefficient for the specific problem.
    
    Assumptions:
    * All inputs are PyTorch tensors
    * All inputs have a single batch dimension (b, ...)
    """
    @staticmethod
    def forward(ctx, problem, *inputs):
        output, solve_ctx = torch.no_grad()(problem.solve)(*inputs)
        ctx.save_for_backward(output, *inputs)
        ctx.problem = problem
        ctx.solve_ctx = solve_ctx
        return output.clone()

    @staticmethod
    def backward(ctx, grad_output):
        output, *inputs = ctx.saved_tensors
        problem = ctx.problem
        solve_ctx = ctx.solve_ctx
        output.requires_grad = True
        inputs = tuple(inputs)
        grad_inputs = problem.gradient(*inputs, y=output, v=grad_output,
            ctx=solve_ctx)
        return (None, *grad_inputs)


# In[15]:


class DeclarativeLayer(torch.nn.Module):
    """Generic declarative layer.
    
    Assumptions:
    * All inputs are PyTorch tensors
    * All inputs have a single batch dimension (b, ...)
    Usage:
        problem = <derived class of *DeclarativeNode>
        declarative_layer = DeclarativeLayer(problem)
        y = declarative_layer(x1, x2, ...)
    """
    def __init__(self, problem):
        super(DeclarativeLayer, self).__init__()
        self.problem = problem
        
    def forward(self, *inputs):
        return DeclarativeFunction.apply(self.problem, *inputs)


# In[16]:


class TrajNet(nn.Module):
    def __init__(self, opt_layer, P, Pdot, input_size=32, hidden_size=64, output_size=4, nvar=11, t_obs=8):
        super(TrajNet, self).__init__()
        self.nvar = nvar
        self.t_obs = t_obs
        self.P = torch.tensor(P, dtype=torch.double).to(device)
        self.Pdot = torch.tensor(Pdot, dtype=torch.double).to(device)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.opt_layer = opt_layer
        self.activation = nn.ReLU()
        self.mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.double).to(device)
    
    def forward(self, x, fixed_params, var_inp):
        batch_size, _ = x.size()
        out = self.activation(self.linear1(x))
        variable_params = self.linear2(out)
        variable_params = self.mask * var_inp + (1-self.mask) * variable_params
        
        # Run optimization
        sol = self.opt_layer(fixed_params, variable_params)
         
        # Compute final trajectory
        x_pred = torch.matmul(self.P, sol[:, :self.nvar].transpose(0, 1))
        y_pred = torch.matmul(self.P, sol[:, self.nvar:2*self.nvar].transpose(0, 1))
        x_pred = x_pred.transpose(0, 1)
        y_pred = y_pred.transpose(0, 1)
        out = torch.cat([x_pred, y_pred], dim=1)
        return out


# In[8]:


class ArgoverseDataset(Dataset):
    def __init__(self, data_path, t_obs=16, dt=0.125,centerline_dir=None):
        self.data = np.load(data_path)
        self.data_path = data_path
        self.t_obs = t_obs
        self.dt = dt
        self.centerline_dir = centerline_dir
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        traj = self.data[idx]
        x_traj = traj[:, 0]
        y_traj = traj[:, 1]
        
        vx_traj = traj[:, 2]
        vy_traj = traj[:, 3]
        psi_traj = traj[:, 4]
        psidot_traj = traj[:, 5]

        v_x = [ (x_traj[k + 1] - x_traj[k])/self.dt  for k in range(len(x_traj) - 1)]
        v_y = [ (y_traj[k + 1] - y_traj[k])/dt  for k in range(len(gt_y) - 1)]
        psi = [ np.arctan(v_y[k]/v_x[k]) if v_x[k] > 0 else (np.arctan(v_y[k]/v_x[k]) + np.pi) for k in range(len(v_x))]
        psidot = [ (psi[k + 1] - psi[k])/dt for k in range(len(psi) - 1) ]
        psi = [i.item() for i in psi]
        psidot = [i.item() for i in psidot]        
        psi_traj = np.pi * psi_traj / 180.0
        psidot_traj = np.pi * psidot_traj / 180.0
                
        x_traj = x_traj - x_traj[0]
        y_traj = y_traj - y_traj[0]
                
        x_inp = x_traj[:self.t_obs]
        y_inp = y_traj[:self.t_obs]
        x_fut = x_traj[self.t_obs:]
        y_fut = y_traj[self.t_obs:]
        psi_fut = psi_traj[self.t_obs:]
        psidot_fut = psidot_traj[self.t_obs:]
        
        vx_beg = vx_traj[self.t_obs-1]
        vy_beg = vy_traj[self.t_obs-1]
        
        vx_beg_prev = vx_traj[self.t_obs-2]
        vy_beg_prev = vy_traj[self.t_obs-2]
        
        ax_beg = (vx_beg - vx_beg_prev) / self.dt
        ay_beg = (vy_beg - vy_beg_prev) / self.dt
        
        vx_fin = vx_traj[2*self.t_obs-1]
        vy_fin = vy_traj[2*self.t_obs-1]
        
        vx_fin_prev = vx_traj[2*self.t_obs-2]
        vy_fin_prev = vy_traj[2*self.t_obs-2]
        
        ax_fin = (vx_fin - vx_fin_prev) / self.dt
        ay_fin = (vy_fin - vy_fin_prev) / self.dt
        
        traj_inp = np.dstack((x_inp, y_inp)).flatten()        
        if self.centerline_dir is not None:
            cs = np.load(self.centerline_dir)[idx][self.t_obs:]
            data = np.load(self.data_path)

            
            c_x = cs[:, 0]
            
            c_y = cs[:, 1]
            c_x -= data[idx][0,0]
            c_y -= data[idx][0,1]
            c_x -= c_x[0]
            c_y -= c_y[0]
            c_x += x_inp[-1]
            c_y += y_inp[-1]
        
#             c_y += y_inp[-1] + 2
            c_inp = np.dstack((c_x, c_y)).flatten()
            traj_inp = np.hstack((traj_inp, c_inp))
            
        vx_fut = vx_traj[self.t_obs:]
        vy_fut = vy_traj[self.t_obs:]
        traj_out = np.hstack((x_fut, y_fut)).flatten()
        
        fixed_params = np.array([x_fut[0], y_fut[0], 0, psi_fut[0], psidot_fut[0]])
        var_inp = np.array([x_inp[-1], y_inp[-1], psi_fut[-1], psidot_fut[-1]])
        return torch.tensor(traj_inp), torch.tensor(traj_out), torch.tensor(fixed_params), torch.tensor(var_inp)


# In[22]:


train_dataset = ArgoverseDataset("/datasets/argoverse/val_data.npy", centerline_dir="/datasets/argoverse/val_centerlines.npy", t_obs=30, dt=0.1)
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=False, num_workers=0)

test_dataset = ArgoverseDataset("/datasets/argoverse/val_test_data.npy", centerline_dir="/datasets/argoverse/val_test_centerlines.npy", t_obs=30, dt=0.1)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=0)


# In[17]:


# train_dataset = TrajectoryDataset("/datasets/argoverse/carla_train.npy", centerline_dir="/datasets/argoverse/carla_train_centerlines.npy")
# train_loader = DataLoader(train_dataset, batch_size=20, shuffle=False, num_workers=0)


# In[19]:


# test_dataset = TrajectoryDataset("/datasets/argoverse/carla_test.npy", centerline_dir="/datasets/argoverse/carla_test_centerlines.npy")
# test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=0)


# In[21]:


offsets_train = np.load("/datasets/argoverse/val_offsets.npy")
# offsets_test = np.load("/datasets/argoverse/val_offsets_test.npy")


# In[31]:


import numpy as np
import matplotlib.pyplot as plt

def plot_traj(cnt, traj_inp, traj_out, traj_pred, obs, batch_num=0, num = 30, offsets = [], cities = [], avm = None, center = True, mode = "train", inp_len=40, c_len=70):
    traj_inp = traj_inp.numpy()
    traj_out = traj_out.numpy()
    traj_pred = traj_pred.detach().numpy()
    
    lane_centerlines = []
    ind = batch_num * 20 + cnt

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # Get lane centerlines which lie within the range of trajectories
    ox = offsets[ind][0] + 2
    oy = offsets[ind][1] + 2
    ox = 0
    oy = 0
    if avm is not None:
        city = cities[ind]
        ox = offsets[ind][0] + 2
        oy = offsets[ind][1] + 2
        x_max = np.max(np.concatenate((traj_inp[:inp_len:2], traj_out[:num], traj_pred[:num]), axis=0)) + ox
        x_min = np.min(np.concatenate((traj_inp[:inp_len:2], traj_out[:num], traj_pred[:num]), axis=0)) + ox
        y_max = np.max(np.concatenate((traj_inp[1:inp_len:2], traj_out[num:], traj_pred[num:]), axis=0)) + oy
        y_min = np.min(np.concatenate((traj_inp[1:inp_len:2], traj_out[num:], traj_pred[num:]), axis=0)) + oy
        
        seq_lane_props = avm.city_lane_centerlines_dict[city]
        for lane_id, lane_props in seq_lane_props.items():
            lane_cl = lane_props.centerline

            if (np.min(lane_cl[:, 0]) < x_max and np.min(lane_cl[:, 1]) < y_max and np.max(lane_cl[:, 0]) > x_min and np.max(lane_cl[:, 1]) > y_min):
                lane_centerlines.append(lane_cl)

        for lane_cl in lane_centerlines:
            if True:
                ax.plot(lane_cl[:, 0], lane_cl[:, 1], "--", color="grey", alpha=1, linewidth=1, zorder=0)

    ax.scatter(traj_inp[:inp_len:2] + ox, traj_inp[1:inp_len:2] + oy, color='blue', label='Inp traj')
    ax.scatter(traj_out[:num] + ox, traj_out[num:] + oy, color='orange', label='GT')
    ax.scatter(traj_pred[:num] + ox, traj_pred[num:] + oy, color='green', label='Pred')

    if center:
        ax.plot(traj_inp[inp_len:c_len:2] + ox , traj_inp[inp_len + 1:c_len:2] + oy, color='black',label='primary-centerline')
    
    ax.legend()
    ax.set_xlim([-50,50])
    ax.set_ylim([-50,50])    
    if mode == "train":
        plt.savefig('./results/{}.png'.format(cnt))
    else:
        plt.savefig('./results/{}.png'.format(batch_num * 20 + cnt))
    plt.close()


# In[32]:


for batch_num, data in enumerate(train_loader):
    traj_inp, traj_out, fixed_params, var_inp = data
    torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=False)
    print(traj_inp.size(), traj_out.size(), fixed_params.size, var_inp.size)
    ade = []
    fde = []
    out = model(traj_inp, fixed_params, var_inp)    
#     plt.scatter(traj_inp[0][:32:2], traj_inp[0][1:32:2])
#     plt.scatter(traj_inp[0][32:96:2], traj_inp[0][33:96:2])
    for ii in range(1):
        gt = [[out[ii][j],out[ii][j + num]] for j in range(len(out[ii])//2)]
        pred = [[traj_out[ii][j],traj_out[ii][j + num]] for j in range(len(out[ii])//2)]
        ade.append(get_ade(np.array(pred), np.array(gt)))
        fde.append(get_fde(np.array(pred), np.array(gt)))
        plot_traj(ii, traj_inp[ii], traj_out[ii], out[ii], {"x": [], "y": []}, offsets=offsets_train, cities = [], avm=None, center=True, inp_len=num * 2, c_len = num * 2 + num_elems * 2, num=num)
    
#     print(fixed_params, var_inp)
    break


# In[33]:


problem = OPTNode()
opt_layer = DeclarativeLayer(problem)

model = TrajNet(opt_layer, problem.P, problem.Pdot, input_size=64)
model = model.double()
model = model.to(device)


# In[34]:


out = model(traj_inp, fixed_params, var_inp)
out.shape


# In[35]:


criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)


# In[36]:


num = 16
num_elems = 16


# In[45]:


epoch_train_loss = []
num_epochs = 20

for epoch in range(num_epochs):
    train_loss = []
    mean_ade = []
    mean_fde = []    
    for batch_num, data in enumerate(train_loader):
        traj_inp, traj_out, fixed_params, var_inp = data
        traj_inp = traj_inp.to(device)
        traj_out = traj_out.to(device)
        fixed_params = fixed_params.to(device)
        var_inp = var_inp.to(device)

        ade = []
        fde = []            
        
        out = model(traj_inp, fixed_params, var_inp)
        loss = criterion(out, traj_out)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.item())
        for ii in range(traj_inp.size()[0]):
            gt = [[out[ii][j],out[ii][j + num]] for j in range(len(out[ii])//2)]
            pred = [[traj_out[ii][j],traj_out[ii][j + num]] for j in range(len(out[ii])//2)]
            ade.append(get_ade(np.array(pred), np.array(gt)))
            fde.append(get_fde(np.array(pred), np.array(gt)))                        
            plot_traj(ii, traj_inp[ii], traj_out[ii], out[ii], {"x": [], "y": []}, offsets=offsets_train, cities = [], avm=None, center=True, inp_len=num * 2, c_len = num * 2 + num_elems * 2, num=num, mode="train", batch_num=batch_num)
        if batch_num % 10 == 0:
            print("Epoch: {}, Batch: {}, Loss: {}".format(epoch, batch_num, loss.item()))
            print("ADE: {}".format(np.mean(ade)), "FDE: {}".format(np.mean(fde)))
    
        mean_ade.append(np.mean(ade))
        mean_fde.append(np.mean(fde))
    
    mean_loss = np.mean(train_loss)
    epoch_train_loss.append(mean_loss)
    torch.save(model.state_dict(), "./checkpoints/final.ckpt")
    print("Epoch: {}, Mean Loss: {}".format(epoch, mean_loss))
    print("Mean ADE: {}".format(np.mean(mean_ade)), "Mean FDE: {}".format(np.mean(mean_fde)))
    print("-"*100)


# In[46]:


with torch.no_grad():
    cnt = 0
    test_loss = []
    mean_ade = []
    mean_fde = []     
    for batch_num, data in enumerate(test_loader):
        traj_inp, traj_out, fixed_params, var_inp = data
        traj_inp = traj_inp.to(device)
        traj_out = traj_out.to(device)
        fixed_params = fixed_params.to(device)
        var_inp = var_inp.to(device)
        
        ade = []
        fde = []        
        
        out = model(traj_inp, fixed_params, var_inp)
        loss = criterion(out, traj_out)
        
        test_loss.append(loss.item())
        print("Batch: {}, Loss: {}".format(batch_num, loss.item()))
        
        for ii in range(traj_inp.size()[0]):
            gt = [[out[ii][j],out[ii][j + num]] for j in range(len(out[ii])//2)]
            pred = [[traj_out[ii][j],traj_out[ii][j + num]] for j in range(len(out[ii])//2)]
            ade.append(get_ade(np.array(pred), np.array(gt)))
            fde.append(get_fde(np.array(pred), np.array(gt)))                        
            plot_traj(ii, traj_inp[ii], traj_out[ii], out[ii], {"x": [], "y": []}, offsets=offsets_test, cities = [], avm=None, center=True, inp_len=num * 2, c_len = num * 2 + num * 2, num=num, mode="test", batch_num=batch_num)

        mean_ade.append(np.mean(ade))
        mean_fde.append(np.mean(fde))  

mean_loss = np.mean(test_loss)
print("Epoch Mean Test Loss: {}".format(mean_loss))
print("Mean ADE: {}".format(np.mean(mean_ade)), "Mean FDE: {}".format(np.mean(mean_fde)))


# In[152]:


import numpy as np
import matplotlib.pyplot as plt
import json
import time

import bernstein_coeff_order10_arbitinterval
import jax.numpy as jnp

from scipy.ndimage.filters import uniform_filter1d


f = open("inp.json", "r")

data = json.load(f)

# to get each part of the json simply use data["x_init"] for example. 

x_init_data = np.asarray(data["x_init"])
y_init_data = np.asarray(data["y_init"])
vx_init_data = np.asarray(data["vx_init"])
vy_init_data = np.asarray(data["vy_init"])


psi_init_data = np.asarray(data["psi_init"])

psi_init_data = np.arctan2( vy_init_data, vx_init_data  )

# psi_init_data = np.arctan2( np.sin(psi_init_data), np.cos(psi_init_data)   )
psidot_init_data = np.asarray(data["psidot_init"])
x_fin_data = np.asarray(data["x_fin"])
y_fin_data = np.asarray(data["y_fin"])
psi_fin_data = np.asarray(data["psi_fin"])

# psi_fin_data = 

psi_fin_data = np.arctan2( np.sin(psi_fin_data), np.cos(psi_fin_data)   )
psidot_fin_data = np.asarray(data["psidot_fin"])

pos = np.load('data/pos_data.npy')
# pos_x = 

print(np.shape(pos))

# psi_fin_data = np.arctan2( xy_data[1, 1]-xy_data[1, 0], xy_data[0,1]-xy_data[0, 0] )

# x_init_data[19, 0] = -12.22
# y_init_data[19, 0] = -0.984



idx = 2

xy_data = pos[idx]


# print(np.arctan2( xy_data[1, 1]-xy_data[1, 0], xy_data[0,1]-xy_data[0, 0] )   )

# print(np.arctan2( vy_init_data[idx, 0], vx_init_data[idx, 0] ) )

# print(psi_init_data[idx, 0])
# kk


# print('x_init =', x_init_data[idx, 0])
# print('y_init = ',y_init_data[idx, 0])
# print('psi_init=',psi_init_data[idx, 0])
# print('vx_init =',vx_init_data[idx, 0])
# print('vy_init = ',vy_init_data[idx, 0])
# print('x_fin = ',x_fin_data[idx, 0])
# print('y_fin =',y_fin_data[idx, 0])
# print('v_init =', np.sqrt( vx_init_data[idx, 0]**2+vy_init_data[idx, 0]**2  ) )

# print(psi_init_data[idx, 0])

# plt.plot(xy_data[0], xy_data[1])
# plt.axis('equal')
# plt.show()




class Optimizer_Nonhol():

	def __init__(self):


		self.rho_eq = 1.0
		# self.rho_goal = 1.0
		self.rho_nonhol = 1.0
		self.rho_psi = self.rho_nonhol
		self.maxiter = 1000
		self.weight_smoothness = 1.0
		self.weight_smoothness_psi = 1.0

		self.t_fin = 3.0
		self.num = 30
		self.t = self.t_fin/self.num

		self.num_batch = 20

		tot_time = np.linspace(0.0, self.t_fin, self.num)
		tot_time_copy = tot_time.reshape(self.num, 1)
		self.P, self.Pdot, self.Pddot = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)
		self.nvar = np.shape(self.P)[1]

		self.lamda_x = np.zeros((self.num_batch, self.nvar))
		self.lamda_y = np.zeros((self.num_batch, self.nvar))

		# self.P_psi = np.identity(self.num)
		# self.Pddot_psi = np.diff( np.diff(np.identity(self.num), axis = 0), axis = 0)
		# self.Pdot_psi = np.diff(self.P_psi, axis = 0)

		# self.P_psi = self.P
		# self.Pddot_psi = self.Pddot
		# self.Pdot_psi = self.Pdot


		# self.nvar_psi = np.shape(self.P_psi)[1]

		self.lamda_psi = np.zeros((self.num_batch, self.nvar))

		self.cost_smoothness = self.weight_smoothness*np.dot(self.Pddot.T, self.Pddot)

		self.cost_smoothness_psi = self.weight_smoothness_psi*np.dot(self.Pddot.T, self.Pddot)
		self.lincost_smoothness_psi = np.zeros(self.nvar)

		# self.rho_mid = 0.01
		# self.mid_idx = np.array([ int(self.num/4), int(self.num/2), int(3*self.num/4)  ])

		self.A_eq = np.vstack(( self.P[0], self.P[-1]    ))

		# self.A_eq = self.P[0].reshape(1, self.nvar)

		self.A_eq_hol = np.vstack(( self.P[0], self.Pdot[0], self.P[-1]   ))


		# self.A_eq_psi = np.vstack(( self.P[0], self.Pdot[0], self.P[-1], self.Pdot[-1]  ))

		self.A_eq_psi = np.vstack(( self.P[0], self.Pdot[0], self.P[-1]))
		# self.A_eq_psi = self.P[0].reshape(1, self.nvar)

		self.A_pos_goal = self.P[0].reshape(1, self.nvar)

		self.A_psi_goal = self.P[0].reshape(1, self.nvar)


		self.A_nonhol = self.Pdot
		self.A_psi = self.P

		######################################################################### Converting to Jax for objective and gradient computation

		self.P_jax = jnp.asarray(self.P)
		self.Pdot_jax = jnp.asarray(self.Pdot)
		self.Pddot_jax = jnp.asarray(self.Pddot)



	def compute_x(self, b_eq_x, b_eq_y):

		b_nonhol_x = self.v*np.cos(self.psi)
		b_nonhol_y = self.v*np.sin(self.psi)

		# cost = self.cost_smoothness+self.rho_nonhol*np.dot(self.A_nonhol.T, self.A_nonhol)+self.rho_eq*np.dot(self.A_eq.T, self.A_eq)+self.rho_goal*np.dot(self.A_pos_goal.T, self.A_pos_goal)
		# lincost_x = -self.lamda_x-self.rho_nonhol*np.dot(self.A_nonhol.T, b_nonhol_x.T ).T-self.rho_eq*np.dot(self.A_eq.T, b_eq_x.T).T-self.rho_goal*np.dot(self.A_pos_goal.T, b_goal_x.T).T
		# lincost_y = -self.lamda_y-self.rho_nonhol*np.dot(self.A_nonhol.T, b_nonhol_y.T ).T-self.rho_eq*np.dot(self.A_eq.T, b_eq_y.T).T-self.rho_goal*np.dot(self.A_pos_goal.T, b_goal_y.T).T

		cost = self.cost_smoothness+self.rho_nonhol*np.dot(self.A_nonhol.T, self.A_nonhol)+self.rho_eq*np.dot(self.A_eq.T, self.A_eq)
		lincost_x = -self.lamda_x-self.rho_nonhol*np.dot(self.A_nonhol.T, b_nonhol_x.T ).T-self.rho_eq*np.dot(self.A_eq.T, b_eq_x.T).T
		lincost_y = -self.lamda_y-self.rho_nonhol*np.dot(self.A_nonhol.T, b_nonhol_y.T ).T-self.rho_eq*np.dot(self.A_eq.T, b_eq_y.T).T


		cost_inv = np.linalg.inv(cost)

		sol_x = np.dot(-cost_inv, lincost_x.T).T
		sol_y = np.dot(-cost_inv, lincost_y.T).T

		self.x = np.dot(self.P, sol_x.T).T
		self.xdot = np.dot(self.Pdot, sol_x.T).T

		self.y = np.dot(self.P, sol_y.T).T
		self.ydot = np.dot(self.Pdot, sol_y.T).T

		return sol_x, sol_y


	def compute_psi( self, psi_temp, b_eq_psi, b_goal_psi  ):

		cost = self.cost_smoothness_psi+self.rho_psi*np.dot(self.A_psi.T, self.A_psi)+self.rho_eq*np.dot(self.A_eq_psi.T, self.A_eq_psi)
		lincost_psi = -self.lamda_psi-self.rho_psi*np.dot(self.A_psi.T, psi_temp.T).T-self.rho_eq*np.dot(self.A_eq_psi.T, b_eq_psi.T).T

		cost_inv = np.linalg.inv(cost)

		sol_psi = np.dot(-cost_inv, lincost_psi.T).T

		self.psi = np.dot(self.P, sol_psi.T).T



		res_psi = np.dot(self.A_psi, sol_psi.T).T-psi_temp
		res_eq_psi = np.dot(self.A_eq_psi, sol_psi.T).T-b_eq_psi

		self.lamda_psi = self.lamda_psi-self.rho_psi*np.dot(self.A_psi.T, res_psi.T).T-self.rho_eq*np.dot(self.A_eq_psi.T, res_eq_psi.T).T

		# self.lamda_psi = self.lamda_psi-self.rho_eq*np.dot(self.A_eq_psi.T, res_eq_psi.T).T
		# self.lamda_psi = lamda_psi_old+0.3*(self.lamda_psi-lamda_psi_old)

		return sol_psi, np.linalg.norm(res_psi), np.linalg.norm(res_eq_psi)

	def compute_holonomic_traj(self, b_eq_x_hol, b_eq_y_hol):

		cost = self.cost_smoothness
		cost_mat = np.vstack((  np.hstack(( cost, self.A_eq_hol.T )), np.hstack(( self.A_eq_hol, np.zeros(( np.shape(self.A_eq_hol)[0], np.shape(self.A_eq_hol)[0] )) )) ))
		cost_mat_inv = np.linalg.inv(cost_mat)
		sol_x = np.dot(cost_mat_inv, np.hstack(( np.zeros(( self.num_batch, self.nvar )), b_eq_x_hol )).T).T
		sol_y = np.dot(cost_mat_inv, np.hstack(( np.zeros(( self.num_batch, self.nvar )), b_eq_y_hol )).T).T


		xdot_guess = np.dot(self.Pdot, sol_x[:,0:self.nvar].T).T
		ydot_guess = np.dot(self.Pdot, sol_y[:,0:self.nvar].T).T

		return xdot_guess, ydot_guess




	def solve(self, x_init, x_fin, y_init, y_fin, v_init, psi_init, psidot_init, psi_fin, psidot_fin ):

		vx_init = v_init*np.cos(psi_init)
		vy_init = v_init*np.sin(psi_init)


		b_eq_x = np.hstack(( x_init, x_fin  ))
		b_eq_y = np.hstack(( y_init, y_fin  ))

		# b_eq_x = x_init
		# b_eq_y = y_init

		# b_goal_x = x_fin 
		# b_goal_y = y_fin 




		# b_eq_psi = np.hstack(( psi_init, psidot_init, psi_fin, psidot_fin  ))

		b_eq_psi = np.hstack(( psi_init, psidot_init, psi_fin))

		# b_eq_psi = psi_init

		b_goal_psi = psi_fin

		b_eq_x_hol = np.hstack(( x_init, vx_init,  x_fin  ))
		b_eq_y_hol = np.hstack(( y_init, vy_init,  y_fin  ))


		xdot_guess, ydot_guess = self.compute_holonomic_traj(b_eq_x_hol, b_eq_y_hol)


		res_psi = np.ones(self.maxiter)
		res_eq_psi = np.ones(self.maxiter)
		res_nonhol = np.ones(self.maxiter)
		res_eq = np.ones(self.maxiter)

		# self.v = np.ones((self.num_batch, self.num))*v_init
		# self.psi = np.ones((self.num_batch, self.num))*psi_init

		# self.xdot = self.v*np.cos(self.psi)
		# self.ydot = self.v*np.sin(self.psi)


		self.v = np.sqrt(xdot_guess**2+ydot_guess**2)
		self.psi = np.unwrap(np.arctan2(ydot_guess, xdot_guess))


		self.xdot = xdot_guess
		self.ydot = ydot_guess

		# plt.plot(self.v.T)
		# plt.show()




		for i in range(0, self.maxiter):






			# self.psi[:, 0] = psi_init[:, 0]
			# self.psi[:, -1] = psi_fin[:, -1]

			# print('check = ',psi_fin[:, -1])
			# kk



			# self.psi = psi_temp	
			c_x, c_y = self.compute_x(b_eq_x, b_eq_y)

			psi_temp = np.unwrap(np.arctan2(self.ydot, self.xdot))

			c_psi, res_psi[i], res_eq_psi[i] = self.compute_psi(psi_temp, b_eq_psi, b_goal_psi  )


			# plt.plot(psi_temp.T)
			# plt.plot(self.psi.T, '-r')
			# plt.show()








			# self.psi = psi_temp

			# self.psi = np.linspace(psi_init, psi_fin, self.num).squeeze().T
			# print(np.shape(self.psi))
			# kk



			# plt.plot(self.psi)
			# plt.show()

			# self.v = self.xdot*np.cos(self.psi)+self.ydot*np.sin(self.psi)
			self.v = np.sqrt(self.xdot**2+self.ydot**2)
			# self.v = self.xdot*np.cos(self.psi)+self.ydot*np.sin(self.psi)

			# self.v[:, 0] = v_init[:, 0]

			res_eq_x = np.dot(self.A_eq, c_x.T).T-b_eq_x
			res_nonhol_x = self.xdot-self.v*np.cos(self.psi) 

			res_eq_y = np.dot(self.A_eq, c_y.T).T-b_eq_y
			res_nonhol_y = self.ydot-self.v*np.sin(self.psi)

			self.lamda_x = self.lamda_x-self.rho_eq*np.dot(self.A_eq.T, res_eq_x.T).T-self.rho_nonhol*np.dot(self.A_nonhol.T, res_nonhol_x.T).T
			self.lamda_y = self.lamda_y-self.rho_eq*np.dot(self.A_eq.T, res_eq_y.T).T-self.rho_nonhol*np.dot(self.A_nonhol.T, res_nonhol_y.T).T


			# self.lamda_x = self.lamda_x-self.rho_eq*np.dot(self.A_eq.T, res_eq_x.T).T
			# self.lamda_y = self.lamda_y-self.rho_eq*np.dot(self.A_eq.T, res_eq_y.T).T


			res_eq[i] = np.linalg.norm(np.hstack(( res_eq_x, res_eq_y   ) ))
			res_nonhol[i] = np.linalg.norm(np.hstack(( res_nonhol_x, res_nonhol_y   ) ))



		plt.figure(1)
		plt.plot(res_eq)

		plt.figure(2)
		plt.plot(res_nonhol)

		plt.figure(3)
		plt.plot(res_eq_psi)

		plt.figure(4)
		plt.plot(self.x.T, self.y.T)
		plt.plot(xy_data[0], xy_data[1])
		plt.axis('equal')

		plt.figure(5)
		plt.plot(self.v.T)


		# plt.show()

		return c_x, c_y, c_psi, self.v, self.x, self.y







################################################################################################### Trajectory data with rotation



prob = Optimizer_Nonhol()


rot_angle = -psi_init_data


# rot_angle = 0.0

psi_init_mod = psi_init_data+rot_angle
psi_fin_mod = np.arctan2( np.sin(psi_fin_data+rot_angle), np.cos(psi_fin_data+rot_angle) )
# print(psi_fin_mod, rot_angle)
# print(psi_init_data[idx, 0], psi_fin_data[idx, 0])
# kk

# print(psi_fin_mod, psi_fin_data[idx, 0], psi_init_data[idx, 0])


x_init_temp = x_init_data
x_fin_temp = x_fin_data

y_init_temp = y_init_data
y_fin_temp = y_fin_data


######################## changing initial and final
x_init_mod = x_init_temp*np.cos(rot_angle)-y_init_temp*np.sin(rot_angle)
y_init_mod = x_init_temp*np.sin(rot_angle)+y_init_temp*np.cos(rot_angle)

x_fin_mod = x_fin_temp*np.cos(rot_angle)-y_fin_temp*np.sin(rot_angle)
y_fin_mod = x_fin_temp*np.sin(rot_angle)+y_fin_temp*np.cos(rot_angle)

##############3


x_init = x_init_mod*np.ones((prob.num_batch,1))
x_fin = x_fin_mod*np.ones((prob.num_batch,1))

y_init = y_init_mod*np.ones((prob.num_batch,1))
y_fin = y_fin_mod*np.ones((prob.num_batch,1))


psi_init = psi_init_mod*np.ones((prob.num_batch,1))
psi_fin = psi_fin_mod*np.ones((prob.num_batch,1))

psidot_init = 0.0*np.ones((prob.num_batch,1))
psidot_fin = 0.0*np.ones((prob.num_batch,1))


v_init_temp = np.sqrt(vx_init_data**2+vy_init_data**2)
v_init =   v_init_temp*np.ones((prob.num_batch,1))





c_x, c_y, c_psi, v, x, y = prob.solve(x_init, x_fin, y_init, y_fin, v_init, psi_init, psidot_init, psi_fin, psidot_fin )



x_rot = x*np.cos(rot_angle)+y*np.sin(rot_angle)
y_rot = -x*np.sin(rot_angle)+y*np.cos(rot_angle)

print(psi_fin_data[idx, 0], prob.psi[:,-1]-rot_angle)
print('.................')
print(psi_init_data[idx, 0], prob.psi[:,0]-rot_angle)


# print(x[:, -1], y[:, -1], x_fin_mod, y_fin_mod)
# print(x_rot[:, -1], y_rot[:, -1], x_fin_data[idx, 0], y_fin_data[idx, 0])
# print(x_rot[:, 0], y_rot[:, 0], x_init_data[idx, 0], y_init_data[idx, 0])




plt.figure(7)
fig, axs = plt.subplots(2)

for i in range(0, prob.num_batch):
	plt.plot(pos[i, 0, :], pos[i,1, : ], '--k', linewidth = 3.0)


# plt.plot(xy_data[0, -1], xy_data[1, -1], 'og', markersize = 9.0)
# plt.plot(xy_data[0, 0], xy_data[1, 0], 'om', markersize = 9.0)
# plt.plot(xy_data[0, 0], xy_data[1, 0], 'om', markersize = 9.0)



plt.axis('equal')
plt.show()








