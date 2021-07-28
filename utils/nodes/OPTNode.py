import os
import sys
sys.path.append("../../ddn/")
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
from ddn.pytorch.node import AbstractDeclarativeNode

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.visualization.visualize_sequences import viz_sequence

avm = ArgoverseMap()

class OPTNode(AbstractDeclarativeNode):
    def __init__(self, P, Pddot, A_eq, A_obs, Q_smoothness, x_obs, y_obs, num=12, num_obs=4, nvar=11, a_obs=1.0, b_obs=1.0, rho_obs=0.3, rho_eq=10.0, weight_smoothness=10, maxiter=300, eps=1e-7, num_tot=48, device="cpu"):
        super().__init__()
        self.P = torch.tensor(P, dtype=torch.double).to(device)
        self.Pddot = torch.tensor(Pddot, dtype=torch.double).to(device)
        self.A_eq = torch.tensor(A_eq, dtype=torch.double).to(device)
        self.A_obs = torch.tensor(A_obs, dtype=torch.double).to(device)
        self.Q_smoothness = torch.tensor(Q_smoothness, dtype=torch.double).to(device)
        self.x_obs = torch.tensor(x_obs, dtype=torch.double).to(device)
        self.y_obs = torch.tensor(y_obs, dtype=torch.double).to(device)
        
        self.num = num
        self.num_obs = num_obs
        self.eps = eps
        self.nvar = nvar        
        self.a_obs = a_obs
        self.b_obs = b_obs        
        self.rho_eq = rho_eq
        self.num_obs = num_obs
        self.maxiter = maxiter
        self.num_tot = num_tot
        self.rho_obs = rho_obs
        self.device = device
        self.weight_smoothness = weight_smoothness
        
    def objective(self, b, lamda_x, lamda_y, y):  
        batch_size, _ = b.size()
        b = b.transpose(0, 1)
        y = y.transpose(0, 1)
        lamda_x = lamda_x.transpose(0, 1)
        lamda_y = lamda_y.transpose(0, 1)
        bx_eq_tensor, by_eq_tensor = torch.split(b, 6, dim=0)
        ones_tensor = torch.ones(self.num_tot, batch_size, dtype=torch.double).to(self.device)

        c_x = y[0:self.nvar]
        c_y = y[self.nvar:2 * self.nvar]
        alpha_obs = y[2 * self.nvar: 2 * self.nvar + self.num_tot]
        d_obs = y[2 * self.nvar + self.num_tot:]

        cost_smoothness_x = 0.5 * self.weight_smoothness * torch.diag(torch.matmul(c_x.T, torch.matmul(self.Q_smoothness, c_x)))
        cost_smoothness_y = 0.5 * self.weight_smoothness * torch.diag(torch.matmul(c_y.T, torch.matmul(self.Q_smoothness, c_y)))

        temp_x_obs = d_obs * torch.cos(alpha_obs) * self.a_obs
        b_obs_x = self.x_obs.view(-1, 1) + temp_x_obs

        temp_y_obs = d_obs * torch.sin(alpha_obs) * self.b_obs
        b_obs_y = self.y_obs.view(-1, 1) + temp_y_obs

        cost_obs_x = 0.5 * self.rho_obs * (torch.sum((torch.matmul(self.A_obs, c_x) - b_obs_x) ** 2, axis=0))
        cost_obs_y = 0.5 * self.rho_obs * (torch.sum((torch.matmul(self.A_obs, c_y) - b_obs_y) ** 2, axis=0))
        cost_slack = self.rho_obs * torch.sum(torch.max(1 - d_obs, ones_tensor), axis=0)

        cost_eq_x = 0.5 * self.rho_eq * torch.sum((torch.matmul(self.A_eq, c_x) - bx_eq_tensor) ** 2, axis=0)
        cost_eq_y = 0.5 * self.rho_eq * torch.sum((torch.matmul(self.A_eq, c_y) - by_eq_tensor) ** 2, axis=0)

        cost_x = cost_smoothness_x + cost_obs_x + cost_eq_x - torch.diag(torch.matmul(lamda_x.transpose(0, 1), c_x))
        cost_y = cost_smoothness_y + cost_obs_y + cost_eq_y - torch.diag(torch.matmul(lamda_y.transpose(0, 1), c_y))
        cost = cost_x + cost_y + self.eps * torch.sum(c_x ** 2, axis=0) + self.eps * torch.sum(c_y ** 2, axis=0) + self.eps * torch.sum(d_obs ** 2, axis=0) + self.eps * torch.sum(alpha_obs ** 2, axis=0) + cost_slack
        return cost
    
    def optimize(self, b, lamda_x, lamda_y):
        bx_eq_tensor, by_eq_tensor = torch.split(b, 6, dim=0)
        
        d_obs = torch.ones(self.num_obs, self.num, dtype=torch.double).to(self.device)
        alpha_obs = torch.zeros(self.num_obs, self.num, dtype=torch.double).to(self.device)
        ones_tensor = torch.ones((self.num_obs, self.num), dtype=torch.double).to(self.device)
        cost_smoothness = self.weight_smoothness * torch.matmul(self.Pddot.T, self.Pddot)
        cost = cost_smoothness + self.rho_obs * torch.matmul(self.A_obs.T, self.A_obs) + self.rho_eq * torch.matmul(self.A_eq.T, self.A_eq)

        for i in range(self.maxiter):
            temp_x_obs = d_obs * torch.cos(alpha_obs) * self.a_obs
            temp_y_obs = d_obs * torch.sin(alpha_obs) * self.b_obs

            b_obs_x = self.x_obs.view(self.num * self.num_obs) + temp_x_obs.view(self.num * self.num_obs)
            b_obs_y = self.y_obs.view(self.num * self.num_obs) + temp_y_obs.view(self.num * self.num_obs)

            lincost_x = -lamda_x - self.rho_obs * torch.matmul(self.A_obs.T, b_obs_x) - self.rho_eq * torch.matmul(self.A_eq.T, bx_eq_tensor)
            lincost_y = -lamda_y - self.rho_obs * torch.matmul(self.A_obs.T, b_obs_y) - self.rho_eq * torch.matmul(self.A_eq.T, by_eq_tensor)

            lincost_x = lincost_x.view(-1, 1)
            lincost_y = lincost_y.view(-1, 1)

            sol_x, _ = torch.solve(lincost_x, -cost)
            sol_y, _ = torch.solve(lincost_y, -cost)

            sol_x = sol_x.view(-1)
            sol_y = sol_y.view(-1)

            x = torch.matmul(self.P, sol_x)
            y = torch.matmul(self.P, sol_y)

            wc_alpha = x - self.x_obs
            ws_alpha = y - self.y_obs
            alpha_obs = torch.atan2(ws_alpha * self.a_obs, wc_alpha * self.b_obs)

            c1_d = self.rho_obs * (self.a_obs ** 2 * torch.cos(alpha_obs) ** 2 + self.b_obs ** 2 * torch.sin(alpha_obs) ** 2)
            c2_d = self.rho_obs * (self.a_obs * wc_alpha * torch.cos(alpha_obs) + self.b_obs * ws_alpha * torch.sin(alpha_obs))
            d_temp = c2_d / c1_d
            d_obs = torch.max(d_temp, ones_tensor)

            res_x_obs_vec = wc_alpha - self.a_obs * d_obs * torch.cos(alpha_obs)
            res_y_obs_vec = ws_alpha - self.b_obs * d_obs * torch.sin(alpha_obs)

            res_eq_x_vec = torch.matmul(self.A_eq, sol_x) - bx_eq_tensor
            res_eq_y_vec = torch.matmul(self.A_eq, sol_y) - by_eq_tensor

            lamda_x -= self.rho_obs * torch.matmul(self.A_obs.T, res_x_obs_vec.view(-1)) + self.rho_eq * torch.matmul(self.A_eq.T, res_eq_x_vec)
            lamda_y -= self.rho_obs * torch.matmul(self.A_obs.T, res_y_obs_vec.view(-1)) + self.rho_eq * torch.matmul(self.A_eq.T, res_eq_y_vec)

        sol = torch.cat([sol_x, sol_y, alpha_obs.view(-1), d_obs.view(-1)])
#         print(sol_x.shape)
        return sol    
    
    def solve(self, b, lamda_x, lamda_y):
        batch_size, _ = b.size()
        b = b.transpose(0, 1)
        lamda_x = lamda_x.transpose(0, 1)
        lamda_y = lamda_y.transpose(0, 1)
        y = torch.zeros(batch_size, 2 * self.nvar + 2 * self.num_tot, dtype=torch.double).to(self.device)
        for i in range(batch_size):
            b_cur = b[:, i]
            lamda_x_cur = lamda_x[:, i]
            lamda_y_cur = lamda_y[:, i]
            sol = self.optimize(b_cur, lamda_x_cur, lamda_y_cur)
#             print(sol.shape)
            y[i, :] = sol
        return y, None
 
 
# #### PyTorch Declarative Function    
# In[20]:

class OptFunction(torch.autograd.Function):
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




# #### PyTorch Declarative Layer

# In[22]:

class OptLayer(torch.nn.Module):
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
        super(OptLayer, self).__init__()
        self.problem = problem
        
    def forward(self, *inputs):
        return OptFunction.apply(self.problem, *inputs)



# #### TrajNet

# In[23]:


class TrajNet(nn.Module):
    def __init__(self, opt_layer, P, input_size=16, hidden_size=128, output_size=12, nvar=11, t_obs=8, device="cpu"):
        super(TrajNet, self).__init__()
        self.P = torch.tensor(P, dtype=torch.double).to(device)
        self.nvar = nvar
        self.t_obs = t_obs
#         hidden_size = 256
        self.linear1 = nn.Linear(input_size, hidden_size)
#         hidden_size = 64
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.opt_layer = opt_layer
        self.activation = nn.ReLU()
        self.mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]], dtype=torch.double)
#        b_inp = np.array([x_fut[0], vx_beg, 0, x_fut[-1], 0, 0, y_fut[0], vy_beg, 0, y_fut[-1], 0, 0])    
        self.device = "cpu"
        
    def forward(self, x, b):
        batch_size, _ = x.size()
        out = self.activation(self.linear1(x))
        b_pred = self.linear2(out)
        b_gen = self.mask * b + (1 - self.mask) * b_pred
        
        # Run optimization
        lamda_x = torch.zeros(batch_size, self.nvar, dtype=torch.double).to(self.device)
        lamda_y = torch.zeros(batch_size, self.nvar, dtype=torch.double).to(self.device)
        sol = self.opt_layer(b_gen, lamda_x, lamda_y)
        # Compute final trajectory
        x_pred = torch.matmul(self.P, sol[:, :self.nvar].transpose(0, 1))
        y_pred = torch.matmul(self.P, sol[:, self.nvar:2*self.nvar].transpose(0, 1))
        x_pred = x_pred.transpose(0, 1)
        y_pred = y_pred.transpose(0, 1)
        out = torch.cat([x_pred, y_pred], dim=1)
        #return out, sol, b_gen, lamda_x, lamda_y
        return out
    