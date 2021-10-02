#!/usr/bin/env python
# coding: utf-8

# In[84]:


import os
import sys
sys.path.append("ddn/")
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
from bernstein import bernstein_coeff_order10_new
from ddn.pytorch.node import AbstractDeclarativeNode


# #### CUDA Initializations

# In[85]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print('Using {} device'.format(device))


# In[86]:


# rho_obs = 0.3
rho_obs = 1.2
rho_eq = 10.0
weight_smoothness = 10


# In[87]:


num = 10
t_fin = 2
a_obs = 1.0
b_obs = 1.0

tot_time = np.linspace(0.0, t_fin, num)
tot_time_copy = tot_time.reshape(num, 1)
P, Pdot, Pddot = bernstein_coeff_order10_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)
nvar = np.shape(P)[1]


# In[88]:


# x_obs_temp = np.hstack((-2.0, -0.79, 3.0, 4.0))
# y_obs_temp = np.hstack((-2.0, 1.0, -0.80, 2.0))

x_obs_temp = np.hstack((-10.0, 100.79, 30.0, 10.0))
y_obs_temp = np.hstack((-10.0, 100.0, -30.80, 10.0))

num_obs = np.shape(x_obs_temp)[0]

x_obs = np.ones((num_obs, num)) * x_obs_temp[:, np.newaxis]
y_obs = np.ones((num_obs, num)) * y_obs_temp[:, np.newaxis]


# In[89]:


obs_inp_list = []
for i in range(num_obs):
    obs_inp_list.extend([x_obs_temp[i], y_obs_temp[i], a_obs])

obs_inp = np.array(obs_inp_list)


# In[90]:


A_obs = np.tile(P, (num_obs, 1))
A_eq = np.vstack((P[0], Pdot[0], Pddot[0], P[-1], Pdot[-1], Pddot[-1]))
Q_smoothness = np.dot(Pddot.T, Pddot)


# In[91]:


class OPTNode(AbstractDeclarativeNode):
    def __init__(self, P, Pddot, A_eq, A_obs, Q_smoothness, x_obs, y_obs, num=12, num_obs=4, nvar=11, a_obs=1.0, b_obs=1.0, rho_obs=0.3, rho_eq=10.0, weight_smoothness=10, maxiter=300, eps=1e-7, num_tot=48):
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
        self.weight_smoothness = weight_smoothness
        
    def objective(self, b, lamda_x, lamda_y, y):  
        batch_size, _ = b.size()
        b = b.transpose(0, 1)
        y = y.transpose(0, 1)
        lamda_x = lamda_x.transpose(0, 1)
        lamda_y = lamda_y.transpose(0, 1)
        bx_eq_tensor, by_eq_tensor = torch.split(b, 6, dim=0)
        ones_tensor = torch.ones(self.num_tot, batch_size, dtype=torch.double).to(device)

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
        cost_eq_y = 0.5 * self.rho_eq * torch.sum((torch.matmul(self. A_eq, c_y) - by_eq_tensor) ** 2, axis=0)

        cost_x = cost_smoothness_x + cost_obs_x + cost_eq_x - torch.diag(torch.matmul(lamda_x.transpose(0, 1), c_x))
        cost_y = cost_smoothness_y + cost_obs_y + cost_eq_y - torch.diag(torch.matmul(lamda_y.transpose(0, 1), c_y))
        cost = cost_x + cost_y + self.eps * torch.sum(c_x ** 2, axis=0) + self.eps * torch.sum(c_y ** 2, axis=0) + self.eps * torch.sum(d_obs ** 2, axis=0) + self.eps * torch.sum(alpha_obs ** 2, axis=0) + cost_slack
        return cost
    
    def optimize(self, b, lamda_x, lamda_y):
        bx_eq_tensor, by_eq_tensor = torch.split(b, 6, dim=0)
        
        d_obs = torch.ones(self.num_obs, self.num, dtype=torch.double).to(device)
        alpha_obs = torch.zeros(self.num_obs, self.num, dtype=torch.double).to(device)
        ones_tensor = torch.ones((self.num_obs, self.num), dtype=torch.double).to(device)
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
        return sol

    def solve(self, b, lamda_x, lamda_y):
        batch_size, _ = b.size()
        b = b.transpose(0, 1)
        lamda_x = lamda_x.transpose(0, 1)
        lamda_y = lamda_y.transpose(0, 1)
        y = torch.zeros(batch_size, 2 * self.nvar + 2 * self.num_tot, dtype=torch.double).to(device)
        for i in range(batch_size):
            b_cur = b[:, i]
            lamda_x_cur = lamda_x[:, i]
            lamda_y_cur = lamda_y[:, i]
            sol = self.optimize(b_cur, lamda_x_cur, lamda_y_cur)
            y[i, :] = sol
        return y, None


# #### PyTorch Declarative Function

# In[92]:


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

# In[93]:


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

# In[94]:


class TrajNet(nn.Module):
    def __init__(self, opt_layer, P, input_size=16, hidden_size=64, output_size=12, nvar=11, t_obs=8):
        super(TrajNet, self).__init__()
        self.P = torch.tensor(P, dtype=torch.double).to(device)
        self.nvar = nvar
        self.t_obs = t_obs
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.opt_layer = opt_layer
        self.activation = nn.ReLU()
        self.mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]], dtype=torch.double)
    
    def forward(self, x, b):
        batch_size, _ = x.size()
        out = self.activation(self.linear1(x))
        b_pred = self.linear2(out)
        b_gen = self.mask * b + (1 - self.mask) * b_pred
        
        # Run optimization
        lamda_x = torch.zeros(batch_size, self.nvar, dtype=torch.double).to(device)
        lamda_y = torch.zeros(batch_size, self.nvar, dtype=torch.double).to(device)
        sol = self.opt_layer(b_gen, lamda_x, lamda_y)

        # Compute final trajectory
        x_pred = torch.matmul(self.P, sol[:, :self.nvar].transpose(0, 1))
        y_pred = torch.matmul(self.P, sol[:, self.nvar:2*self.nvar].transpose(0, 1))
        
        x_pred = x_pred.transpose(0, 1)
        y_pred = y_pred.transpose(0, 1)
        out = torch.cat([x_pred, y_pred], dim=1)
        #return out, sol, b_gen, lamda_x, lamda_y
        return out


# In[95]:


opt_node = OPTNode(
    P, 
    Pddot, 
    A_eq, 
    A_obs, 
    Q_smoothness, 
    x_obs, 
    y_obs,
    num=num, 
    num_obs=num_obs, 
    nvar=nvar, 
    a_obs=a_obs, 
    b_obs=b_obs, 
    rho_obs=rho_obs, 
    rho_eq=rho_eq,
    weight_smoothness=weight_smoothness, 
    maxiter=300, 
    eps=1e-7, 
    num_tot=num*num_obs
)
opt_layer = OptLayer(opt_node)


# In[96]:


model = TrajNet(opt_layer, P, input_size=2 * num + 3 * num_obs + 4)
model = model.double()
model = model.to(device)


# #### Model Testing

# In[97]:


x_inp = torch.randn(10, 36, dtype=torch.double)
y_out = torch.randn(10, 50, dtype=torch.double)
b_inp = torch.randn(10, 12, dtype=torch.double)


# In[98]:


out = model(x_inp, b_inp)
out.size()


# #### Trajectory DataLoader

# In[99]:


class TrajectoryDataset(Dataset):
    def __init__(self, root_dir, t_obs=10, dt=0.4):
        self.root_dir = root_dir
        self.t_obs = t_obs
        self.dt = dt
    
    def __len__(self):
        return len(np.load(self.root_dir))
    
    def get_vel(self, pos):
        return (pos[-1] - pos[-2]) / self.dt
    
    def get_acc(self, vel):
        return (vel[-1] - vel[-2]) / self.dt
    
    def __getitem__(self, idx):
        file_name = "{}.npy".format(idx)
        file_path = os.path.join(self.root_dir, file_name)
        file_path = self.root_dir
        data = np.load(file_path)[idx]
        x_traj = data[:, 0]
        y_traj = data[:, 1]
        
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
        traj_inp = np.hstack((traj_inp, vel_acc_inp, obs_inp))
        traj_out = np.hstack((x_fut, y_fut)).flatten()
        b_inp = np.array([x_inp[-1], vx_beg, ax_beg, 0, 0, 0, y_inp[-1], vy_beg, ay_beg, 0, 0, 0])
        return torch.tensor(traj_inp), torch.tensor(traj_out), torch.tensor(b_inp)


# In[102]:


train_dataset = TrajectoryDataset("./train/test.npy", t_obs=10, dt=0.08)
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=0)
# test_dataset = TrajectoryDataset("./test2/data/", t_obs=25, dt=0.08)
# test_loader = DataLoader(test_dataset, batch_size=20, shuffle=True, num_workers=0)


# In[103]:


for i, data in enumerate(train_loader):
    traj_inp, traj_out, b_inp = data
    print(traj_inp.size(), traj_out.size(), b_inp.size())
    break


# In[104]:


out = model(traj_inp, b_inp)
out.size()


# #### Training

# In[105]:


criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# In[106]:


traj_inp.size(), traj_out.size()


# In[107]:


epoch_train_loss = []
num_epochs = 25

for epoch in range(num_epochs):
    train_loss = []
    for batch_num, data in enumerate(train_loader):
        traj_inp, traj_out, b_inp = data
        traj_inp = traj_inp.to(device)
        traj_out = traj_out.to(device)
        b_inp = b_inp.to(device)

        out = model(traj_inp, b_inp)
        loss = criterion(out, traj_out)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.item())
        
        cnt = 0
        for i in range(traj_inp.size()[0]):
            plot_traj(cnt, traj_inp[i], traj_out[i], out[i], {"x": x_obs_temp, "y": y_obs_temp})
            cnt += 1
        
        if batch_num % 10 == 0:
            print("Epoch: {}, Batch: {}, Loss: {}".format(epoch, batch_num, loss.item()))
    
    mean_loss = np.mean(train_loss)
    epoch_train_loss.append(mean_loss)
    print("Epoch: {}, Mean Loss: {}".format(epoch, mean_loss))
    print("-"*100)


# #### Testing

# In[ ]:


def plot_traj(cnt, traj_inp, traj_out, traj_pred, obs=[]):
    traj_inp = traj_inp.numpy()[:2*num]
    traj_out = traj_out.numpy()
    traj_pred = traj_pred.detach().numpy()
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(traj_inp[::2], traj_inp[1::2], label='Inp traj')
    ax.scatter(traj_out[:num], traj_out[num:], label='GT')
    ax.scatter(traj_pred[:num], traj_pred[num:], label='Pred')
#     ax.scatter(obs["x"], obs["y"], s=20, label="Obstacles")
    
    x_obs_temp = obs["x"]
    y_obs_temp = obs["y"]
    
#     th = np.linspace(0, 2 * np.pi, 100)
#     for i in range(0, num_obs):
#         x_circ = x_obs_temp[i] + a_obs * np.cos(th)
#         y_circ = y_obs_temp[i] + b_obs * np.sin(th)
#         ax.plot(x_circ, y_circ, '-k')
    
    ax.legend()
    plt.savefig('./results/{}.png'.format(cnt))
    plt.close()

# def plot_traj(cnt, traj_inp, traj_out, traj_pred):
#     traj_inp = traj_inp.numpy()[:2*num]
#     traj_out = traj_out.numpy()
#     traj_pred = traj_pred.numpy()
    
#     fig, ax = plt.subplots(1, 1, figsize=(6, 6))
#     ax.scatter(traj_inp[::2], traj_inp[1::2], label='Inp traj')
#     ax.scatter(traj_out[:25], traj_out[25:], label='GT')
#     ax.scatter(traj_pred[:25], traj_pred[25:], label='Pred')
    
#     th = np.linspace(0, 2 * np.pi, 100)
#     for i in range(0, num_obs):
#         x_circ = x_obs_temp[i] + a_obs * np.cos(th)
#         y_circ = y_obs_temp[i] + b_obs * np.sin(th)
#         ax.plot(x_circ, y_circ, '-k')
    
#     ax.legend()
#     ax.set_xlim([-7, 7])
#     ax.set_ylim([-7, 7])
#     plt.savefig('./results2/{}.png'.format(cnt))
#     plt.close()


# In[ ]:


with torch.no_grad():
    cnt = 0
    test_loss = []
    for batch_num, data in enumerate(test_loader):
        traj_inp, traj_out, b_inp = data
        traj_inp = traj_inp.to(device)
        traj_out = traj_out.to(device)
        b_inp = b_inp.to(device)

        out = model(traj_inp, b_inp)
        loss = criterion(out, traj_out)
        
        test_loss.append(loss.item())
        print("Batch: {}, Loss: {}".format(batch_num, loss.item()))
        for i in range(traj_inp.size()[0]):
            plot_traj(cnt, traj_inp[i], traj_out[i], out[i])
            cnt += 1
mean_loss = np.mean(test_loss)
print("Epoch Mean Test Loss: {}".format(mean_loss))


# In[ ]:


traj_inp.numpy()[0][:2*num].shape, traj_out.numpy()[0].shape


# In[ ]:


traj_inp[6]


# In[ ]:


traj_out[6]


# In[ ]:


out[6]


# In[ ]:


torch.sqrt(torch.sum((traj_out[6] - out[6]) ** 2))


# In[ ]:


fin = torch.tensor([traj_out[6][1], traj_out[6][23]])
fin_pred = torch.tensor([out[6][11], out[6][23]])
fin, fin_pred


# In[ ]:


torch.sqrt(torch.sum((fin_pred - fin) ** 2))


# In[ ]:


torch.sqrt(torch.sum((traj_out[6][-1] - out[6][-1]) ** 2))


# In[ ]:





# In[ ]:


criterion(traj_out[6], out[6])


# In[ ]:


def disp_traj(cnt, traj_inp, traj_out, traj_pred):
    traj_inp = traj_inp.numpy()
    traj_out = traj_out.numpy()
    traj_pred = traj_pred.numpy()
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(traj_inp[::2], traj_inp[1::2], label='Inp traj')
    ax.scatter(traj_out[:12], traj_out[12:], label='GT')
    ax.scatter(traj_out[11], traj_out[23], label='Fin GT')
    ax.scatter(traj_pred[:12], traj_pred[12:], label='Pred')
    ax.scatter(traj_pred[11], traj_pred[23], label='Fin Pred')
    
    th = np.linspace(0, 2 * np.pi, 100)
    for i in range(0, num_obs):
        x_circ = x_obs_temp[i] + a_obs * np.cos(th)
        y_circ = y_obs_temp[i] + b_obs * np.sin(th)
        ax.plot(x_circ, y_circ, '-k')
    
    ax.legend()
    ax.set_xlim([-7, 7])
    ax.set_ylim([-7, 7])
    plt.show()


# In[ ]:


disp_traj(0, traj_inp[8], traj_out[8], out[8].detach())

