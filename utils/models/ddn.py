import os
import sys
# sys.path.append("/Users/shashanks./Downloads/Installations/ddn/")
sys.path.append("./ddn/")
sys.path.append("./")
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import scipy.special
import torch.nn as nn
import matplotlib.pyplot as plt

from scipy.linalg import block_diag
from torch.utils.data import Dataset, DataLoader
from ddn.pytorch.node import AbstractDeclarativeNode

def bernstein_coeff_order10_new(n, tmin, tmax, t_actual):
    l = tmax - tmin
    t = (t_actual - tmin) / l

    P0 = scipy.special.binom(n, 0) * ((1 - t) ** (n - 0)) * t ** 0
    P1 = scipy.special.binom(n, 1) * ((1 - t) ** (n - 1)) * t ** 1
    P2 = scipy.special.binom(n, 2) * ((1 - t) ** (n - 2)) * t ** 2
    P3 = scipy.special.binom(n, 3) * ((1 - t) ** (n - 3)) * t ** 3
    P4 = scipy.special.binom(n, 4) * ((1 - t) ** (n - 4)) * t ** 4
    P5 = scipy.special.binom(n, 5) * ((1 - t) ** (n - 5)) * t ** 5
    P6 = scipy.special.binom(n, 6) * ((1 - t) ** (n - 6)) * t ** 6
    P7 = scipy.special.binom(n, 7) * ((1 - t) ** (n - 7)) * t ** 7
    P8 = scipy.special.binom(n, 8) * ((1 - t) ** (n - 8)) * t ** 8
    P9 = scipy.special.binom(n, 9) * ((1 - t) ** (n - 9)) * t ** 9
    P10 = scipy.special.binom(n, 10) * ((1 - t) ** (n - 10)) * t ** 10

    P0dot = -10.0 * (-t + 1) ** 9
    P1dot = -90.0 * t * (-t + 1) ** 8 + 10.0 * (-t + 1) ** 9
    P2dot = -360.0 * t ** 2 * (-t + 1) ** 7 + 90.0 * t * (-t + 1) ** 8
    P3dot = -840.0 * t ** 3 * (-t + 1) ** 6 + 360.0 * t ** 2 * (-t + 1) ** 7
    P4dot = -1260.0 * t ** 4 * (-t + 1) ** 5 + 840.0 * t ** 3 * (-t + 1) ** 6
    P5dot = -1260.0 * t ** 5 * (-t + 1) ** 4 + 1260.0 * t ** 4 * (-t + 1) ** 5
    P6dot = -840.0 * t ** 6 * (-t + 1) ** 3 + 1260.0 * t ** 5 * (-t + 1) ** 4
    P7dot = -360.0 * t ** 7 * (-t + 1) ** 2 + 840.0 * t ** 6 * (-t + 1) ** 3
    P8dot = 45.0 * t ** 8 * (2 * t - 2) + 360.0 * t ** 7 * (-t + 1) ** 2
    P9dot = -10.0 * t ** 9 + 9 * t ** 8 * (-10.0 * t + 10.0)
    P10dot = 10.0 * t ** 9

    P0ddot = 90.0 * (-t + 1) ** 8
    P1ddot = 720.0 * t * (-t + 1) ** 7 - 180.0 * (-t + 1) ** 8
    P2ddot = 2520.0 * t ** 2 * (-t + 1) ** 6 - 1440.0 * t * (-t + 1) ** 7 + 90.0 * (-t + 1) ** 8
    P3ddot = 5040.0 * t ** 3 * (-t + 1) ** 5 - 5040.0 * t ** 2 * (-t + 1) ** 6 + 720.0 * t * (-t + 1) ** 7
    P4ddot = 6300.0 * t ** 4 * (-t + 1) ** 4 - 10080.0 * t ** 3 * (-t + 1) ** 5 + 2520.0 * t ** 2 * (-t + 1) ** 6
    P5ddot = 5040.0 * t ** 5 * (-t + 1) ** 3 - 12600.0 * t ** 4 * (-t + 1) ** 4 + 5040.0 * t ** 3 * (-t + 1) ** 5
    P6ddot = 2520.0 * t ** 6 * (-t + 1) ** 2 - 10080.0 * t ** 5 * (-t + 1) ** 3 + 6300.0 * t ** 4 * (-t + 1) ** 4
    P7ddot = -360.0 * t ** 7 * (2 * t - 2) - 5040.0 * t ** 6 * (-t + 1) ** 2 + 5040.0 * t ** 5 * (-t + 1) ** 3
    P8ddot = 90.0 * t ** 8 + 720.0 * t ** 7 * (2 * t - 2) + 2520.0 * t ** 6 * (-t + 1) ** 2
    P9ddot = -180.0 * t ** 8 + 72 * t ** 7 * (-10.0 * t + 10.0)
    P10ddot = 90.0 * t ** 8
    90.0 * t ** 8

    P = np.hstack((P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10))
    Pdot = np.hstack((P0dot, P1dot, P2dot, P3dot, P4dot, P5dot, P6dot, P7dot, P8dot, P9dot, P10dot)) / l
    Pddot = np.hstack((P0ddot, P1ddot, P2ddot, P3ddot, P4ddot, P5ddot, P6ddot, P7ddot, P8ddot, P9ddot, P10ddot)) / (l ** 2)
    return P, Pdot, Pddot

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
    
class TrajNetLSTMSimple(nn.Module):
    def __init__(self, opt_layer, P, Pdot, input_size=2, hidden_size=16, embedding_size = 128, output_size=2, nvar=11, t_obs=8, num_layers = 1, device="cpu"):
        super(TrajNetLSTMSimple, self).__init__()
        self.nvar = nvar
        self.t_obs = t_obs
        self.P = torch.tensor(P, dtype=torch.double).to(device)
        self.Pdot = torch.tensor(Pdot, dtype=torch.double).to(device)        
        self.opt_layer = opt_layer
        self.linear1 = nn.Linear(input_size, embedding_size)
#         self.linear2 = nn.Linear(embedding_size, output_size)
        self.linear2 = nn.Linear(output_size, embedding_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.lstm1 = nn.LSTMCell(embedding_size, hidden_size)
        self.lstm2 = nn.LSTMCell(embedding_size, hidden_size)        
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.activation = nn.ReLU()
        self.dtype=torch.float64
        self.mask = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.double).to(device)
        
        
    def forward(self, x, fixed_params, var_inp):
        batch_size, _, _ = x.size()
        out = x
        encoder_hidden = (torch.zeros(batch_size, self.hidden_size, dtype=self.dtype), torch.zeros(batch_size, self.hidden_size, dtype=self.dtype))
#         hidden = self.lstm1(embedded, hidden)
        
        for i in range(20):
            encoder_input = x[:, i, :]
            embedded = self.activation(self.linear1(encoder_input))
            encoder_hidden = self.lstm1(embedded, encoder_hidden)
        
        decoder_input = encoder_input[:, :2]
        decoder_hidden = encoder_hidden
        
        decoder_outputs = torch.zeros(20, 30, 2)
        for i in range(30):
            embedded = self.activation(self.linear2(decoder_input))
            decoder_hidden = self.lstm2(embedded, decoder_hidden)
            decoder_output = self.linear3(decoder_hidden[0])
            decoder_input = decoder_output
            decoder_outputs[:, i, :] = decoder_output
        # Run optimization
        pad_zero = torch.zeros(decoder_output.shape[0], 1)
        variable_params = torch.cat((decoder_output, pad_zero), axis=1)
        # Run optimization
        variable_params = self.mask * var_inp + (1-self.mask) * variable_params

        sol = self.opt_layer(fixed_params, variable_params)
         
        # Compute final trajectory
        x_pred = torch.matmul(self.P, sol[:, :self.nvar].transpose(0, 1))
        y_pred = torch.matmul(self.P, sol[:, self.nvar:2*self.nvar].transpose(0, 1))
        x_pred = x_pred.transpose(0, 1)
        y_pred = y_pred.transpose(0, 1)
#         print(x_pred.shape, y_pred.shape)
        x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1], 1)
        y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1], 1)
#         x_pred = x_pred.reshape(x_pred.size[0], x_pred.size[1], 1).shape
#         y_pred = y_pred.reshape(x_pred.size[0], x_pred.size[1], 1).shape
        out = torch.cat([x_pred, y_pred], dim=2)
#         print(out.shape)
#         print(torch.cat([x_pred, y_pred]).shape)
        return out

class TrajNet(nn.Module):
    def __init__(self, opt_layer, P, Pdot, input_size=2, hidden_size=16, embedding_size = 128, output_size=2, nvar=11, t_obs=8, num_layers = 1, device="cpu"):
        super(TrajNet, self).__init__()
        self.nvar = nvar
        self.t_obs = t_obs
        self.P = torch.tensor(P, dtype=torch.double).to(device)
        self.Pdot = torch.tensor(Pdot, dtype=torch.double).to(device)
        self.linear1 = nn.Linear(input_size, embedding_size)
        self.linear2 = nn.Linear(embedding_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size + 1)
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.opt_layer = opt_layer
        self.activation = nn.ReLU()
        self.dtype=torch.float64        
#         self.mask = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.double).to(device)
        self.mask = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.double).to(device)
        
    def forward(self, x, fixed_params, var_inp):
#         batch_size, _, __ = x.size()
        batch_size, _ = x.size()        
#         x = x.reshape(batch_size, _ * __)
        out1 = self.activation(self.linear1(x)) # 40 to 128
        out2 = self.activation(self.linear2(out1)) # 128 to 16
        variable_params = self.linear3(out2) # 16 to 3
        # to shape (b, 3)
        # Run optimization
        variable_params = self.mask * var_inp + (1-self.mask) * variable_params

        sol = self.opt_layer(fixed_params, variable_params)
         
        # Compute final trajectory
        x_pred = torch.matmul(self.P, sol[:, :self.nvar].transpose(0, 1))
        y_pred = torch.matmul(self.P, sol[:, self.nvar:2*self.nvar].transpose(0, 1))
        x_pred = x_pred.transpose(0, 1)
        y_pred = y_pred.transpose(0, 1)
        out = torch.cat([x_pred, y_pred], dim=1)
        return out
    
class TrajNetLSTM(nn.Module):
    def __init__(self, opt_layer, P, Pdot, input_size=2, hidden_size=16, embedding_size = 2, output_size = 2, nvar=11, t_obs=8, num_layers = 1, device="cpu"):
        super(TrajNetLSTM, self).__init__()
        self.nvar = nvar
        self.t_obs = t_obs
        self.P = torch.tensor(P, dtype=torch.double).to(device)
        self.Pdot = torch.tensor(Pdot, dtype=torch.double).to(device)
        self.linear1 = nn.Linear(input_size, embedding_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.linear3 = nn.Linear(embedding_size, 60)
        self.linear3 = nn.Linear(hidden_size, output_size + 1)
        self.encoderlstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.decoderlstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.opt_layer = opt_layer
        self.activation = nn.ReLU()
        self.dtype=torch.float64        
        self.mask = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.double).to(device)
    
    def forward(self, x, fixed_params, var_inp):
#         batch_size, _ = x.size()
        batch_size, _, __ = x.size()
        out = x
        hidden_state = torch.zeros(self.num_layers, out.size(0), self.hidden_size, dtype=self.dtype)
        cell_state = torch.zeros(self.num_layers, out.size(0), self.hidden_size, dtype=self.dtype)
        torch.nn.init.xavier_normal_(hidden_state)
        torch.nn.init.xavier_normal_(cell_state)
        out, (hidden_state, cell_state) = self.encoderlstm(out, (hidden_state, cell_state))
        
#         # out of shape (b, 20, 2) -> (b, 30, 2)
        pad = torch.zeros(20, 10, self.hidden_size)
        out = torch.cat((out, pad), dim=1)
        
#         # from shape (b, 16) to (b, 2)
        hidden_state = self.linear2(hidden_state)
        cell_state = self.linear2(cell_state)
        out, (hidden_state, cell_state) = self.decoderlstm(out, (hidden_state, cell_state))

#         print(out[:,-1].shape)
        pad_zeros = torch.zeros(out.shape[0], 1, dtype=self.dtype)
        variable_params = torch.cat((out[:, -1],pad_zeros), dim=1)
        
#         variable_params = self.activation(self.linear(out.reshape(out2.shape[0], -1)))

        # Run optimization
        variable_params = self.mask * var_inp + (1-self.mask) * variable_params

        sol = self.opt_layer(fixed_params, variable_params)
         
        # Compute final trajectory
        x_pred = torch.matmul(self.P, sol[:, :self.nvar].transpose(0, 1))
        y_pred = torch.matmul(self.P, sol[:, self.nvar:2*self.nvar].transpose(0, 1))
        x_pred = x_pred.transpose(0, 1)
        y_pred = y_pred.transpose(0, 1)
        out = torch.cat([x_pred, y_pred], dim=1)
        return out    

class TrajNetLSTMSimpler(nn.Module):
    def __init__(self, opt_layer, P, Pdot, input_size=2, hidden_size=16, embedding_size = 64, output_size = 2, nvar=11, t_obs=8, num_layers = 1, device="cpu"):
        super(TrajNetLSTMSimpler, self).__init__()
        self.nvar = nvar
        self.t_obs = t_obs
        self.P = torch.tensor(P, dtype=torch.double).to(device)
        self.Pdot = torch.tensor(Pdot, dtype=torch.double).to(device)
        self.linear1 = nn.Linear(input_size, embedding_size)
        #self.linear2 = nn.Linear(hidden_size, output_size)
        self.linear2 = nn.Linear(embedding_size, output_size + 1)
        #self.linear3 = nn.Linear(hidden_size, output_size + 1)
        self.encoderlstm = nn.LSTM(input_size = embedding_size, hidden_size = embedding_size, batch_first=True)
        #self.decoderlstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.opt_layer = opt_layer
        self.activation = nn.ReLU()
        self.dtype=torch.float64        
        self.mask = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.double).to(device)
    
    def forward(self, x, fixed_params, var_inp):
#         batch_size, _ = x.size()
        batch_size, _, __ = x.size()
        out = self.activation(self.linear1(x))
        #hidden_state = torch.zeros(self.num_layers, out.size(0), self.hidden_size, dtype=self.dtype)
        #cell_state = torch.zeros(self.num_layers, out.size(0), self.hidden_size, dtype=self.dtype)
        #torch.nn.init.xavier_normal_(hidden_state)
        #torch.nn.init.xavier_normal_(cell_state)
        out, (hidden_state, cell_state) = self.encoderlstm(out)
        
#         # out of shape (b, 20, 2) -> (b, 30, 2)
        variable_params = self.linear2(hidden_state[0])
        #variable_params = torch.cat((out[:, -1],pad_zeros), dim=1)
        
#         variable_params = self.activation(self.linear(out.reshape(out2.shape[0], -1)))

        # Run optimization
        variable_params = self.mask * var_inp + (1-self.mask) * variable_params

        sol = self.opt_layer(fixed_params, variable_params)
         
        # Compute final trajectory
        x_pred = torch.matmul(self.P, sol[:, :self.nvar].transpose(0, 1))
        y_pred = torch.matmul(self.P, sol[:, self.nvar:2*self.nvar].transpose(0, 1))
        x_pred = x_pred.transpose(0, 1)
        y_pred = y_pred.transpose(0, 1)
        out = torch.cat([x_pred, y_pred], dim=1)
        return out    
