"""lstm_train_test.py runs the LSTM baselines training/inference on forecasting dataset.
Note: The training code for these baselines is covered under the patent <PATENT_LINK>.
Example usage:
python lstm_train_test.py 
    --model_path saved_models/lstm.pth.tar 
    --test_features ../data/forecasting_data_test.pkl 
    --train_features ../data/forecasting_data_train.pkl 
    --val_features ../data/forecasting_data_val.pkl 
    --use_delta --normalize
"""

import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Tuple, Union

import argparse
import joblib
from joblib import Parallel, delayed
import numpy as np
import pickle as pkl
from termcolor import cprint
import torch
import torch.nn as nn
import torch.nn.functional as F

from logger import Logger
import utils.baseline_config as config
import utils.baseline_utils as baseline_utils
from utils.lstm_utils import ModelUtils, LSTMDataset
from ddn.pytorch.node import AbstractDeclarativeNode
from bernstein import bernstein_coeff_order10_new

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
global_step = 0
best_loss = float("inf")
np.random.seed(100)

ROLLOUT_LENS = [1, 10, 30]


def parse_arguments() -> Any:
    """Arguments for running the baseline.
    Returns:
        parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_batch_size",
                        type=int,
                        default=512,
                        help="Test batch size")
    parser.add_argument("--model_path",
                        required=False,
                        type=str,
                        help="path to the saved model")
    parser.add_argument("--obs_len",
                        default=20,
                        type=int,
                        help="Observed length of the trajectory")
    parser.add_argument("--pred_len",
                        default=30,
                        type=int,
                        help="Prediction Horizon")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the trajectories if non-map baseline is used",
    )
    parser.add_argument(
        "--use_delta",
        action="store_true",
        help="Train on the change in position, instead of absolute position",
    )
    parser.add_argument(
        "--train_features",
        default="",
        type=str,
        help="path to the file which has train features.",
    )
    parser.add_argument(
        "--val_features",
        default="",
        type=str,
        help="path to the file which has val features.",
    )
    parser.add_argument(
        "--test_features",
        default="",
        type=str,
        help="path to the file which has test features.",
    )
    parser.add_argument(
        "--joblib_batch_size",
        default=100,
        type=int,
        help="Batch size for parallel computation",
    )
    parser.add_argument("--use_map",
                        action="store_true",
                        help="Use the map based features")
    parser.add_argument("--use_social",
                        action="store_true",
                        help="Use social features")
    parser.add_argument("--test",
                        action="store_true",
                        help="If true, only run the inference")
    parser.add_argument("--train_batch_size",
                        type=int,
                        default=512,
                        help="Training batch size")
    parser.add_argument("--val_batch_size",
                        type=int,
                        default=512,
                        help="Val batch size")
    parser.add_argument("--end_epoch",
                        type=int,
                        default=5000,
                        help="Last epoch")
    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="Learning rate")
    parser.add_argument(
        "--traj_save_path",
        required=False,
        type=str,
        help=
        "path to the pickle file where forecasted trajectories will be saved.",
    )
    return parser.parse_args()


class EncoderRNN(nn.Module):
    """Encoder Network."""
    def __init__(self,
                 input_size: int = 2,
                 embedding_size: int = 8,
                 hidden_size: int = 16):
        """Initialize the encoder network.
        Args:
            input_size: number of features in the input
            embedding_size: Embedding size
            hidden_size: Hidden size of LSTM
        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(input_size, embedding_size)
        self.lstm1 = nn.LSTMCell(embedding_size, hidden_size)

    def forward(self, x: torch.FloatTensor, hidden: Any) -> Any:
        """Run forward propagation.
        Args:
            x: input to the network
            hidden: initial hidden state
        Returns:
            hidden: final hidden 
        """
        embedded = F.relu(self.linear1(x))
        hidden = self.lstm1(embedded, hidden)
        return hidden


class DecoderRNN(nn.Module):
    """Decoder Network."""
    def __init__(self, embedding_size=8, hidden_size=16, output_size=2):
        """Initialize the decoder network.
        Args:
            embedding_size: Embedding size
            hidden_size: Hidden size of LSTM
            output_size: number of features in the output
        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(output_size, embedding_size)
        self.lstm1 = nn.LSTMCell(embedding_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        """Run forward propagation.
        Args:
            x: input to the network
            hidden: initial hidden state
        Returns:
            output: output from lstm
            hidden: final hidden state
        """
        embedded = F.relu(self.linear1(x))
        hidden = self.lstm1(embedded, hidden)
        output = self.linear2(hidden[0])
        return output, hidden

class OPTNode(AbstractDeclarativeNode):
    def __init__(self, rho_eq=1.0, rho_goal=1.0, rho_nonhol=1.0, rho_psi=1.0, maxiter=1000, weight_smoothness=1.0, weight_smoothness_psi=1.0, t_fin=2.0, num=30):
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
        self.A_eq_psi = np.vstack((self.P[0], self.Pdot[0], self.P[-1]))
        
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
        
        self.lamda_x = None
        self.lamda_y = None
        self.lamda_psi = None
        
    def compute_x(self, v, psi, b_eq_x, b_eq_y):
        b_nonhol_x = v * torch.cos(psi)
        b_nonhol_y = v * torch.sin(psi)
    
        cost = self.cost_smoothness + self.rho_nonhol * torch.matmul(self.A_nonhol.T, self.A_nonhol) + self.rho_eq * torch.matmul(self.A_eq.T, self.A_eq)
        lincost_x = -self.lamda_x - self.rho_nonhol * torch.matmul(self.A_nonhol.T, b_nonhol_x.T).T - self.rho_eq * torch.matmul(self.A_eq.T, b_eq_x.T).T
        lincost_y = -self.lamda_y - self.rho_nonhol * torch.matmul(self.A_nonhol.T, b_nonhol_y.T).T - self.rho_eq * torch.matmul(self.A_eq.T, b_eq_y.T).T

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
        lincost_psi = -self.lamda_psi - self.rho_psi * torch.matmul(self.A_psi.T, psi_temp.T).T - self.rho_eq * torch.matmul(self.A_eq_psi.T, b_eq_psi.T).T

        cost_inv = torch.linalg.inv(cost)

        sol_psi = torch.matmul(-cost_inv, lincost_psi.T).T

        psi = torch.matmul(self.P, sol_psi.T).T

        res_psi = torch.matmul(self.A_psi, sol_psi.T).T - psi_temp
        res_eq_psi = torch.matmul(self.A_eq_psi, sol_psi.T).T - b_eq_psi

        self.lamda_psi = self.lamda_psi - self.rho_psi * torch.matmul(self.A_psi.T, res_psi.T).T - self.rho_eq * torch.matmul(self.A_eq_psi.T, res_eq_psi.T).T

        return sol_psi, 0, 0, psi

    
    def solve(self, fixed_params, variable_params):
        batch_size, _ = fixed_params.size()
        x_init, y_init, v_init, psi_init, psidot_init = torch.chunk(fixed_params, 5, dim=1)
        x_fin, y_fin, psi_fin = torch.chunk(variable_params, 3, dim=1)
        
        b_eq_x = torch.cat((x_init, x_fin), dim=1)
        b_eq_y = torch.cat((y_init, y_fin), dim=1)
        b_eq_psi = torch.cat((psi_init, psidot_init, psi_fin), dim=1)
        
        v = torch.ones(batch_size, self.num, dtype=torch.double).to(device) * v_init
        psi = torch.ones(batch_size, self.num, dtype=torch.double).to(device) * psi_init
        xdot = v * torch.cos(psi)
        ydot = v * torch.sin(psi)
        
        self.lamda_x = torch.zeros(batch_size, self.nvar, dtype=torch.double).to(device)
        self.lamda_y = torch.zeros(batch_size, self.nvar, dtype=torch.double).to(device)
        self.lamda_psi = torch.zeros(batch_size, self.nvar, dtype=torch.double).to(device)
        
        for i in range(0, self.maxiter):
            psi_temp = torch.atan2(ydot, xdot)
            c_psi, res_psi, res_eq_psi, psi = self.compute_psi(psi, self.lamda_psi, psi_temp, b_eq_psi)
            c_x, c_y, x, xdot, y, ydot = self.compute_x(v, psi, b_eq_x, b_eq_y)
            
            v = torch.sqrt(xdot ** 2 + ydot ** 2)
            #v[:, 0] = v_init[:, 0]

            res_eq_x = torch.matmul(self.A_eq, c_x.T).T - b_eq_x
            res_nonhol_x = xdot - v * torch.cos(psi)

            res_eq_y = torch.matmul(self.A_eq, c_y.T).T - b_eq_y
            res_nonhol_y = ydot - v * torch.sin(psi)

            
            self.lamda_x = self.lamda_x - self.rho_eq * torch.matmul(self.A_eq.T, res_eq_x.T).T - self.rho_nonhol * torch.matmul(self.A_nonhol.T, res_nonhol_x.T).T
            self.lamda_y = self.lamda_y - self.rho_eq * torch.matmul(self.A_eq.T, res_eq_y.T).T - self.rho_nonhol * torch.matmul(self.A_nonhol.T, res_nonhol_y.T).T
        
        
        primal_sol = torch.hstack((c_x, c_y, c_psi, v))
        return primal_sol, None
    
    def objective(self, fixed_params, variable_params, y):
        c_x = y[:, :self.nvar]
        c_y = y[:, self.nvar:2*self.nvar]
        c_psi = y[:, 2*self.nvar:3*self.nvar]
        v = y[:, 3*self.nvar:]
        
        x_init, y_init, v_init, psi_init, psidot_init = torch.chunk(fixed_params, 5, dim=1)
        x_fin, y_fin, psi_fin = torch.chunk(variable_params, 3, dim=1)
        
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
        cost_psi = 0.5*self.rho_eq*(torch.sum((psi[:, -1] - psi_fin) ** 2, 1) + torch.sum((psi[:, 0] - psi_init) ** 2, 1)
                                    + torch.sum((psidot[:, 0] - psidot_init) ** 2, 1))
        #cost_v = 0.5*self.rho_eq*torch.sum((v[:, 0] - v_init) ** 2, 1)
        cost_cancel = torch.diagonal(torch.matmul(-self.lamda_x, c_x.T) + torch.matmul(-self.lamda_y, c_y.T) + torch.matmul(-self.lamda_psi, c_psi.T))
        
        cost_smoothness = 0.5*self.weight_smoothness*(torch.sum(xddot**2, 1) + torch.sum(yddot**2, 1)) + 0.5*self.weight_smoothness_psi*torch.sum(psiddot**2, 1)
        return cost_nonhol + cost_pos + cost_psi + cost_smoothness + cost_cancel #+ cost_v 


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

class OptimNode(nn.Module):
    def __init__(self, opt_layer, P, nvar=11):
        super().__init__()
        self.opt_layer = opt_layer
        self.nvar = nvar
        self.P = P

    def forward(self, fixed_inp, var_inp):
        sol = self.opt_layer(fixed_inp, var_inp)
        x_pred = torch.matmul(self.P, sol[:, :self.nvar].transpose(0, 1))
        y_pred = torch.matmul(self.P, sol[:, self.nvar:2*self.nvar].transpose(0, 1))
        x_pred = x_pred.transpose(0, 1)
        y_pred = y_pred.transpose(0, 1)
        out = torch.stack([x_pred, y_pred], dim=2)
        
        return out

def train(
        train_loader: Any,
        epoch: int,
        criterion: Any,
        logger: Logger,
        encoder: Any,
        decoder: Any,
        model: Any,
        encoder_optimizer: Any,
        decoder_optimizer: Any,
        model_utils: ModelUtils,
        rollout_len: int = 30,
) -> None:
    """Train the lstm network.
    Args:
        train_loader: DataLoader for the train set
        epoch: epoch number
        criterion: Loss criterion
        logger: Tensorboard logger
        encoder: Encoder network instance
        decoder: Decoder network instance
        encoder_optimizer: optimizer for the encoder network
        decoder_optimizer: optimizer for the decoder network
        model_utils: instance for ModelUtils class
        rollout_len: current prediction horizon
    """
    args = parse_arguments()
    global global_step

    for i, (_input, target, helpers) in enumerate(train_loader):
        _input = _input.to(device)
        target = target.to(device)

        # Set to train mode
        encoder.train()
        decoder.train()

        # Zero the gradients
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Encoder
        batch_size = _input.shape[0]
        input_length = _input.shape[1]
        output_length = target.shape[1]
        input_shape = _input.shape[2]

        # Initialize encoder hidden state
        encoder_hidden = model_utils.init_hidden(
            batch_size,
            encoder.module.hidden_size if use_cuda else encoder.hidden_size)

        # Initialize losses
        loss = 0

        # Encode observed trajectory
        for ei in range(input_length):
            encoder_input = _input[:, ei, :]
            encoder_hidden = encoder(encoder_input, encoder_hidden)

        # Initialize decoder input with last coordinate in encoder
        decoder_input = encoder_input[:, :2]

        # Initialize decoder hidden state as encoder hidden state
        decoder_hidden = encoder_hidden

        decoder_outputs = torch.zeros(target.shape).to(device)


        psi_init = None
        # Decode hidden state in future trajectory
        for di in range(rollout_len):
            decoder_output, decoder_hidden = decoder(decoder_input,
                                                     decoder_hidden)
            decoder_outputs[:, di, :] = decoder_output

            if psi_init is None:
                dx = decoder_output[:, 0] - decoder_input[:, 0]
                dy = decoder_output[:, 1] - decoder_input[:, 1]
                psi_init = torch.atan(torch.div(torch.sin(torch.div(dy, dx)), torch.cos(torch.div(dy, dx))))

            # Update loss
            #loss += criterion(decoder_output[:, :2], target[:, di, :2])
            dx = decoder_output[:, 0] - decoder_input[:, 0]
            dy = decoder_output[:, 1] - decoder_input[:, 1]
            psi_fin = torch.atan(torch.div(torch.sin(torch.div(dy, dx)), torch.cos(torch.div(dy, dx))))
            
            # Use own predictions as inputs at next step
            decoder_input = decoder_output

        

        fixed_params = torch.stack([encoder_input[:, 0], encoder_input[:, 1], torch.zeros(batch_size).to(device), psi_init, torch.zeros(batch_size).to(device)]).double().transpose(0, 1).to(device)
        variable_params = torch.stack([decoder_output[:, 0], decoder_output[:, 1], psi_fin]).double().transpose(0, 1).to(device)

        out = model.forward(fixed_params, variable_params)
        loss = criterion(out, target.double())
        
        # Get average loss for pred_len
        
        
        # Backpropagate
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        if global_step % 1000 == 0:

            # Log results
            print(
                f"Train -- Epoch:{epoch}, loss:{loss}, Rollout:{rollout_len}")

            logger.scalar_summary(tag="Train/loss",
                                  value=loss.item(),
                                  step=epoch)

        global_step += 1


def validate(
        val_loader: Any,
        epoch: int,
        criterion: Any,
        logger: Logger,
        encoder: Any,
        decoder: Any,
        encoder_optimizer: Any,
        decoder_optimizer: Any,
        model_utils: ModelUtils,
        prev_loss: float,
        decrement_counter: int,
        rollout_len: int = 30,
) -> Tuple[float, int]:
    """Validate the lstm network.
    Args:
        val_loader: DataLoader for the train set
        epoch: epoch number
        criterion: Loss criterion
        logger: Tensorboard logger
        encoder: Encoder network instance
        decoder: Decoder network instance
        encoder_optimizer: optimizer for the encoder network
        decoder_optimizer: optimizer for the decoder network
        model_utils: instance for ModelUtils class
        prev_loss: Loss in the previous validation run
        decrement_counter: keeping track of the number of consecutive times loss increased in the current rollout
        rollout_len: current prediction horizon
    """
    args = parse_arguments()
    global best_loss
    total_loss = []

    for i, (_input, target, helpers) in enumerate(val_loader):

        _input = _input.to(device)
        target = target.to(device)

        # Set to eval mode
        encoder.eval()
        decoder.eval()

        # Encoder
        batch_size = _input.shape[0]
        input_length = _input.shape[1]
        output_length = target.shape[1]
        input_shape = _input.shape[2]

        # Initialize encoder hidden state
        encoder_hidden = model_utils.init_hidden(
            batch_size,
            encoder.module.hidden_size if use_cuda else encoder.hidden_size)

        # Initialize loss
        loss = 0

        # Encode observed trajectory
        for ei in range(input_length):
            encoder_input = _input[:, ei, :]
            encoder_hidden = encoder(encoder_input, encoder_hidden)

        # Initialize decoder input with last coordinate in encoder
        decoder_input = encoder_input[:, :2]

        # Initialize decoder hidden state as encoder hidden state
        decoder_hidden = encoder_hidden

        decoder_outputs = torch.zeros(target.shape).to(device)

        # Decode hidden state in future trajectory
        for di in range(output_length):
            decoder_output, decoder_hidden = decoder(decoder_input,
                                                     decoder_hidden)
            decoder_outputs[:, di, :] = decoder_output

            # Update losses for all benchmarks
            loss += criterion(decoder_output[:, :2], target[:, di, :2])

            # Use own predictions as inputs at next step
            decoder_input = decoder_output

        # Get average loss for pred_len
        loss = loss / output_length
        total_loss.append(loss)

        if i % 10 == 0:

            cprint(
                f"Val -- Epoch:{epoch}, loss:{loss}, Rollout: {rollout_len}",
                color="green",
            )

    # Save
    val_loss = sum(total_loss) / len(total_loss)

    if val_loss <= best_loss:
        best_loss = val_loss
        if args.use_map:
            save_dir = "saved_models/lstm_map"
        elif args.use_social:
            save_dir = "saved_models/lstm_social"
        else:
            save_dir = "saved_models/lstm"

        os.makedirs(save_dir, exist_ok=True)
        model_utils.save_checkpoint(
            save_dir,
            {
                "epoch": epoch + 1,
                "rollout_len": rollout_len,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "best_loss": val_loss,
                "encoder_optimizer": encoder_optimizer.state_dict(),
                "decoder_optimizer": decoder_optimizer.state_dict(),
            },
        )

    logger.scalar_summary(tag="Val/loss", value=val_loss.item(), step=epoch)

    # Keep track of the loss to change preiction horizon
    if val_loss <= prev_loss:
        decrement_counter = 0
    else:
        decrement_counter += 1

    return val_loss, decrement_counter


def infer_absolute(
        test_loader: torch.utils.data.DataLoader,
        encoder: EncoderRNN,
        decoder: DecoderRNN,
        start_idx: int,
        forecasted_save_dir: str,
        model_utils: ModelUtils,
):
    """Infer function for non-map LSTM baselines and save the forecasted trajectories.
    Args:
        test_loader: DataLoader for the test set
        encoder: Encoder network instance
        decoder: Decoder network instance
        start_idx: start index for the current joblib batch
        forecasted_save_dir: Directory where forecasted trajectories are to be saved
        model_utils: ModelUtils instance
    """
    args = parse_arguments()
    forecasted_trajectories = {}

    for i, (_input, target, helpers) in enumerate(test_loader):

        _input = _input.to(device)

        batch_helpers = list(zip(*helpers))

        helpers_dict = {}
        for k, v in config.LSTM_HELPER_DICT_IDX.items():
            helpers_dict[k] = batch_helpers[v]

        # Set to eval mode
        encoder.eval()
        decoder.eval()

        # Encoder
        batch_size = _input.shape[0]
        input_length = _input.shape[1]
        input_shape = _input.shape[2]

        # Initialize encoder hidden state
        encoder_hidden = model_utils.init_hidden(
            batch_size,
            encoder.module.hidden_size if use_cuda else encoder.hidden_size)

        # Encode observed trajectory
        for ei in range(input_length):
            encoder_input = _input[:, ei, :]
            encoder_hidden = encoder(encoder_input, encoder_hidden)

        # Initialize decoder input with last coordinate in encoder
        decoder_input = encoder_input[:, :2]

        # Initialize decoder hidden state as encoder hidden state
        decoder_hidden = encoder_hidden

        decoder_outputs = torch.zeros(
            (batch_size, args.pred_len, 2)).to(device)

        # Decode hidden state in future trajectory
        for di in range(args.pred_len):
            decoder_output, decoder_hidden = decoder(decoder_input,
                                                     decoder_hidden)
            decoder_outputs[:, di, :] = decoder_output

            # Use own predictions as inputs at next step
            decoder_input = decoder_output

        # Get absolute trajectory
        abs_helpers = {}
        abs_helpers["REFERENCE"] = np.array(helpers_dict["DELTA_REFERENCE"])
        abs_helpers["TRANSLATION"] = np.array(helpers_dict["TRANSLATION"])
        abs_helpers["ROTATION"] = np.array(helpers_dict["ROTATION"])
        abs_inputs, abs_outputs = baseline_utils.get_abs_traj(
            _input.clone().cpu().numpy(),
            decoder_outputs.detach().clone().cpu().numpy(),
            args,
            abs_helpers,
        )

        for i in range(abs_outputs.shape[0]):
            seq_id = int(helpers_dict["SEQ_PATHS"][i])
            forecasted_trajectories[seq_id] = [abs_outputs[i]]

    with open(os.path.join(forecasted_save_dir, f"{start_idx}.pkl"),
              "wb") as f:
        pkl.dump(forecasted_trajectories, f)


def infer_map(
        test_loader: torch.utils.data.DataLoader,
        encoder: EncoderRNN,
        decoder: DecoderRNN,
        start_idx: int,
        forecasted_save_dir: str,
        model_utils: ModelUtils,
):
    """Infer function for map-based LSTM baselines and save the forecasted trajectories.
    Args:
        test_loader: DataLoader for the test set
        encoder: Encoder network instance
        decoder: Decoder network instance
        start_idx: start index for the current joblib batch
        forecasted_save_dir: Directory where forecasted trajectories are to be saved
        model_utils: ModelUtils instance
    """
    args = parse_arguments()
    global best_loss
    forecasted_trajectories = {}
    for i, (_input, target, helpers) in enumerate(test_loader):

        _input = _input.to(device)

        batch_helpers = list(zip(*helpers))

        helpers_dict = {}
        for k, v in config.LSTM_HELPER_DICT_IDX.items():
            helpers_dict[k] = batch_helpers[v]

        # Set to eval mode
        encoder.eval()
        decoder.eval()

        # Encoder
        batch_size = _input.shape[0]
        input_length = _input.shape[1]

        # Iterate over every element in the batch
        for batch_idx in range(batch_size):
            num_candidates = len(
                helpers_dict["CANDIDATE_CENTERLINES"][batch_idx])
            curr_centroids = helpers_dict["CENTROIDS"][batch_idx]
            seq_id = int(helpers_dict["SEQ_PATHS"][batch_idx])
            abs_outputs = []

            # Predict using every centerline candidate for the current trajectory
            for candidate_idx in range(num_candidates):
                curr_centerline = helpers_dict["CANDIDATE_CENTERLINES"][
                    batch_idx][candidate_idx]
                curr_nt_dist = helpers_dict["CANDIDATE_NT_DISTANCES"][
                    batch_idx][candidate_idx]

                _input = torch.FloatTensor(
                    np.expand_dims(curr_nt_dist[:args.obs_len].astype(float),
                                   0)).to(device)

                # Initialize encoder hidden state
                encoder_hidden = model_utils.init_hidden(
                    1, encoder.module.hidden_size
                    if use_cuda else encoder.hidden_size)

                # Encode observed trajectory
                for ei in range(input_length):
                    encoder_input = _input[:, ei, :]
                    encoder_hidden = encoder(encoder_input, encoder_hidden)

                # Initialize decoder input with last coordinate in encoder
                decoder_input = encoder_input[:, :2]

                # Initialize decoder hidden state as encoder hidden state
                decoder_hidden = encoder_hidden

                decoder_outputs = torch.zeros((1, args.pred_len, 2)).to(device)

                # Decode hidden state in future trajectory
                for di in range(args.pred_len):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden)
                    decoder_outputs[:, di, :] = decoder_output

                    # Use own predictions as inputs at next step
                    decoder_input = decoder_output

                # Get absolute trajectory
                abs_helpers = {}
                abs_helpers["REFERENCE"] = np.expand_dims(
                    np.array(helpers_dict["CANDIDATE_DELTA_REFERENCES"]
                             [batch_idx][candidate_idx]),
                    0,
                )
                abs_helpers["CENTERLINE"] = np.expand_dims(curr_centerline, 0)

                abs_input, abs_output = baseline_utils.get_abs_traj(
                    _input.clone().cpu().numpy(),
                    decoder_outputs.detach().clone().cpu().numpy(),
                    args,
                    abs_helpers,
                )

                # array of shape (1,30,2) to list of (30,2)
                abs_outputs.append(abs_output[0])
            forecasted_trajectories[seq_id] = abs_outputs

    os.makedirs(forecasted_save_dir, exist_ok=True)
    with open(os.path.join(forecasted_save_dir, f"{start_idx}.pkl"),
              "wb") as f:
        pkl.dump(forecasted_trajectories, f)


def infer_helper(
        curr_data_dict: Dict[str, Any],
        start_idx: int,
        encoder: EncoderRNN,
        decoder: DecoderRNN,
        model_utils: ModelUtils,
        forecasted_save_dir: str,
):
    """Run inference on the current joblib batch.
    Args:
        curr_data_dict: Data dictionary for the current joblib batch
        start_idx: Start idx of the current joblib batch
        encoder: Encoder network instance
        decoder: Decoder network instance
        model_utils: ModelUtils instance
        forecasted_save_dir: Directory where forecasted trajectories are to be saved
    """
    args = parse_arguments()
    curr_test_dataset = LSTMDataset(curr_data_dict, args, "test")
    curr_test_loader = torch.utils.data.DataLoader(
        curr_test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        collate_fn=model_utils.my_collate_fn,
    )

    if args.use_map:
        print(f"#### LSTM+map inference at index {start_idx} ####")
        infer_map(
            curr_test_loader,
            encoder,
            decoder,
            start_idx,
            forecasted_save_dir,
            model_utils,
        )

    else:
        print(f"#### LSTM+social inference at {start_idx} ####"
              ) if args.use_social else print(
                  f"#### LSTM inference at {start_idx} ####")
        infer_absolute(
            curr_test_loader,
            encoder,
            decoder,
            start_idx,
            forecasted_save_dir,
            model_utils,
        )


def main():
    """Main."""
    args = parse_arguments()

    if not baseline_utils.validate_args(args):
        return

    print(f"Using all ({joblib.cpu_count()}) CPUs....")
    if use_cuda:
        print(f"Using all ({torch.cuda.device_count()}) GPUs...")

    model_utils = ModelUtils()

    # key for getting feature set
    # Get features
    if args.use_map and args.use_social:
        baseline_key = "map_social"
    elif args.use_map:
        baseline_key = "map"
    elif args.use_social:
        baseline_key = "social"
    else:
        baseline_key = "none"

    # Get data
    data_dict = baseline_utils.get_data(args, baseline_key)

    # Get model
    criterion = nn.MSELoss()
    encoder = EncoderRNN(
        input_size=len(baseline_utils.BASELINE_INPUT_FEATURES[baseline_key]))
    decoder = DecoderRNN(output_size=2)
    problem = OPTNode()
    opt_layer = DeclarativeLayer(problem)
    model = OptimNode(opt_layer, problem.P)
    if use_cuda:
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
        model = nn.DataParallel(model)
    encoder.to(device)
    decoder.to(device)
    model.to(device)
    
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    # If model_path provided, resume from saved checkpoint
    if args.model_path is not None and os.path.isfile(args.model_path):
        epoch, rollout_len, _ = model_utils.load_checkpoint(
            args.model_path, encoder, decoder, encoder_optimizer,
            decoder_optimizer)
        start_epoch = epoch + 1
        start_rollout_idx = ROLLOUT_LENS.index(rollout_len) + 1

    else:
        start_epoch = 0
        start_rollout_idx = 0

    if not args.test:

        # Tensorboard logger
        log_dir = os.path.join(os.getcwd(), "lstm_logs", baseline_key)

        # Get PyTorch Dataset
        train_dataset = LSTMDataset(data_dict, args, "train")
        val_dataset = LSTMDataset(data_dict, args, "val")

        # Setting Dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=model_utils.my_collate_fn,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            drop_last=False,
            shuffle=False,
            collate_fn=model_utils.my_collate_fn,
        )

        print("Training begins ...")

        decrement_counter = 0

        epoch = start_epoch
        global_start_time = time.time()
        for i in range(start_rollout_idx, len(ROLLOUT_LENS)):
            rollout_len = ROLLOUT_LENS[i]
            logger = Logger(log_dir, name="{}".format(rollout_len))
            best_loss = float("inf")
            prev_loss = best_loss
            while epoch < args.end_epoch:
                start = time.time()
                train(
                    train_loader,
                    epoch,
                    criterion,
                    logger,
                    encoder,
                    decoder,
                    model,
                    encoder_optimizer,
                    decoder_optimizer,
                    model_utils,
                    rollout_len,
                )
                end = time.time()

                print(
                    f"Training epoch completed in {(end - start) / 60.0} mins, Total time: {(end - global_start_time) / 60.0} mins"
                )

                epoch += 1
                if epoch % 5 == 0:
                    start = time.time()
                    prev_loss, decrement_counter = validate(
                        val_loader,
                        epoch,
                        criterion,
                        logger,
                        encoder,
                        decoder,
                        encoder_optimizer,
                        decoder_optimizer,
                        model_utils,
                        prev_loss,
                        decrement_counter,
                        rollout_len,
                    )
                    end = time.time()
                    print(
                        f"Validation completed in {(end - start) / 60.0} mins, Total time: {(end - global_start_time) / 60.0} mins"
                    )

                    # If val loss increased 3 times consecutively, go to next rollout length
                    if decrement_counter > 2:
                        break

    else:

        start_time = time.time()

        temp_save_dir = tempfile.mkdtemp()

        test_size = data_dict["test_input"].shape[0]
        test_data_subsets = baseline_utils.get_test_data_dict_subset(
            data_dict, args)

        # test_batch_size should be lesser than joblib_batch_size
        Parallel(n_jobs=-2, verbose=2)(
            delayed(infer_helper)(test_data_subsets[i], i, encoder, decoder,
                                  model_utils, temp_save_dir)
            for i in range(0, test_size, args.joblib_batch_size))

        baseline_utils.merge_saved_traj(temp_save_dir, args.traj_save_path)
        shutil.rmtree(temp_save_dir)

        end = time.time()
        print(f"Test completed in {(end - start_time) / 60.0} mins")
        print(f"Forecasted Trajectories saved at {args.traj_save_path}")


if __name__ == "__main__":
    main()
