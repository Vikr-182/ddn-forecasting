#!/usr/bin/env python
# coding: utf-8

# In[2]:


import time
import numpy as np
import scipy.special
import jax.numpy as jnp
import matplotlib.pyplot as plt
# import optimizer_traj_opt

from jax import jit, jacfwd, jacrev, random, vmap,grad
from jax.config import config
config.update("jax_enable_x64", True)


# In[3]:


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

    P = np.hstack((P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10))
    Pdot = np.hstack((P0dot, P1dot, P2dot, P3dot, P4dot, P5dot, P6dot, P7dot, P8dot, P9dot, P10dot)) / l
    Pddot = np.hstack((P0ddot, P1ddot, P2ddot, P3ddot, P4ddot, P5ddot, P6ddot, P7ddot, P8ddot, P9ddot, P10ddot)) / (l ** 2)
    return P, Pdot, Pddot


# In[4]:


import matplotlib.pyplot as plt

def plot(traj, obs, cnt = 0, draw = True, x_mid = None, y_mid = None):
    plt.xlim([np.amin(traj["x"])-1, np.amax(traj["x"])+1])    
    plt.ylim([np.amin(traj["y"])-1, np.amax(traj["y"])+1])    
    plt.scatter(traj["x"], traj["y"], label="Trajectory")    
    plt.scatter(obs["x"], obs["y"], s=200, label="Obstacles")
    if x_mid is not None:
        plt.plot(x_mid[0], y_mid[0],  'og', markersize = 20.0)
        plt.plot(x_mid[1], y_mid[1],  'og', markersize = 20.0)
        plt.plot(x_mid[2], y_mid[2],  'og', markersize = 20.0)

    plt.axis('equal')    
    plt.legend()
    plt.savefig("tests/{}.png".format(cnt))
    plt.clf()
    if draw:
        plt.show()


# In[5]:


class OptimizerLane():

    def __init__(self):
        self.rho_eq = 1.0
        self.rho_goal = 1.0
        self.rho_lane = 1.0
        self.rho_nonhol = 1.0
        self.rho_w_psi = 1.0
        self.rho_psi = 1.0
        self.maxiter = 500
        self.weight_smoothness = 1.0
        self.weight_smoothness_psi = 1.0

        self.t_fin = 8.0
        self.num = 100
        self.t = self.t_fin/self.num

        tot_time = np.linspace(0.0, self.t_fin, self.num)
        tot_time_copy = tot_time.reshape(self.num, 1)
        self.P, self.Pdot, self.Pddot = bernstein_coeff_order10_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)
        self.nvar = np.shape(self.P)[1]
        self.A_eq_psi = np.vstack((  self.P[0], self.Pdot[0], self.P[-1], self.Pdot[-1]         ))
        
        self.cost_smoothness = self.weight_smoothness*np.dot(self.Pddot.T, self.Pddot)
        self.cost_smoothness_v = self.weight_smoothness*np.dot(self.Pddot.T, self.Pddot)
        
        self.cost_smoothness_psi = self.weight_smoothness_psi*np.dot(self.Pddot.T, self.Pddot)
        self.lincost_smoothness_psi = np.zeros(self.nvar)
        
        
        self.rho_mid = 0.01
        self.mid_idx = np.array([ int(self.num/4), int(self.num/2), int(3*self.num/4)  ])        
        
        self.P_jax = jnp.asarray(self.P)
        self.Pdot_jax = jnp.asarray(self.Pdot)
        self.Pddot_jax = jnp.asarray(self.Pddot)

        # self.lamda_wc = np.zeros(self.nvar)
        # self.lamda_ws = np.zeros(self.nvar)

        # self.lamda_psi = np.zeros(self.nvar)
        # self.lamda_v = np.zeros(self.nvar)
        # self.cost_smoothness = block_diag(self.weight_smoothness*np.dot(self.Pddot.T, self.Pddot), 0.01*np.dot(self.Pddot.T, self.Pddot) )
        ######################################################################### Converting to Jax for objective and gradient computation


    def compute_w_psi(self, x_init, y_init, x_fin, y_fin, x_mid, y_mid, psi, v, lamda_wc, lamda_ws):
        A_w = self.P
        A_w_psi = self.P
        b_wc_psi = np.cos(psi)
        b_ws_psi = np.sin(psi)
        
        temp_x = np.cumsum(self.P*(v*self.t)[:, np.newaxis] , axis = 0)
        temp_y = np.cumsum(self.P*(v*self.t)[:, np.newaxis] , axis = 0)
        
        
        
        
        A_x = temp_x[0:self.num-1] ########## \sum_t (x-x_t)^2  = \Vert A_x[-1] c_w_c_psi -x_t\Vert_2^2
        A_y = temp_y[0:self.num-1]

        A_x_goal = A_x[-1].reshape(1, self.nvar)
        b_x_goal = np.array([x_fin-x_init])

        A_y_goal = A_y[-1].reshape(1, self.nvar)
        b_y_goal = np.array([y_fin-y_init])

        A_x_mid = A_x[self.mid_idx]
        A_y_mid = A_y[self.mid_idx]

        b_x_mid = x_mid-x_init
        b_y_mid = y_mid-y_init

        obj_x_goal = self.rho_goal*np.dot(A_x_goal.T, A_x_goal)
        linterm_augment_x_goal = -self.rho_goal*np.dot(A_x_goal.T, b_x_goal)

        obj_y_goal = self.rho_goal*np.dot(A_y_goal.T, A_y_goal)
        linterm_augment_y_goal = -self.rho_goal*np.dot(A_y_goal.T, b_y_goal)

        obj_x_mid = self.rho_mid*np.dot(A_x_mid.T, A_x_mid)
        linterm_augment_x_mid = -self.rho_mid*np.dot(A_x_mid.T, b_x_mid)

        obj_y_mid = self.rho_mid*np.dot(A_y_mid.T, A_y_mid)
        linterm_augment_y_mid = -self.rho_mid*np.dot(A_y_mid.T, b_y_mid)

        obj_wc_psi = self.rho_w_psi*np.dot(A_w_psi.T, A_w_psi)
        linterm_augment_wc_psi = -self.rho_w_psi*np.dot(A_w_psi.T, b_wc_psi)
        
        obj_ws_psi = self.rho_w_psi*np.dot(A_w_psi.T, A_w_psi)
        linterm_augment_ws_psi = -self.rho_w_psi*np.dot(A_w_psi.T, b_ws_psi)

        cost_wc = obj_wc_psi+obj_x_goal+obj_x_mid
        lincost_wc = -lamda_wc+linterm_augment_x_goal+linterm_augment_wc_psi+linterm_augment_x_mid

        cost_ws = obj_y_goal+obj_ws_psi+obj_y_mid
        lincost_ws = -lamda_ws+linterm_augment_y_goal+linterm_augment_ws_psi+linterm_augment_y_mid

        c_wc_psi = np.linalg.solve(-cost_wc, lincost_wc)
        c_ws_psi = np.linalg.solve(-cost_ws, lincost_ws)

        wc = np.dot(self.P, c_wc_psi)
        ws = np.dot(self.P, c_ws_psi)

        return wc, ws, c_wc_psi, c_ws_psi


    def compute_psi(self, wc, ws, b_eq_psi, lamda_psi):

        A_psi = self.P
        b_psi = np.arctan2(ws, wc)

        obj_psi = self.rho_psi*np.dot(A_psi.T, A_psi)
        linterm_augment_psi = -self.rho_psi*np.dot(A_psi.T, b_psi)

        cost_psi = self.cost_smoothness_psi+obj_psi+self.rho_eq*np.dot(self.A_eq_psi.T, self.A_eq_psi)
        lincost_psi = -lamda_psi+linterm_augment_psi-self.rho_eq*np.dot(self.A_eq_psi.T, b_eq_psi)

        sol = np.linalg.solve(-cost_psi, lincost_psi)

        c_psi = sol[0:self.nvar]

        psi = np.dot(self.P, c_psi)
        # self.psidot = np.dot(self.Pdot, c_psi)
        # self.psiddot = np.dot(self.Pddot, c_psi)


        res_psi = np.dot(A_psi, c_psi)-b_psi
        res_eq_psi = np.dot(self.A_eq_psi, c_psi)-b_eq_psi
        lamda_psi = lamda_psi-self.rho_psi*np.dot(A_psi.T, res_psi)-self.rho_eq*np.dot(self.A_eq_psi.T, res_eq_psi)


        # self.lamda_psi = self.lamda_psi+0.90*(self.lamda_psi-lamda_psi_old)

        return psi, c_psi, np.linalg.norm(res_psi), np.linalg.norm(res_eq_psi), lamda_psi

    def compute_v(self, v_init, x_init, x_fin, x_mid, y_mid, psi, lamda_v):

        temp_x = np.cumsum(self.P*(np.cos(psi)*self.t)[:, np.newaxis], axis = 0)
        temp_y = np.cumsum(self.P*(np.sin(psi)*self.t)[:, np.newaxis], axis = 0)

        A_x = temp_x[0:self.num-1]
        A_y = temp_y[0:self.num-1]

        A_x_goal = A_x[-1].reshape(1, self.nvar)
        b_x_goal = np.array([x_fin-x_init ])

        A_y_goal = A_y[-1].reshape(1, self.nvar)
        b_y_goal = np.array([y_fin-y_init ])

        A_x_mid = A_x[self.mid_idx]
        A_y_mid = A_y[self.mid_idx]

        b_x_mid = x_mid-x_init
        b_y_mid = y_mid-y_init

        A_vel_init = self.P[0].reshape(1, self.nvar)
        b_vel_init = np.array([v_init])

        obj_x_goal = self.rho_goal*np.dot(A_x_goal.T, A_x_goal)
        linterm_augment_x_goal = -self.rho_goal*np.dot(A_x_goal.T, b_x_goal)

        obj_y_goal = self.rho_goal*np.dot(A_y_goal.T, A_y_goal)
        linterm_augment_y_goal = -self.rho_goal*np.dot(A_y_goal.T, b_y_goal)

        obj_x_mid = self.rho_mid*np.dot(A_x_mid.T, A_x_mid)
        linterm_augment_x_mid = -self.rho_mid*np.dot(A_x_mid.T, b_x_mid)

        obj_y_mid = self.rho_mid*np.dot(A_y_mid.T, A_y_mid)
        linterm_augment_y_mid = -self.rho_mid*np.dot(A_y_mid.T, b_y_mid)

        obj_v_init = self.rho_eq*np.dot(A_vel_init.T, A_vel_init)
        linterm_augment_v_init = -self.rho_eq*np.dot(A_vel_init.T, b_vel_init)

        cost = obj_x_goal+obj_y_goal+self.cost_smoothness_v+obj_x_mid+obj_y_mid+obj_v_init
        lincost = -lamda_v+linterm_augment_x_goal+linterm_augment_y_goal+linterm_augment_x_mid+linterm_augment_y_mid+linterm_augment_v_init

        sol = np.linalg.solve(-cost, lincost)

        # cv = hstack((cv_1, cv_2, sol, cv_10, cv_11))

        v = np.dot(self.P, sol)
        # self.vdot = np.dot(self.Pdot, sol)


        res_v_init = np.dot(A_vel_init, sol)-b_vel_init
        lamda_v = lamda_v-self.rho_eq*np.dot(A_vel_init.T, res_v_init)

        return sol, lamda_v, v



    def solve(self, x_init, y_init, x_fin, y_fin, v_init, v_fin, psi_init, psidot_init, psi_fin, psidot_fin, x_mid, y_mid):
        v = v_init*np.ones(self.num)
        psi = psi_init*np.ones(self.num)


        res_psi = np.ones(self.maxiter)
        res_w_psi = np.ones(self.maxiter)
        res_w = np.ones(self.maxiter)
        res_eq_psi = np.ones(self.maxiter)
        res_eq = np.ones(self.maxiter)

        lamda_wc = np.zeros(self.nvar)
        lamda_ws = np.zeros(self.nvar)
        lamda_psi = np.zeros(self.nvar)
        lamda_v = np.zeros(self.nvar)

        b_eq_psi = np.hstack((  psi_init, psidot_init, psi_fin, psidot_fin        ))


        for i in range(0, self.maxiter):


            wc, ws, c_wc_psi, c_ws_psi = self.compute_w_psi(x_init, y_init, x_fin, y_fin, x_mid, y_mid, psi, v, lamda_wc, lamda_ws)
            psi, c_psi, res_psi[i], res_eq_psi[i], lamda_psi = self.compute_psi(wc, ws, b_eq_psi, lamda_psi)
            c_v, lamda_v, v = self.compute_v(v_init, x_init, x_fin, x_mid, y_mid, psi, lamda_v)



            res_wc = wc-np.cos(psi)
            res_ws = ws-np.sin(psi)
            self.A_w = self.P

            # lamda_wc_old = self.lamda_wc
            # lamda_ws_old = self.lamda_ws


            lamda_wc = lamda_wc-self.rho_w_psi*np.dot(self.A_w.T, res_wc)
            lamda_ws = lamda_ws-self.rho_w_psi*np.dot(self.A_w.T, res_ws)

            # self.lamda_wc = self.lamda_wc+0.90*(self.lamda_wc-lamda_wc_old)
            # self.lamda_ws = self.lamda_ws+0.90*(self.lamda_ws-lamda_ws_old)


            res_w[i] = np.linalg.norm( np.hstack(( res_wc, res_ws ))  )
            # res_eq[i] = np.linalg.norm( np.hstack((  res_eq_x_vec, res_eq_y_vec     ))  )
            # res_obs_lane[i] = np.linalg.norm( np.hstack(( res_x_obs_vec_lane, res_y_obs_vec_lane         ))  )


        primal_sol = np.hstack(( c_psi, c_v         ))
        dual_sol = np.hstack((  lamda_wc, lamda_ws, lamda_psi, lamda_v       ))

        return primal_sol, dual_sol, res_eq_psi, res_w, c_v


# In[6]:


def objective(P_jax, Pdot_jax, nvar, mid_idx, t, rho_params, boundary_init, variable_params, primal_sol):
    rho_goal = rho_params[0]
    rho_eq = rho_params[1]
    rho_mid = rho_params[2]



    x_init = boundary_init[0]
    y_init = boundary_init[1]
    v_init = boundary_init[2]
    psi_init = boundary_init[3]
    psidot_init = boundary_init[4]

    # variable_params = jnp.hstack(( x_fin, y_fin, psi_fin, psidot_fin, x_mid, y_mid       ))

    x_fin = variable_params[0]
    y_fin = variable_params[1]
    psi_fin = variable_params[2]
    psidot_fin = variable_params[3]
    x_mid = variable_params[4:7]
    y_mid = variable_params[7:10]


    # c_wc_psi_jax = primal_sol_jax[0:nvar]
    # c_ws_psi_jax = primal_sol[nvar:2*nvar]

    c_psi_jax = primal_sol[0:nvar]
    c_v_jax = primal_sol[nvar:2*nvar]

    v_jax = jnp.dot(P_jax, c_v_jax)
    psi_jax = jnp.dot(P_jax, c_psi_jax)
    psidot_jax = jnp.dot(Pdot_jax, c_psi_jax)

    x_temp = x_init+jnp.cumsum(v_jax*jnp.cos(psi_jax)*t)
    y_temp = y_init+jnp.cumsum(v_jax*jnp.sin(psi_jax)*t)

    x = jnp.hstack(( x_init, x_temp[0:-1]    ))
    y = jnp.hstack(( y_init, y_temp[0:-1]    ))

    cost_final_pos = 0.5*rho_goal*((x[-1]-x_fin)**2+(y[-1]-y_fin)**2)
    cost_psi_term = 0.5*rho_eq*( ( psi_jax[0]-psi_init)**2+( psidot_jax[0]-psidot_init)**2+( psi_jax[-1]-psi_fin)**2+( psidot_jax[-1]-psidot_fin)**2 )
    cost_v_term = 0.5*rho_eq*(v_jax[0]-v_init)**2
    cost_mid_term = 0.5*rho_mid*( jnp.sum(( x[mid_idx]-x_mid  )**2)+jnp.sum(( y[mid_idx]-y_mid  )**2) )

    cost = cost_final_pos+cost_psi_term+cost_v_term+cost_mid_term

    return cost


# In[7]:


import cvxopt
from cvxopt import solvers
def compute_sol_qp(rho_obs, rho_eq, weight_smoothness, num_obs, bx_eq, by_eq, P, Pdot, Pddot, x_obs, y_obs, a_obs, b_obs):
    maxiter = 2
    nvar = np.shape(P)[1]
    num = np.shape(P)[0]

#     print(P.shape)
#     print(Pdot.shape)
#     print(Pddot.shape)
    
    A_eq = np.vstack((P[0], Pdot[0], Pddot[0], P[-1], Pdot[-1], Pddot[-1]))
    A_obs = np.tile(P, (num_obs, 1))

    cost_smoothness = weight_smoothness * np.dot(Pddot.T, Pddot)

    alpha_obs = np.zeros((num_obs, num))
    d_obs = np.ones((num_obs, num))

    lamda_x = np.zeros(nvar)
    lamda_y = np.zeros(nvar)
    res_obs = np.ones(maxiter)
    res_eq = np.ones(maxiter)
    d_min = np.ones(maxiter)
    b_x_obs = x_obs.reshape(num_obs * num)
    b_y_obs = y_obs.reshape(num_obs * num)    
    cost = cost_smoothness + rho_obs * np.dot(A_obs.T, A_obs) + rho_eq * np.dot(A_eq.T, A_eq)
    for i in range(0, maxiter):
        temp_x_obs = d_obs * np.cos(alpha_obs) * a_obs
#         print("temp_x_obs.shape: {}, x_obs.shape: {} ".format(temp_x_obs.shape, x_obs.shape))
        b_obs_x = x_obs.reshape(num * num_obs) + temp_x_obs.reshape(num * num_obs)
#         print("x_obs.shape: {}, b_obs_x.shape: {}".format(x_obs.shape, b_obs_x.shape))

        temp_y_obs = d_obs * np.sin(alpha_obs) * b_obs
        b_obs_y = y_obs.reshape(num * num_obs) + temp_y_obs.reshape(num*num_obs)

        bb_obs = np.hstack((x_obs,y_obs))
        A_eq = np.vstack((P[0], Pdot[0], Pddot[0], P[-1], Pdot[-1], Pddot[-1]))

        A_obs = np.tile(P, (num_obs, 1))
        b_eq = np.hstack((bx_eq, by_eq))

        Q_obs = np.dot(A_obs.T, A_obs)
        Q_eq = np.dot(A_eq.T, A_eq)
        qx_obs = np.dot(b_x_obs.T, A_obs) * 2
        qy_obs = np.dot(b_y_obs.T, A_obs) * 2
        qx_eq = np.dot(bx_eq.T, A_eq) * 2
        qy_eq = np.dot(by_eq.T, A_eq) * 2
        q = np.hstack((qx_eq, qy_eq))
        print(qx_obs.shape)
        A_eq = cvxopt.matrix(A_eq, tc='d')
        bx_eq_m = cvxopt.matrix(bx_eq, tc='d')
        by_eq_m = cvxopt.matrix(by_eq, tc='d')
        b_eq = cvxopt.matrix(b_eq, tc='d')
        Q = cvxopt.matrix(weight_smoothness * Q_smoothness_new + rho_obs * Q_obs + rho_eq * Q_eq, tc='d')
        q_x = cvxopt.matrix(rho_obs * qx_obs + rho_eq * qx_eq, tc='d')
        q_y = cvxopt.matrix(rho_obs * qy_obs + rho_eq * qy_eq, tc='d')
        
        print(q_x.size)
        print(bx_eq.shape)
        sol_x = solvers.qp(Q, q_x, None, None, A_eq, bx_eq_m)
        sol_x = np.array(sol_x['x'])
        sol_y = solvers.qp(Q, q_y, None, None, A_eq, by_eq_m)
        sol_y = np.array(sol_y['x'])
        x = np.dot(P, sol_x)#.reshape(num)
        y = np.dot(P, sol_y)#.reshape(num)

        print("x.shape: {}, y.shape: {}".format(x.shape, y.shape))

#         wc_alpha = (x - x_obs)
#         ws_alpha = (y - y_obs)
        x = x.reshape((num, 1))
        y = y.reshape((num, 1))
# #         print("wc_alpha.shape: {}, ws_alpha.shape: {}".format(wc_alpha.shape, ws_alpha.shape))
#         alpha_obs = np.arctan2(ws_alpha * a_obs, wc_alpha * b_obs)
        
#         c1_d = 1.0 * rho_obs * (a_obs ** 2 * np.cos(alpha_obs) ** 2 + b_obs ** 2 * np.sin(alpha_obs) ** 2)
#         c2_d = 1.0 * rho_obs * (a_obs * wc_alpha * np.cos(alpha_obs) + b_obs * ws_alpha * np.sin(alpha_obs))

#         d_temp = c2_d / c1_d
#         d_obs = np.maximum(np.ones((num_obs, num)), d_temp)
#         d_min[i] = np.amin(d_temp)

#         res_x_obs_vec = wc_alpha - a_obs * d_obs * np.cos(alpha_obs)
#         res_y_obs_vec = ws_alpha - b_obs * d_obs * np.sin(alpha_obs)
        
#         res_eq_x_vec = np.dot(A_eq, sol_x) - bx_eq
#         res_eq_y_vec = np.dot(A_eq, sol_y) - by_eq
    

    return x.squeeze(),y.squeeze()


# In[8]:


def compute_sol(rho_obs, rho_eq, weight_smoothness, num_obs, bx_eq, by_eq, P, Pdot, Pddot, x_obs, y_obs, A_obs, A_eq, a_obs=1, b_obs=1):    
    maxiter = 300
    nvar = np.shape(P)[1]
    num = np.shape(P)[0]

    A_eq = np.vstack((P[0], Pdot[0], Pddot[0], P[-1], Pdot[-1], Pddot[-1]))
    A_obs = np.tile(P, (num_obs, 1))

    cost_smoothness = weight_smoothness * np.dot(Pddot.T, Pddot)

    alpha_obs = np.zeros((num_obs, num))
    d_obs = np.ones((num_obs, num))

    lamda_x = np.zeros(nvar)
    lamda_y = np.zeros(nvar)
    res_obs = np.ones(maxiter)
    res_eq = np.ones(maxiter)
    d_min = np.ones(maxiter)

    cost = cost_smoothness + rho_obs * np.dot(A_obs.T, A_obs) + rho_eq * np.dot(A_eq.T, A_eq)

    
    for i in range(0, maxiter):
        temp_x_obs = d_obs * np.cos(alpha_obs) * a_obs
#         print("temp_x_obs.shape: {}, x_obs.shape: {} ".format(temp_x_obs.shape, x_obs.shape))
        b_obs_x = x_obs.reshape(num * num_obs) + temp_x_obs.reshape(num * num_obs)
#         print("x_obs.shape: {}, b_obs_x.shape: {}".format(x_obs.shape, b_obs_x.shape))

        temp_y_obs = d_obs * np.sin(alpha_obs) * b_obs
        b_obs_y = y_obs.reshape(num * num_obs) + temp_y_obs.reshape(num*num_obs)

        lincost_x = -lamda_x - rho_obs * np.dot(A_obs.T, b_obs_x) - rho_eq * np.dot(A_eq.T, bx_eq)
        lincost_y = -lamda_y - rho_obs * np.dot(A_obs.T, b_obs_y) - rho_eq * np.dot(A_eq.T, by_eq)

        sol_x = np.linalg.solve(-cost, lincost_x)
        sol_y = np.linalg.solve(-cost, lincost_y)
        
#         print("sol_x.shape: {}, sol_y.shape: {}".format(sol_x.shape, sol_y.shape))

        x = np.dot(P, sol_x)
        y = np.dot(P, sol_y)
        
#         print("x.shape: {}, y.shape: {}".format(x.shape, y.shape))

        wc_alpha = (x - x_obs)
        ws_alpha = (y - y_obs)
#         print("wc_alpha.shape: {}, ws_alpha.shape: {}".format(wc_alpha.shape, ws_alpha.shape))
        alpha_obs = np.arctan2(ws_alpha * a_obs, wc_alpha * b_obs)
        
        c1_d = 1.0 * rho_obs * (a_obs ** 2 * np.cos(alpha_obs) ** 2 + b_obs ** 2 * np.sin(alpha_obs) ** 2)
        c2_d = 1.0 * rho_obs * (a_obs * wc_alpha * np.cos(alpha_obs) + b_obs * ws_alpha * np.sin(alpha_obs))

        d_temp = c2_d / c1_d
        d_obs = np.maximum(np.ones((num_obs, num)), d_temp)
        d_min[i] = np.amin(d_temp)

        res_x_obs_vec = wc_alpha - a_obs * d_obs * np.cos(alpha_obs)
        res_y_obs_vec = ws_alpha - b_obs * d_obs * np.sin(alpha_obs)
        
        res_eq_x_vec = np.dot(A_eq, sol_x) - bx_eq
        res_eq_y_vec = np.dot(A_eq, sol_y) - by_eq

#         print("update : {},lamda_x: {}".format((rho_obs*np.dot(A_obs.T, res_x_obs_vec.reshape(num_obs * num)) - rho_eq * np.dot(A_eq.T, res_eq_x_vec)).shape, lamda_x.shape))
#         print("update : {}".format(rho_obs*np.dot(A_obs.T, res_y_obs_vec.reshape(num_obs * num)) - rho_eq * np.dot(A_eq.T, res_eq_y_vec).shape, lamda_y.shape))
        lamda_x = lamda_x-rho_obs*np.dot(A_obs.T, res_x_obs_vec.reshape(num_obs * num)) - rho_eq * np.dot(A_eq.T, res_eq_x_vec)
        lamda_y = lamda_y-rho_obs*np.dot(A_obs.T, res_y_obs_vec.reshape(num_obs * num)) - rho_eq * np.dot(A_eq.T, res_eq_y_vec)

        res_eq[i] = np.linalg.norm(np.hstack((res_eq_x_vec,  res_eq_y_vec)))
        res_obs[i] = np.linalg.norm(np.hstack((res_x_obs_vec, res_y_obs_vec)))

    slack_obs = np.sqrt((d_obs - 1))
#     plt.figure(1)
#     plt.plot(res_obs)
#     plt.figure(2)
#     plt.plot(res_eq)
#     plt.figure(3)
#     plt.plot(d_min)    
#     plt.show()
    return x, y, sol_x, sol_y, alpha_obs.reshape(num_obs*num), d_obs.reshape(num_obs*num), lamda_x, lamda_y, slack_obs.reshape(num_obs*num)


# In[9]:


def cost_fun_qp(aug_sol_jax, param_sol):
    x_init, vx_init, ax_init, x_fin, vx_fin, ax_fin, y_init, vy_init, ay_init, y_fin, vy_fin, ay_fin = param_sol

    bx_eq_jax =  jnp.hstack((x_init, vx_init, ax_init, x_fin, vx_fin, ax_fin))
    by_eq_jax =  jnp.hstack((y_init, vy_init, ay_init, y_fin, vy_fin, ay_fin))

    c_x = aug_sol_jax[0:nvar]
    c_y = aug_sol_jax[nvar:2*nvar]

    num_tot = num_obs * num

    alpha_obs = aug_sol_jax[2*nvar:2*nvar+num_tot]
    d_obs = aug_sol_jax[2*nvar+num_tot:2*nvar+2*num_tot]

    cost_smoothness_x = 0.5 * weight_smoothness * jnp.dot(c_x.T, jnp.dot(Q_smoothness_jax, c_x))
    cost_smoothness_y = 0.5 * weight_smoothness * jnp.dot(c_y.T, jnp.dot(Q_smoothness_jax, c_y))

    temp_x_obs = d_obs * jnp.cos(alpha_obs) * a_obs
    b_obs_x = x_obs_jax.reshape(num * num_obs) + temp_x_obs

    temp_y_obs = d_obs * jnp.sin(alpha_obs) * b_obs
    b_obs_y = y_obs_jax.reshape(num * num_obs) + temp_y_obs

    cost_obs_x = 0.5 * rho_obs * (jnp.sum((jnp.dot(A_obs_jax, c_x) - b_obs_x) ** 2))
    cost_obs_y = 0.5 * rho_obs * (jnp.sum((jnp.dot(A_obs_jax, c_y) - b_obs_y) ** 2))
    
    cost_slack = rho_obs * jnp.sum(jnp.maximum(jnp.zeros(num_tot), -d_obs + 1))
    cost_eq_x = 0.5 * rho_eq * (jnp.sum((jnp.dot(A_eq_jax, c_x) - bx_eq_jax) ** 2))
    cost_eq_y = 0.5 * rho_eq * (jnp.sum((jnp.dot(A_eq_jax, c_y) - by_eq_jax) ** 2))
    
    cost_x = cost_smoothness_x + cost_obs_x + cost_eq_x - jnp.dot(lamda_x_jax.T, c_x)
    cost_y = cost_smoothness_y + cost_obs_y + cost_eq_y - jnp.dot(lamda_y_jax.T, c_y)
   
    eps = 10 ** (-8.0)
    cost = cost_x + cost_y + eps * jnp.sum(c_x ** 2) + eps * jnp.sum(c_y ** 2) + eps * jnp.sum(d_obs ** 2) + eps * jnp.sum(alpha_obs ** 2) + cost_slack
    return cost


# In[10]:


def cost_fun(aug_sol_jax, param_sol, a_obs, b_obs, A_obs_jax, b_obs_jax, A_eq_jax, bx_eq_jax, by_eq_jax, num_obs = 4, num = 20):
    x_init, vx_init, ax_init, x_fin, vx_fin, ax_fin, y_init, vy_init, ay_init, y_fin, vy_fin, ay_fin = param_sol

    bx_eq_jax =  jnp.hstack((x_init, vx_init, ax_init, x_fin, vx_fin, ax_fin))
    by_eq_jax =  jnp.hstack((y_init, vy_init, ay_init, y_fin, vy_fin, ay_fin))

    c_x = aug_sol_jax[0:nvar]
    c_y = aug_sol_jax[nvar:2*nvar]

    num_tot = num_obs * num

    alpha_obs = aug_sol_jax[2*nvar:2*nvar+num_tot]
    d_obs = aug_sol_jax[2*nvar+num_tot:2*nvar+2*num_tot]

    cost_smoothness_x = 0.5 * weight_smoothness * jnp.dot(c_x.T, jnp.dot(Q_smoothness_jax, c_x))
    cost_smoothness_y = 0.5 * weight_smoothness * jnp.dot(c_y.T, jnp.dot(Q_smoothness_jax, c_y))

    temp_x_obs = d_obs * jnp.cos(alpha_obs) * a_obs
    b_obs_x = x_obs_jax.reshape(num * num_obs) + temp_x_obs

    temp_y_obs = d_obs * jnp.sin(alpha_obs) * b_obs
    b_obs_y = y_obs_jax.reshape(num * num_obs) + temp_y_obs

    cost_obs_x = 0.5 * rho_obs * (jnp.sum((jnp.dot(A_obs_jax, c_x) - b_obs_x) ** 2))
    cost_obs_y = 0.5 * rho_obs * (jnp.sum((jnp.dot(A_obs_jax, c_y) - b_obs_y) ** 2))
    
    cost_slack = rho_obs * jnp.sum(jnp.maximum(jnp.zeros(num_tot), -d_obs + 1))
    cost_eq_x = 0.5 * rho_eq * (jnp.sum((jnp.dot(A_eq_jax, c_x) - bx_eq_jax) ** 2))
    cost_eq_y = 0.5 * rho_eq * (jnp.sum((jnp.dot(A_eq_jax, c_y) - by_eq_jax) ** 2))
    
    cost_x = cost_smoothness_x + cost_obs_x + cost_eq_x - jnp.dot(lamda_x_jax.T, c_x)
    cost_y = cost_smoothness_y + cost_obs_y + cost_eq_y - jnp.dot(lamda_y_jax.T, c_y)
   
    eps = 10 ** (-8.0)
    cost = cost_x + cost_y + eps * jnp.sum(c_x ** 2) + eps * jnp.sum(c_y ** 2) + eps * jnp.sum(d_obs ** 2) + eps * jnp.sum(alpha_obs ** 2) + cost_slack
    return cost


# In[11]:


def generate(params, x_obs_temp, y_obs_temp, num = 20, cost_fun = cost_fun, compute_sol = compute_sol):
    x_min = -6.0
    x_max = 6.0

    y_min = -6.0
    y_max = 6.0

    t_fin = 8.0
    num = 50
    
    (x_init, y_init, vx_init, ax_init, vy_init, ay_init, x_fin, y_fin, vx_fin, ax_fin, vy_fin, ay_fin) = params
    
    tot_time = np.linspace(0.0, t_fin, num)
    tot_time_copy = tot_time.reshape(num, 1)
    P, Pdot, Pddot = bernstein_coeff_order10_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)
    nvar = np.shape(P)[1]
    num = np.shape(P)[0]

    num_obs = np.shape(x_obs_temp)[0]

    a_obs = 1.0
    b_obs = 1.0

    x_obs = np.ones((num_obs, num)) * x_obs_temp[:, np.newaxis]
    y_obs = np.ones((num_obs, num)) * y_obs_temp[:, np.newaxis]   
    
    rho_obs = 0.3
    rho_eq = 10.0
    weight_smoothness = 10    
    
    bx_eq =  np.hstack((x_init, vx_init, ax_init, x_fin, vx_fin, ax_fin))
    by_eq =  np.hstack((y_init, vy_init, ay_init, y_fin, vy_fin, ay_fin))
    
    A_eq = np.vstack((P[0], Pdot[0], Pddot[0], P[-1], Pdot[-1], Pddot[-1]))
    A_obs = np.tile(P, (num_obs, 1))
    Q_smoothness = np.dot(Pddot.T, Pddot)
    Q_smoothness_new = Q_smoothness
    
    P_jax = jnp.asarray(P)
    A_eq_jax = jnp.asarray(A_eq)
    A_obs_jax = jnp.asarray(A_obs)
    x_obs_jax = jnp.asarray(x_obs)
    y_obs_jax = jnp.asarray(y_obs)
    Q_smoothness_jax = jnp.asarray(Q_smoothness)
    
    x, y, sol_x, sol_y, alpha_obs, d_obs, lamda_x, lamda_y, slack_obs = compute_sol(rho_obs, rho_eq, weight_smoothness, num_obs, bx_eq, by_eq, P, Pdot, Pddot, x_obs, y_obs, a_obs, b_obs)
    
    return x,y


# In[15]:


import random
arr = []
obs = []
for i in range(100):
    x_init = random.randrange(0,5)
    y_init = random.randrange(0,5)
    vx_init = 0.0
    ax_init = 0.0
    vy_init = 0.0
    ay_init = 0.0    

    x_fin = random.randrange(0,5)
    y_fin = random.randrange(0,5)
    vx_fin = 0.0
    ax_fin = 0.0
    vy_fin = 0.0
    ay_fin = 0.0
    
    params = (x_init, y_init, vx_init, ax_init, vy_init, ay_init, x_fin, y_fin, vx_fin, ax_fin, vy_fin, ay_fin)

    x_obs_temp = np.random.rand(1,4)
    y_obs_temp = np.random.rand(1,4)

    x_obs_temp = np.hstack((-10.0, 10.79, 30.0, 10.0))
    y_obs_temp = np.hstack((-10.0, 10.0, -30.80, 10.0))
    x,y = generate(params, x_obs_temp, y_obs_temp, compute_sol = compute_sol) 
#     arr.append([x,y])
    obs.append([x_obs_temp, y_obs_temp])
    barr = np.array([x,y]).T
    arr.append(barr)
    
    plot({"x":x, "y":y}, {"x":x_obs_temp, "y":y_obs_temp}, cnt = i, draw = False)


# In[17]:


np.save("./train/test.npy", arr)
np.save("./train/obstacle.npy", obs)
np.array(arr).shape


# In[1]:


# import random
# arr = []
# obs = []
# for i in range(100):
    
#     prob = OptimizerLane()
    
#     x_init = random.randrange(0,5)
#     y_init = random.randrange(0,5)
#     vx_init = 0.0
#     ax_init = 0.0
#     vy_init = 0.0
#     ay_init = 0.0    

#     x_fin = random.randrange(0,5)
#     y_fin = random.randrange(0,5)
#     vx_fin = 0.0
#     ax_fin = 0.0
#     vy_fin = 0.0
#     ay_fin = 0.0
    
#     vdot_init = 0.0
#     vdot_fin = 0.0


#     v_init = vx_init
#     v_fin =  vx_fin
    
    
#     psi_init = 0.0
#     psidot_init = 0.0
#     psiddot_init = 0.0    
    
#     psi_fin = -random.randrange(-90,90)*np.pi/180
#     psidot_fin = 0.0
#     psiddot_fin = 0.0

#     xsum = x_init + x_fin
#     ysum = y_init + y_fin
#     x_mid = np.hstack(( xsum * 0.25, xsum * 0.5, xsum * 0.75))
#     y_mid = np.hstack(( ysum * 0.25, ysum * 0.5, ysum * 0.75))    
    
#     params = (x_init, y_init, vx_init, ax_init, vy_init, ay_init, x_fin, y_fin, vx_fin, ax_fin, vy_fin, ay_fin)

#     x_obs_temp = np.random.rand(1,4)
#     y_obs_temp = np.random.rand(1,4)

#     x_obs_temp = np.hstack((-2.0, 0.79, 3.0, 4.0))
#     y_obs_temp = np.hstack((-2.0, 5.0, -0.80, 2.0))
    
#     # ############# NEW OPTIMIZER
#     primal_sol, dual_sol, res_eq_psi, res_w, c_v = prob.solve(x_init, y_init, x_fin, y_fin, v_init, v_fin, psi_init, psidot_init, psi_fin, psidot_fin, x_mid, y_mid)

#     rho_params = jnp.hstack(( prob.rho_goal, prob.rho_eq, prob.rho_mid         ))

#     boundary_params = jnp.hstack(( x_init, y_init, x_fin, y_fin, v_init, v_fin, psi_init, psidot_init, psi_fin, psidot_fin ))

#     boundary_init = jnp.hstack(( x_init, y_init, v_init, psi_init, psidot_init  ))
#     variable_params = jnp.hstack(( x_fin, y_fin, psi_fin, psidot_fin, x_mid, y_mid       ))

#     primal_sol_jax = jnp.asarray(primal_sol)

#     cost = objective(prob.P_jax, prob.Pdot_jax, prob.nvar, prob.mid_idx, prob.t, rho_params, boundary_init, variable_params, primal_sol_jax)

#     grad_cost = grad(objective, argnums = (8) )

#     hess_inp = jacfwd(jacrev(objective, argnums = (8)), argnums = (8)  )
#     hess_param = jacfwd(jacrev(objective, argnums = (8)), argnums=7)

#     F_yy = hess_inp(prob.P_jax, prob.Pdot_jax, prob.nvar, prob.mid_idx, prob.t, rho_params, boundary_init, variable_params, primal_sol_jax)
#     F_xy = hess_param(prob.P_jax, prob.Pdot_jax, prob.nvar, prob.mid_idx, prob.t, rho_params, boundary_init, variable_params, primal_sol_jax)

#     F_yy_inv = jnp.linalg.inv(F_yy)
#     dgx = jnp.dot(-F_yy_inv, F_xy)


#     delt = 4.5
#     variable_params_new = jnp.hstack(( x_fin+delt, y_fin, psi_fin, psidot_fin, x_mid, y_mid       ))
#     cur_diff = variable_params_new-variable_params
#     dgx_prod = jnp.matmul(dgx, cur_diff)
#     new_sol = primal_sol_jax+dgx_prod    
    
#     c_psi = primal_sol[0:prob.nvar]
#     c_v = primal_sol[prob.nvar:2*prob.nvar]

#     v = np.dot(prob.P, c_v )
#     psi = np.dot(prob.P, c_psi)

#     x_temp = x_init+np.cumsum(v*np.cos(psi)*prob.t)
#     y_temp = y_init+np.cumsum(v*np.sin(psi)*prob.t)

#     x = np.hstack(( x_init, x_temp[0:-1]    ))
#     y = np.hstack(( y_init, y_temp[0:-1]    ))
#     th = np.linspace(0, 2*np.pi, 100)
    
#     c_psi = primal_sol[0:prob.nvar]
#     c_v = primal_sol[prob.nvar:2*prob.nvar]



#     v = np.dot(prob.P, c_v )
#     psi = np.dot(prob.P, c_psi)

#     x_temp = x_init+np.cumsum(v*np.cos(psi)*prob.t)
#     y_temp = y_init+np.cumsum(v*np.sin(psi)*prob.t)

#     x = np.hstack(( x_init, x_temp[0:-1]    ))
#     y = np.hstack(( y_init, y_temp[0:-1]    ))
    
    
#     # #############
#     print(i)
    
# #     x,y = generate(params, x_obs_temp, y_obs_temp, compute_sol = compute_sol_qp)
#     arr.append([x,y])
#     obs.append([x_obs_temp, y_obs_temp])
#     barr = np.array([x,y]).T
#     arr.append(barr.T)
    
#     plot({"x":x, "y":y}, {"x":x_obs_temp, "y":y_obs_temp}, cnt = i, draw = False, x_mid= x_mid, y_mid = y_mid)   


# In[28]:


print(np.array(arr).shape)


# In[21]:


print(np.array(arr[0]).shape)


# In[22]:


print(np.array(obs).shape)


# In[ ]:


x_obs_temp = np.hstack((-2.0, -0.79, 3.0, 4.0))
y_obs_temp = np.hstack((-2.0, 1.0, -0.80, 2.0))
print(x_obs_temp.shape)

