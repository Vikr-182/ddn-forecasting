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
print('v_init =', np.sqrt( vx_init_data[idx, 0]**2+vy_init_data[idx, 0]**2  ) )

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

#print(psi_fin_data[idx, 0], prob.psi[:,-1]-rot_angle)
#print('.................')
#print(psi_init_data[idx, 0], prob.psi[:,0]-rot_angle)


# print(x[:, -1], y[:, -1], x_fin_mod, y_fin_mod)
# print(x_rot[:, -1], y_rot[:, -1], x_fin_data[idx, 0], y_fin_data[idx, 0])
# print(x_rot[:, 0], y_rot[:, 0], x_init_data[idx, 0], y_init_data[idx, 0])



#plt.plot(x_rot.T, y_rot.T, '-r', linewidth = 3.0)
#for i in range(0, prob.num_batch):
#	plt.plot(pos[i, 0, :], pos[i,1, : ], '--k', linewidth = 3.0)

# plt.plot(xy_data[0, -1], xy_data[1, -1], 'og', markersize = 9.0)
# plt.plot(xy_data[0, 0], xy_data[1, 0], 'om', markersize = 9.0)
# plt.plot(xy_data[0, 0], xy_data[1, 0], 'om', markersize = 9.0)


plt.axis('equal')

plt.show()



for i in range(0, prob.num_batch):
    plt.plot(pos[i, 0, :], pos[i,1, : ], '--k', linewidth = 3.0)
    plt.plot(x_rot.T[:, i], y_rot.T[:, i], '-r', linewidth = 3.0)
    plt.axis('equal')
    plt.savefig("results/plot_" + str(i) + ".png")
    plt.clf()
#    plt.show()
print(x_rot.T.shape)


