# Code containing all the routines for L1-HQP

import sys, os
from tasho import task_prototype_rockit as tp
from tasho import input_resolution
from tasho import robot as rob
import casadi as cs
from casadi import pi, cos, sin
from rockit import MultipleShooting, Ocp
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import matplotlib.pyplot as plt
# import tkinter
from pylab import *
import copy
import hqp as hqp_p
import json
import  unittest

class lexLP_duality_trick:
	"""
	Class contains all the functions for implemnting L1-HQP
	"""
	def __init__(self, ts = 0.005):

		self.variables_dot = [] #derivative of the variables
		self.slacks = {} #Slacks for the constraints
		self.slack_weights = {} #the weights for the slack variables
		self.constraints = {}
		self.constraints_numbers = {} #stores the correspondence between constraint id in opti.g
		#and the constraints of a particular priority.
		self.constraint_type = {}
		self.constraint_options_lb = {}
		self.constraint_options_ub = {}
		self.variables0 = [] #Expression for the current value of the variables
		self.cascadedHQP_active = False
		self.opti = cs.Opti() #creating CasADi's optistack
		self.obj = 0 #the variable to be minimized.
		self.constraint_counter = 0
		self.cHQP_optis = {}
		self.cHQP_slacks = {}

	def activate_cascadedHQP(priority_levels):
		""" Should be activated before creating any variables or adding any constraints"""
		cHQP = []
		for i in range(priority_levels):
			cHQP.append(cs.Opti())

		self.cascadedHQP_active = True



	def create_variable(self, size, weight = 0):

		opti = self.opti
		var0 = opti.parameter(size, 1)
		var_dot = opti.variable(size, 1)
		self.variables0 = cs.vertcat(self.variables0, var0)
		self.variables_dot = cs.vertcat(self.variables_dot, var_dot)
		#Adding to the objective the regularization cost
		self.obj += weight*cs.sumsqr(var_dot)

		return var0, var_dot

	def set_value(self, var, val):
		self.opti.set_value(var, val)

	def create_constraint(self, expression, ctype, priority = 0, options = {}):

		opti = self.opti
		shape = expression.shape[0]
		if priority >= 1:
			if priority not in self.slacks:
				self.slacks[priority] = []
				self.slack_weights[priority] = []
				self.constraints[priority] = []
				self.constraint_type[priority] = []
				self.constraint_options_lb[priority]= []
				self.constraint_options_ub[priority] = []
				self.constraints_numbers[priority] = []

			
			slack_var = opti.variable(shape, 1)
			# self.obj += cs.sumsqr(slack_var)*1e-6*0 #regularization

			if ctype == 'ub':
				opti.subject_to(expression <= slack_var + options['ub'])
				opti.subject_to(slack_var >= 0)
				for i in range(shape):
					self.constraints_numbers[priority].append([self.constraint_counter, self.constraint_counter + 1])
					self.constraint_counter += 2

			

			self.slacks[priority] = cs.vertcat(self.slacks[priority], slack_var)
			self.constraints[priority] = cs.vertcat(self.constraints[priority], expression)
			self.constraint_type[priority] = self.constraint_type[priority] +  [ctype]*expression.shape[0]


			if 'ub' in options:
				if options['ub'].shape[0] != 1:
					self.constraint_options_ub[priority] = cs.vertcat(self.constraint_options_ub[priority], options['ub'])
				else:
					self.constraint_options_ub[priority] = cs.vertcat(self.constraint_options_ub[priority], options['ub']*expression.shape[0])

		elif priority == 0:
			if priority not in self.constraints:
				self.constraints[priority] = []
				self.constraints_numbers[priority] = []
				self.constraint_type[priority] = []
				self.constraint_options_lb[priority]= []
				self.constraint_options_ub[priority] = []
			self.constraint_type[priority] = self.constraint_type[priority] +  [ctype]*expression.shape[0]


			if ctype == 'ub':
				opti.subject_to(expression <= options['ub'])
				for i in range(shape):
					self.constraints_numbers[priority].append([self.constraint_counter])
					self.constraint_counter += 1


			if 'ub' in options:
				if options['ub'].shape[0] != 1:
					self.constraint_options_ub[priority] = cs.vertcat(self.constraint_options_ub[priority], options['ub'])
				else:
					self.constraint_options_ub[priority] = cs.vertcat(self.constraint_options_ub[priority], options['ub']*expression.shape[0])

			self.constraints[priority] = cs.vertcat(self.constraints[priority], expression)


	def configure(self):

		#Create functions to compute the constraint expressions
		opti = self.opti
		self.number_priorities = len(self.slacks)
		self.constraint_funs = {}
		self.constraint_expression_funs = {}
		constraint_jacs = {}

		# print("Number of priority levels")
		# print(self.slacks)

		for i in range(0, self.number_priorities + 1):
			self.constraint_funs[i] = cs.Function('cons' + str(i), [self.variables0, self.variables_dot], [cs.jacobian(self.constraints[i], self.variables_dot)])
			constraint_jacs[i] = cs.jacobian(self.constraints[i], self.variables_dot)
			self.constraint_expression_funs[i] = cs.Function('cons' + str(i), [self.variables0, self.variables_dot], [self.constraints[i]])

		elaborate_dual = False
		if elaborate_dual:
			nu = {}
			for i in range(2, self.number_priorities):
				for j in range(1, i):
					nu[(i, j)] = opti.variable()
					opti.subject_to(nu[(i,j)] >= 0)
			p_star = {}
			for i in range(1, self.number_priorities):
				p_star[i] = opti.variable()
				opti.subject_to(cs.DM.ones(self.slacks[i].shape[0], 1).T@self.slacks[i] <= p_star[i])


		#create the dual vectors
		lams = {}
		for i in range(1, self.number_priorities):
			for j in range(0, i+ 1):

				lams[(i, j)] = opti.variable(constraint_jacs[j].shape[0], 1)
				opti.subject_to(lams[(i, j)] >= 0)

			if elaborate_dual & i >= 2:
				for j in range(1, i):
					opti.subject_to(lams[(i, j)] >= nu[(i,j)])

			#Adding the dual feasible constraints
			opti.subject_to(lams[(i, j)] <= 1)

			constraint_acc = 0
			for j in range(0, i+1):
				constraint_acc += constraint_jacs[j].T@lams[(i, j)]
			opti.subject_to(constraint_acc == 0)

			#Adding duality trick constraints
			constraint_acc = 0
			for j in range(1, i):
				if not elaborate_dual:
					constraint_acc += -lams[(i, j)].T@self.slacks[j]
				else:
					constraint_acc += -nu[i,j]*p_star[j]
			for j in range(0, i+1):
				constraint_acc += -lams[(i, j)].T@self.constraint_options_ub[j]
			opti.subject_to(cs.DM.ones(self.slacks[i].shape[0], 1).T@self.slacks[i] <= constraint_acc)



		#Taking the L1 norm penalty of the last priority level as the objective function
		obj = cs.DM.ones(self.slacks[self.number_priorities].shape[0], 1).T@self.slacks[self.number_priorities]
		self.obj += obj
		self.opti.minimize(self.obj)

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

 	
if __name__ == '__main__':

	#A set of small examples with known results to test the lexLP results
	hqp = lexLP_duality_trick()

	x0, x = hqp.create_variable(1)
	hqp.create_constraint(x, 'ub', 1, {'ub':cs.DM([1,])})
	hqp.create_constraint(-x, 'ub', 2, {'ub':cs.DM([-3,])})
	hqp.create_constraint(x, 'ub', 3, {'ub':cs.DM([0.5,])})
	hqp.create_constraint(x, 'ub', 0, {'ub':cs.DM([100,])})

	hqp.configure()

	hqp.opti.solver("ipopt")
	# qpsol_options = {'error_on_fail':False}
	# hqp.opti.solver("sqpmethod", {"expand":True, "qpsol": 'qpoases', 'qpsol_options': qpsol_options,  'print_iteration': True, 'print_header': True, 'print_status': True, "print_time":True, 'max_iter': 1000})
	sol = hqp.opti.solve()

	print(sol.value(x))
	assert cs.fabs(sol.value(x) - 1) <= 1e-5
	assert cs.fabs(sol.value(hqp.slacks[1]) - 0) <= 1e-5
	assert cs.fabs(sol.value(hqp.slacks[2]) - 2) <= 1e-5
	assert cs.fabs(sol.value(hqp.slacks[3]) - 0.5) <= 1e-5
	

	#Test two on the counter example in the paper
	hqp = lexLP_duality_trick()
	x0, x = hqp.create_variable(2, 1e-6)
	hqp.create_constraint(x[1], 'ub', 0, {'ub':cs.DM([0,])})
	hqp.create_constraint(-x[1], 'ub', 0, {'ub':cs.DM([0,])})
	hqp.create_constraint(x[0] - x[1], 'ub', 1, {'ub':cs.DM([0,])})
	hqp.create_constraint(x[1] - x[0], 'ub', 1, {'ub':cs.DM([0+1e-6,])})
	hqp.create_constraint(-x[0], 'ub', 2, {'ub':cs.DM([-1,])})
	hqp.create_constraint(x[0], 'ub', 2, {'ub':cs.DM([1,])})

	hqp.configure()
	# hqp.opti.solver("ipopt")
	qpsol_options = {'error_on_fail':False}
	hqp.opti.solver("sqpmethod", {"expand":True, "qpsol": 'qpoases', 'qpsol_options': qpsol_options,  'print_iteration': True, 'print_header': True, 'print_status': True, "print_time":True, 'max_iter': 1000})
	sol = hqp.opti.solve()

	assert cs.fabs(sol.value(x[0]) - 0) <= 1e-6
	assert cs.fabs(sol.value(x[1]) - 0) <= 1e-5
	assert cs.fabs(sol.value(hqp.slacks[2][0]) - 1) <= 1e-5
	assert cs.fabs(sol.value(hqp.slacks[2][1]) - 0) <= 1e-5



	#Benchmarking the duality-trick with sequential
	run_data = {}
	run_data['total_ineq_constraints'] = 30
	run_data['ineq_con_rank'] = 10
	run_data['n'] = 20

	#Now comparing LexLP with the sequential method for a significantly larger problem
	failure = 0
	for pl in [2,3,4,5,6,7,8,9,10]:
		run_data[pl] = {}
		run_data[pl]['fails'] = 0
		run_data[pl]['counter'] = 0
		run_data[pl]['seq_time_array'] = []
		run_data[pl]['dt_time_array'] = []
		run_data[pl]['hqp-l1_ta'] = []
		for rand_seed in range(300):
			n = run_data['n']
			ineq_con_rank = run_data['ineq_con_rank']
			total_ineq_constraints = run_data['total_ineq_constraints']

			total_ineq_constraints = 30
			ineq_con_rank = 10
			# pl = 2
			con_per_level = int(total_ineq_constraints/pl)
			n_ineq_per_level = int(total_ineq_constraints / pl)
			ineq_last_level = n_ineq_per_level + (total_ineq_constraints - n_ineq_per_level*pl)		
			print("Using random seed " + str(rand_seed))
			np.random.seed(rand_seed) 		
			A_ineq_all = cs.DM(np.random.randn(ineq_con_rank, n))
			A_ineq_extra = (A_ineq_all.T@cs.DM(np.random.randn(ineq_con_rank, total_ineq_constraints - ineq_con_rank))).T
			A_ineq_all = cs.vertcat(A_ineq_all, A_ineq_extra)
			A_ineq_all = A_ineq_all.full()
			np.random.shuffle(A_ineq_all)
			b_ineq_all = np.random.randn(total_ineq_constraints)
			# print(b_ineq_all)
			# print(b_ineq_all_lower)
			#normalizing the each row vector
			row_ineqvec_norm = []
			for i in range(A_ineq_all.shape[0]):
				row_ineqvec_norm.append(cs.norm_1(A_ineq_all[i,:])) 
				b_ineq_all[i] /= row_ineqvec_norm[i]
				for j in range(A_ineq_all.shape[1]):
					A_ineq_all[i,j] = A_ineq_all[i,j]/row_ineqvec_norm[i]	
			A_ineq = {}
			b_upper = {}
			hqp = hqp_p.hqp()
			hqp_dt = lexLP_duality_trick()	
			x, x_dot = hqp.create_variable(n, 1e-6)
			x_dt, x_dot_dt = hqp_dt.create_variable(n, 1e-6)
			params = x
			params_init = [0]*n
			#create these tasks
			counter_ineq = 0
			for i in range(pl):
		
				if i != pl-1:
		
					A_ineq[i] = A_ineq_all[counter_ineq:counter_ineq + n_ineq_per_level, :]
					b_upper[i] = b_ineq_all[counter_ineq:counter_ineq + n_ineq_per_level]
					counter_ineq += n_ineq_per_level
					hqp.create_constraint(A_ineq[i]@x_dot, 'ub', priority = i, options = {'ub':b_upper[i]})
					# hqp.create_constraint(-A_ineq[i]@x_dot, 'ub', priority = i, options = {'ub':-b_upper[i] + 1})
					hqp_dt.create_constraint(A_ineq[i]@x_dot_dt, 'ub', priority = i, options = {'ub':b_upper[i]})
					# hqp_dt.create_constraint(-A_ineq[i]@x_dot_dt, 'ub', priority = i, options = {'ub':-b_upper[i] + 1})
	
				else:	
		
					A_ineq[i] = A_ineq_all[counter_ineq:counter_ineq + ineq_last_level, :]
					b_upper[i] = b_ineq_all[counter_ineq:counter_ineq + ineq_last_level]
					counter_ineq += ineq_last_level
					hqp.create_constraint(A_ineq[i]@x_dot, 'ub', priority = i, options = {'ub':b_upper[i]})
					hqp_dt.create_constraint(A_ineq[i]@x_dot_dt, 'ub', priority = i, options = {'ub':b_upper[i]})
			# qpsol_options = {'error_on_fail':False }
			# hqp.opti.solver("sqpmethod", {"expand":True, "qpsol": 'qpoases', 'qpsol_options': qpsol_options, 'print_iteration': False, 'print_header': False, 'print_status': False, "print_time":True, 'max_iter': 1000})
			hqp.opti.solver("ipopt", {"expand":True, 'ipopt':{'tol':1e-6, 'print_level':0}})
			hqp.variables0 = params
			hqp.configure()
			sol_chqp = hqp.solve_cascadedQP5(params_init, [0]*n, solver = 'ipopt')
			if sol_chqp == False:
				continue
			chqp_time = hqp.time_taken
			hqp.time_taken = 0
			sol_ahqp2 = hqp.solve_HQPl1(params_init, [0]*n, gamma_init = 10.0)
			hqp_dt.configure()
			hqp_dt.opti.solver("ipopt",{"expand":True, 'ipopt':{'tol':1e-6, 'print_level':0}})
			# qpsol_options = {'error_on_fail':False}
			# hqp_dt.opti.solver("sqpmethod", {"expand":True, "qpsol": 'qpoases', 'qpsol_options': qpsol_options,  'print_iteration': True, 'print_header': True, 'print_status': True, "print_time":True, 'max_iter': 10})
			blockPrint()
			try:
				sol = hqp_dt.opti.solve()
			except:
				continue
			enablePrint()

			run_data[pl]['counter'] += 1
			run_data[pl]['seq_time_array'].append(chqp_time)
			run_data[pl]['dt_time_array'].append(sol.stats()['t_wall_total'])
			run_data[pl]['hqp-l1_ta'].append(hqp.time_taken)
			# print(sol.stats())
			print("Time taken by chqp = " + str(chqp_time))
			print("Time taken by hqp-l1 = " + str(hqp.time_taken))
			print("Time taken by dt = " + str(sol.stats()['t_wall_total']))
			# print("Error norm seq and dt = " + str(cs.norm_1(sol_chqp[pl-1].value(hqp.cHQP_xdot[pl-1]) - sol.value(x_dot_dt))))
			# print("Error norm seq and ahqp2 = " + str(cs.norm_1(sol_chqp[pl-1].value(hqp.cHQP_xdot[pl-1]) - sol_ahqp2.value(x_dot))))
			for i in range(1, pl):
				print("violation at level = " + str(i))
				print("By sequential = " + str(cs.DM.ones(1, hqp.slacks[i].shape[0])@sol_chqp[i].value(hqp.cHQP_slacks[i])))
				print("By hqp-l1 = " + str(cs.DM.ones(1, hqp.slacks[i].shape[0])@sol_ahqp2.value(hqp.slacks[i])))
				print("By duality trick = " + str(cs.DM.ones(1, hqp.slacks[i].shape[0])@sol.value(hqp_dt.slacks[i])))
				if cs.DM.ones(1, hqp.slacks[i].shape[0])@sol.value(hqp_dt.slacks[i]) <= cs.DM.ones(1, hqp.slacks[i].shape[0])@sol_chqp[i].value(hqp.cHQP_slacks[i]) -1e-3:
					print("DT gives a better lex optimal solution!!!!!!!!!")
				if not cs.DM.ones(1, hqp.slacks[i].shape[0])@sol.value(hqp_dt.slacks[i]) <= cs.DM.ones(1, hqp.slacks[i].shape[0])@sol_chqp[i].value(hqp.cHQP_slacks[i]) + 1e-3:
					failure += 1
					run_data[pl]['fails'] += 1
					print("Failure!!!!!!!!!!!!!")
					break
		print("Number of failures  "  + str(failure))

	print(run_data)
	with open('../hqp_l1/dt_vs_chqp_benchmark_ipopt5.txt', 'w') as fp:
		json.dump(run_data, fp)
			# assert cs.norm_1(sol_chqp[pl-1].value(hqp.cHQP_xdot[pl-1]) - sol_ahqp2.value(x_dot)) <= 1e-6
			# print(sol.value(x_dot_dt))
			# print(sol_ahqp2.value(x_dot))
	
		


