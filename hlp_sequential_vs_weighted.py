#code to test the correctness of hierarchy and to visualize the effect of gamma
# The testing would be done on random matrices that represent the task function
# Jacobians

import sys, os
# sys.path.insert(0, "/home/ajay/Desktop/hqp_l1")
from lexopt_mosek import  lexopt_mosek
import casadi as cs
from casadi import pi, cos, sin
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import json
import pickle
import copy

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

#making this a function: dictionary input takes n, total_eq_constraints, total_ineq_constraints, pl (priority levels)
#random seed, gamma value
#gets as output whether cHQP ran, HQP-L1 ran, hierarchy failure, same constraints, identical solution, geq_constraints

def hqpvschqp(params):
	result = {}
	n = params['n']
	total_eq_constraints = params['total_eq_constraints']
	total_ineq_constraints = params['total_ineq_constraints']
	pl = params['pl'] #priority levels
	gamma = params['gamma']
	rand_seed = params['rand_seed']
	adaptive_method = params['adaptive_method']#True

	n_eq_per_level = int(total_eq_constraints / pl)
	eq_last_level = n_eq_per_level + (total_eq_constraints - n_eq_per_level*pl)
	n_ineq_per_level = int(total_ineq_constraints / pl)
	ineq_last_level = n_ineq_per_level + (total_ineq_constraints - n_ineq_per_level*pl)
	# print(n_eq_per_level)
	# print(n_ineq_per_level)
	# print(eq_first_level)
	# print(ineq_first_level)
	count_hierarchy_failue = 0
	count_same_constraints = 0
	count_identical_solution = 0
	count_geq_constraints = 0

	hlp = lexopt_mosek()
	print("Using random seed " + str(rand_seed))
	# n_eq_per_level = 6
	# n_ineq_per_level = 6
	np.random.seed(rand_seed) #for 45, 1, 8 HQP-l1 does not agree with the cHQP
	#15, 18, 11 for 8 priority levels

	x, x_dot = hlp.create_variable(n, [-1e6]*n, [1e6]*n) #high enough bounds to not matter
	A_eq_all = cs.DM(np.random.randn(params['eq_con_rank'], n))
	A_extra = (A_eq_all.T@cs.DM(np.random.randn(params['eq_con_rank'], total_eq_constraints - params['eq_con_rank']))).T
	A_eq_all = cs.vertcat(A_eq_all, A_extra)
	A_eq_all = A_eq_all.full()
	np.random.shuffle(A_eq_all)
	A_eq_all += np.random.randn(total_eq_constraints, n)*1e-5
	b_eq_all = np.random.randn(total_eq_constraints)
	#normalizing the each row vector
	row_vec_norm = []
	for i in range(A_eq_all.shape[0]):
		row_vec_norm.append(cs.norm_1(A_eq_all[i,:]))
		# print(row_vec_norm)
		b_eq_all[i] /= row_vec_norm[i]
		for j in range(A_eq_all.shape[1]):
			A_eq_all[i,j] = A_eq_all[i,j].T/row_vec_norm[i]



	A_ineq_all = cs.DM(np.random.randn(params['ineq_con_rank'], n))
	A_ineq_extra = (A_ineq_all.T@cs.DM(np.random.randn(params['ineq_con_rank'], total_ineq_constraints - params['ineq_con_rank']))).T
	A_ineq_all = cs.vertcat(A_ineq_all, A_ineq_extra)
	A_ineq_all = A_ineq_all.full()
	np.random.shuffle(A_ineq_all)
	b_ineq_all = np.random.randn(total_ineq_constraints)
	b_ineq_all_lower = b_ineq_all - 1
	# print(b_ineq_all)
	# print(b_ineq_all_lower)
	#normalizing the each row vector
	row_ineqvec_norm = []
	for i in range(A_ineq_all.shape[0]):
		row_ineqvec_norm.append(cs.norm_1(A_ineq_all[i,:]))
		b_ineq_all[i] /= row_ineqvec_norm[i]
		b_ineq_all_lower[i] /= row_ineqvec_norm[i]
		for j in range(A_ineq_all.shape[1]):
			A_ineq_all[i,j] = A_ineq_all[i,j]/row_ineqvec_norm[i]



	A_eq = {}
	b_eq = {}
	A_eq_opti = {}
	b_eq_opti = {}
	A_ineq_opti = {}
	b_upper_opti = {}

	A_ineq = {}
	b_upper = {}
	b_lower = {}

	params = x
	params_init = [0]*n
	#create these tasks
	counter_eq = 0
	counter_ineq = 0
	# print(row_ineqvec_norm)
	for i in range(pl):

		if i != pl-1:
			A_eq[i] = A_eq_all[counter_eq:counter_eq + n_eq_per_level, :]
			b_eq[i] = b_eq_all[counter_eq:counter_eq + n_eq_per_level]
			counter_eq += n_eq_per_level
			hlp.create_constraint(cs.mtimes(A_eq[i],x_dot) - b_eq[i], 'eq', i, {'b':0})


			A_ineq[i] = A_ineq_all[counter_ineq:counter_ineq + n_ineq_per_level, :]
			b_upper[i] = b_ineq_all[counter_ineq:counter_ineq + n_ineq_per_level]
			b_lower[i] = b_ineq_all_lower[counter_ineq:counter_ineq + n_ineq_per_level]
			counter_ineq += n_ineq_per_level
			hlp.create_constraint(A_ineq[i]@x_dot, 'lub', i, {'lb':b_lower[i], 'ub':b_upper[i]})

		else:
			A_eq[i] = A_eq_all[counter_ineq:counter_ineq + eq_last_level, :]
			b_eq[i] = b_eq_all[counter_eq:counter_eq + eq_last_level]
			counter_eq += eq_last_level
			hlp.create_constraint(cs.mtimes(A_eq[i],x_dot) - b_eq[i], 'eq', i, {'b':0})


			A_ineq[i] = A_ineq_all[counter_ineq:counter_ineq + ineq_last_level, :]
			b_upper[i] = b_ineq_all[counter_ineq:counter_ineq + ineq_last_level]
			b_lower[i] = b_ineq_all_lower[counter_ineq:counter_ineq + ineq_last_level]
			counter_ineq += ineq_last_level
			hlp.create_constraint(A_ineq[i]@x_dot, 'lub', i, {'lb':b_lower[i], 'ub':b_upper[i]})

	hlp.configure_constraints()
	hlp.compute_matrices([0]*n, [0]*n)
	hlp.configure_weighted_problem()
	hlp.configure_sequential_problem()


	try:
		hlp.solve_sequential_problem()
		result['slp_time'] = hlp.time_taken
		result['slp_status'] = True
	except:
		result['slp_status'] = False

	if not adaptive_method:
		hlp.time_taken = 0
		sol = hlp.solve_weighted_method([gamma]*(pl-2))
	else:
		# tic = time.time()
		sol, hierarchical_failure = hqp.solve_adaptive_hqp3(params_init, [0]*n, gamma_init = gamma)
		# toc = time.time() - tic
		tp = 0 #true positive
		fp = 0
		tn = 0
		fn = 0
	hlp_status = True
	result['hlp_status'] = hlp_status
	if not hlp_status:
		print("hlp-l1 failed")
	else:
		result['hlp_time'] = hlp.time_taken
		if adaptive_method:
			result['heuristic_prediction'] = len(hierarchical_failure) == 0
		# if not adaptive_method:
		# 	return False, False, False, True, False, False
		# else:
		# 	return False, False, False, True, False, False, False

	#further analysis and comparison only if both hqp and chqp returned without failing

	lex_opt = True
	if hlp_status and result['slp_status']:
		for i in range(1,pl-1):
			weighted_obj = sum(hlp.Mw_dict[i]['eq_slack'].level()) + sum(hlp.Mw_dict[i]['lub_slack'].level())
			sequential_obj = hlp.optimal_slacks[i]
			if weighted_obj > sequential_obj + 1e-5:
				lex_opt = False
				if verbose:
					print("Lex norm unsatisfied!!!!!!!!!!!!!!!!! by " + str(weighted_obj - sequential_obj))


		result['lex_opt'] = lex_opt

	return result

if __name__ == "__main__":

	n = 25
	total_eq_constraints = 25
	eq_con_rank = 15
	total_ineq_constraints = 25
	ineq_con_rank = 15
	params = {}
	params['n'] = n
	params['eq_con_rank'] = eq_con_rank
	params['ineq_con_rank'] = ineq_con_rank
	params['total_eq_constraints'] = total_eq_constraints
	params['total_ineq_constraints'] = total_ineq_constraints
	# gamma_vals = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 7.0, 10.0, 14.0, 15.0, 20.0, 40.0, 60.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0]
	# pl_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10]
	# gamma_vals = [0.1]
	gamma_vals = [500]
	# pl_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10]

	# gamma_vals = [10.0, 20.0, 50.0]
	# pl_vals = [3, 4, 9, 10]
	pl_vals = [3]
	# gamma_vals = [10.0]
	# pl_vals = [20]
	results = {}

	verbose = True
	adaptive_method = False
	params['adaptive_method'] = adaptive_method
	for gamma in gamma_vals:
		print("gamma value :" + str(gamma))
		for pl in pl_vals:
			print("priority levels :" + str(pl))
			count_lex_opt = 0
			total_trials = 0
			slp_time_arr = []
			hlp_time_arr = []
			params['pl'] = pl
			params['gamma'] = gamma
			for rand_seed in range(1000, 1300):
				# print(rand_seed)
				params['rand_seed'] = rand_seed

				result = hqpvschqp(params)

				slp_status = result['slp_status']
				hlp_status = result['hlp_status']
				if slp_status and hlp_status:
					total_trials += 1
					slp_time_arr.append(result['slp_time'])
					hlp_time_arr.append(result['hlp_time'])

					if result['lex_opt']:
						count_lex_opt += 1

			if verbose:
				print("Total success " + str(count_lex_opt))
				print('Total trials ' + str(total_trials))


			results[str(pl) + ',' + str(gamma)] = [count_lex_opt, total_trials, slp_time_arr, hlp_time_arr]

	print(results)
	with open('./slp_weighted_hlp.txt', 'w') as fp:
		json.dump(results, fp)
