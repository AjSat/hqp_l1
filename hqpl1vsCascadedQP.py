#code to test the correctness of hierarchy and to visualize the effect of gamma
# The testing would be done on random matrices that represent the task function
# Jacobians

import sys, os
sys.path.insert(0, "/home/ajay/Desktop/hqp_l1")
import  hqp as hqp_p
from tasho import task_prototype_rockit as tp
from tasho import input_resolution
from tasho import robot as rob
import casadi as cs
from casadi import pi, cos, sin
import numpy as np
import matplotlib.pyplot as plt
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
	
	n = params['n']
	total_eq_constraints = params['total_eq_constraints']
	total_ineq_constraints = params['total_ineq_constraints']
	pl = params['pl'] #priority levels
	gamma = params['gamma']
	rand_seed = params['rand_seed']

	n_eq_per_level = int(total_eq_constraints / pl)
	eq_first_level = n_eq_per_level + (total_eq_constraints - n_eq_per_level*pl)
	n_ineq_per_level = int(total_ineq_constraints / pl)
	ineq_first_level = n_ineq_per_level + (total_ineq_constraints - n_ineq_per_level*pl)
	# print(n_eq_per_level)
	# print(n_ineq_per_level)
	# print(eq_first_level)
	# print(ineq_first_level)
	count_hierarchy_failue = 0
	count_same_constraints = 0
	count_identical_solution = 0
	count_geq_constraints = 0
	
	hqp = hqp_p.hqp()
	print("Using random seed " + str(rand_seed))
	# n_eq_per_level = 6
	# n_ineq_per_level = 6
	np.random.seed(rand_seed) #for 45, 1, 8 HQP-l1 does not agree with the cHQP 
	#15, 18, 11 for 8 priority levels
	
	x, x_dot = hqp.create_variable(n, 1e-6)
	A_eq_all = cs.DM(np.random.randn(params['eq_con_rank'], n))
	A_extra = (A_eq_all.T@cs.DM(np.random.randn(params['eq_con_rank'], total_eq_constraints - params['eq_con_rank']))).T
	A_eq_all = cs.vertcat(A_eq_all, A_extra)
	A_eq_all = A_eq_all.full()
	np.random.shuffle(A_eq_all)
	# print(A_eq_all)
	# afsdkl= jkljl
	b_eq_all = np.random.randn(total_eq_constraints)
	A_ineq_all = cs.DM(np.random.randn(params['ineq_con_rank'], n))
	A_ineq_extra = (A_ineq_all.T@cs.DM(np.random.randn(params['ineq_con_rank'], total_ineq_constraints - params['ineq_con_rank']))).T
	A_ineq_all = cs.vertcat(A_ineq_all, A_ineq_extra)
	A_ineq_all = A_ineq_all.full()
	np.random.shuffle(A_ineq_all)
	b_ineq_all = np.random.randn(total_ineq_constraints)

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

	for i in range(pl):

		if i != 0:
			A_eq[i] = A_eq_all[counter_eq:counter_eq + n_eq_per_level, :]
			b_eq[i] = b_eq_all[counter_eq:counter_eq + n_eq_per_level]
			counter_eq += n_eq_per_level
			hqp.create_constraint(cs.mtimes(A_eq[i],x_dot) - b_eq[i], 'equality', priority = i)
	

			A_ineq[i] = A_ineq_all[counter_ineq:counter_ineq + n_ineq_per_level, :]
			b_upper[i] = b_ineq_all[counter_ineq:counter_ineq + n_ineq_per_level]
			counter_ineq += n_ineq_per_level
			b_lower[i] = b_upper[i] - 1
			hqp.create_constraint(A_ineq[i]@x_dot, 'lub', priority = i, options = {'lb':b_upper[i] - 1, 'ub':b_upper[i]})

		else:
			A_eq[i] = A_eq_all[0:eq_first_level, :]
			b_eq[i] = b_eq_all[0:eq_first_level]
			counter_eq += eq_first_level
			hqp.create_constraint(cs.mtimes(A_eq[i],x_dot) - b_eq[i], 'equality', priority = i)
	

			A_ineq[i] = A_ineq_all[0:ineq_first_level, :]
			b_upper[i] = b_ineq_all[0:ineq_first_level]
			counter_ineq += ineq_first_level
			b_lower[i] = b_upper[i] - 1
			hqp.create_constraint(A_ineq[i]@x_dot, 'lub', priority = i, options = {'lb':b_upper[i] - 1, 'ub':b_upper[i]})

	# print(A_eq)
	# p_opts = {"expand":True}
	# s_opts = {"max_iter": 100, 'tol':1e-8}#, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-6}
	# hqp.opti.solver('ipopt', p_opts, s_opts)
	hqp.opti.solver("sqpmethod", {"expand":True, "qpsol": 'qpoases', 'print_iteration': False, 'print_header': True, 'print_status': False, "print_time":True, 'max_iter': 1000})

	blockPrint()
	hqp.variables0 = params
	hqp.configure()

	# hqp.setup_cascadedQP()
	adaptive_method = True
	
	sol_chqp = hqp.solve_cascadedQP2(params_init, [0]*n)
	enablePrint()
	chqp_status = sol_chqp != False
	if not chqp_status:
		print("cHQP failed")
		if not adaptive_method:
			return False, False, False, False, False, False
		else:
			return False, False, False, False, False, False, False

	blockPrint()
	if not adaptive_method:
		hqp.time_taken = 0
		sol = hqp.solve_HQPl1(params_init, [0]*n, gamma_init = gamma)
		enablePrint()
		print("Time taken by the non-adaptive method = " + str(hqp.time_taken))
	else:
		# tic = time.time()
		sol, hierarchical_failure = hqp.solve_adaptive_hqp(params_init, [0]*n, gamma_init = gamma)
		# toc = time.time() - tic
		tp = 0 #true positive
		fp = 0
		tn = 0
		fn = 0
	hqp_status = sol != False
	enablePrint()
	# print("Total time taken adaptive HQP = " + str(toc))
	if not hqp_status:
		print("hqp-l1 failed")
		if not adaptive_method:
			return False, False, False, True, False, False
		else:
			return False, False, False, True, False, False, False



	x_dot_sol = sol.value(x_dot)
	# print(x_dot_sol)
	con_viol2 = []
	x_dot_sol2 = sol_chqp[pl - 1].value(x_dot)
	# print(x_dot_sol2)
	con_viol = []

	geq_constraints_satisfied = True
	same_constraints_satisfied = True
	lex_con_norm = True

	verbose = False
	running_counter_satisfied_con_hqp = 0
	running_counter_satisfied_con_chqp = 0
	for i in range(1,pl):
		
		sol_hqp = sol.value(hqp.slacks[i])
		
		con_viol.append(cs.norm_1(sol_hqp))
		sol_cHQP = sol_chqp[pl-1].value(hqp.slacks[i])
		
		con_viol2.append(cs.norm_1(sol_cHQP))

		#Computing which constraints are satisfied
		satisfied_con_hqp = sol_hqp <= 1e-8
		satisfied_con_chqp = sol_cHQP <= 1e-8

		#computing whether the same constriants are satisfied
		if (satisfied_con_hqp != satisfied_con_chqp).any():
			same_constraints_satisfied = False

		#compute if a geq number of constraints are satisfied by the hqp
		running_counter_satisfied_con_chqp += sum(satisfied_con_chqp)
		running_counter_satisfied_con_hqp += sum(satisfied_con_hqp)
		if running_counter_satisfied_con_hqp < running_counter_satisfied_con_chqp:
			geq_constraints_satisfied = False

		if cs.norm_1(sol_hqp) > cs.norm_1(sol_cHQP) + 1e-4:
			lex_con_norm = False
			if verbose:
				print("Lex norm unsatisfied!!!!!!!!!!!!!!!!!")

		if verbose: #make true if print
			print("Level hqp-l1" + str(i))
			print(sol_hqp)
			print("Level chqp" + str(i))
			print(sol_cHQP)
			print(satisfied_con_hqp)
			print(satisfied_con_chqp)

	if verbose:
		print(x_dot_sol)
		print(x_dot_sol2)




	if verbose:
		print(con_viol)
		print(con_viol2)
		# print("diff between solutions = " + str(max(cs.fabs(x_dot_sol - x_dot_sol2).full())))

	identical_solution = max(cs.fabs(x_dot_sol - x_dot_sol2).full()) <= 1e-4
	if identical_solution:
		# print("Identical solution by both methods!!")
		count_identical_solution += 1
		count_geq_constraints += 1
		count_same_constraints += 1
		if adaptive_method:
			if len(hierarchical_failure) > 0:
				fp += 1
			else:
				tn += 1
	
	elif same_constraints_satisfied:
		# print("same constraints are satisfied")
		count_same_constraints += 1
		count_geq_constraints += 1
		if adaptive_method:
			if len(hierarchical_failure) > 0:
				fp += 1
			else:
				tn += 1
	
	elif geq_constraints_satisfied or lex_con_norm:
		# print("The same of greater number of constriants satisfied at each level")
		count_geq_constraints += 1
		if adaptive_method:
			if len(hierarchical_failure) > 0:
				fp += 1
			else:
				tn += 1

	else:
		# print("hierarchy failed!!!!!!!!!!!!!!!!")
		count_hierarchy_failue += 1
		if adaptive_method:
			if len(hierarchical_failure) > 0:
				tp += 1
			else:
				fn += 1
	if verbose:
		print("hierarchy failed " + str(count_hierarchy_failue))
		print("Identical solution " + str(count_identical_solution))
		print("Same constraints satisfied " + str(count_same_constraints))
		print("Geq constraints satisfied " + str(count_geq_constraints))

	if not adaptive_method:
		return identical_solution[0], same_constraints_satisfied, geq_constraints_satisfied, chqp_status, hqp_status, lex_con_norm
	else:
		return identical_solution[0], same_constraints_satisfied, geq_constraints_satisfied, chqp_status, hqp_status, lex_con_norm, {'tn': tn, 'tp':tp, 'fp':fp, 'fn':fn}
# print(sol_chqp[pl - 1].value(hqp.constraints[1]))
# print(hqp.constraint_options_lb[1])
# print(hqp.constraint_options_ub[1])

if __name__ == "__main__":

	n = 25
	total_eq_constraints = 3
	eq_con_rank = 2
	total_ineq_constraints = 3
	ineq_con_rank = 2
	params = {}
	params['n'] = n
	params['eq_con_rank'] = eq_con_rank
	params['ineq_con_rank'] = ineq_con_rank
	params['total_eq_constraints'] = total_eq_constraints
	params['total_ineq_constraints'] = total_ineq_constraints
	# gamma_vals = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 7.0, 10.0, 14.0, 15.0, 20.0, 40.0, 60.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0]
	# pl_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10]
	# gamma_vals = [0.2,  1.0,  5.0]
	# pl_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

	gamma_vals = [0.1]
	pl_vals = [2]
	results = {}

	verbose = True
	adaptive_method = True

	for gamma in gamma_vals:
		print("gamma value :" + str(gamma))
		for pl in pl_vals:
			print("priority levels :" + str(pl))
			count_hierarchy_failue = 0
			count_same_constraints = 0
			count_identical_solution = 0
			count_geq_constraints = 0
			count_lex_con_norm = 0
			if adaptive_method:
				count_tp = 0
				count_tn = 0
				count_fp = 0
				count_fn = 0
				fp_cases = []
				fn_cases = []

			total_trials = 0
			params['pl'] = pl
			params['gamma'] = gamma
			for rand_seed in range(1000, 1100):
				# print(rand_seed)
				params['rand_seed'] = rand_seed
				
				if not adaptive_method:
					identical_solution, same_constraints_satisfied, geq_constraints_satisfied, chqp_status, hqp_status, lex_con_norm = hqpvschqp(params)
				else:
					identical_solution, same_constraints_satisfied, geq_constraints_satisfied, chqp_status, hqp_status, lex_con_norm, hf_heuristic = hqpvschqp(params)
				
				if chqp_status and hqp_status:
					total_trials += 1
					if lex_con_norm:
						count_lex_con_norm += 1
					if identical_solution:
						count_identical_solution += 1
						count_geq_constraints += 1
						count_same_constraints += 1
					elif same_constraints_satisfied:
						count_same_constraints += 1
						count_geq_constraints += 1
					elif geq_constraints_satisfied:
						count_geq_constraints += 1
					elif not lex_con_norm:

						count_hierarchy_failue += 1

					if adaptive_method and hqp_status:
						count_tp += hf_heuristic['tp']
						count_tn += hf_heuristic['tn']
						count_fp += hf_heuristic['fp']
						count_fn += hf_heuristic['fn']
						if hf_heuristic['fp']:
							fp_cases.append(rand_seed)
						elif hf_heuristic['fn']:
							fn_cases.append(rand_seed)

			if verbose:
				print("hierarchy failed " + str(count_hierarchy_failue))
				print("Identical solution " + str(count_identical_solution))
				print("Same constraints satisfied " + str(count_same_constraints))
				print("Geq constraints satisfied " + str(count_geq_constraints))
				print('Total trials ' + str(total_trials))
				if adaptive_method:
					print("hierarchical failure detection by heuristic:")
					print("FP = " + str(count_fp) + " FN = " + str(count_fn) + " TP = " + str(count_tp) + " TN = " + str(count_tn))
					print("FP cases are " + str(fp_cases))
					print("FN cases are " + str(fn_cases))
			if not adaptive_method:
				results[str(pl) + ',' + str(gamma)] = [count_hierarchy_failue, count_identical_solution, count_same_constraints, count_geq_constraints, total_trials, count_lex_con_norm]
			else:
				hf_heuristic = {'tp':count_tp, 'tn':count_tn, 'fn':count_fn, 'fp':count_fp}
				results[str(pl) + ',' + str(gamma)] = [count_hierarchy_failue, count_identical_solution, count_same_constraints, count_geq_constraints, total_trials, count_lex_con_norm, hf_heuristic]
	print(results)
	with open('../hqp_l1/hqp_vs_chqp_results.txt', 'w') as fp:
		json.dump(results, fp)

