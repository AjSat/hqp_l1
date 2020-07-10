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

import copy

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__



n = 25
total_eq_constraints = 25
total_ineq_constraints = 25
pl = 2 #priority levels

n_eq_per_level = int(total_eq_constraints / pl)
eq_first_level = n_eq_per_level + (total_eq_constraints - n_eq_per_level*pl)
n_ineq_per_level = int(total_ineq_constraints / pl)
ineq_first_level = n_ineq_per_level + (total_ineq_constraints - n_ineq_per_level*pl)
print(n_eq_per_level)
print(n_ineq_per_level)
print(eq_first_level)
print(ineq_first_level)
count_hierarchy_failue = 0
count_same_constraints = 0
count_identical_solution = 0
count_geq_constraints = 0

gamma_vals = [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 7.0, 10.0, 14.0, 15.0, 20.0, 40.0, 60, 100, 150, 200, 300, 400, 500]

for rand_seed in range(1000,1050):
	
	hqp = hqp_p.hqp()
	print("Using random seed " + str(rand_seed))
	# n_eq_per_level = 6
	# n_ineq_per_level = 6
	np.random.seed(rand_seed) #for 45, 1, 8 HQP-l1 does not agree with the cHQP 
	#15, 18, 11 for 8 priority levels
	
	x, x_dot = hqp.create_variable(n, 1e-6)
	A_eq_all = cs.DM(np.random.randn(total_eq_constraints, n))
	b_eq_all = np.random.randn(total_eq_constraints)
	A_ineq_all = cs.DM(np.random.randn(total_ineq_constraints, n))
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
			hqp.create_constraint(A_ineq[i]@x_dot, 'lub', priority = i, options = {'lb':b_upper[i] - 10, 'ub':b_upper[i]})

		else:
			A_eq[i] = A_eq_all[0:eq_first_level, :]
			b_eq[i] = b_eq_all[0:eq_first_level]
			counter_eq += eq_first_level
			hqp.create_constraint(cs.mtimes(A_eq[i],x_dot) - b_eq[i], 'equality', priority = i)
	

			A_ineq[i] = A_ineq_all[0:ineq_first_level, :]
			b_upper[i] = b_ineq_all[0:ineq_first_level]
			counter_ineq += ineq_first_level
			b_lower[i] = b_upper[i] - 1
			hqp.create_constraint(A_ineq[i]@x_dot, 'lub', priority = i, options = {'lb':b_upper[i] - 10, 'ub':b_upper[i]})

	# print(A_eq)
	# p_opts = {"expand":True}
	# s_opts = {"max_iter": 100, 'tol':1e-8}#, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-6}
	# hqp.opti.solver('ipopt', p_opts, s_opts)
	hqp.opti.solver("sqpmethod", {"expand":True, "qpsol": 'qpoases', 'print_iteration': False, 'print_header': True, 'print_status': False, "print_time":False, 'max_iter': 1000})

	blockPrint()
	hqp.variables0 = params
	hqp.configure()

	# hqp.setup_cascadedQP()
	sol_chqp = hqp.solve_cascadedQP(params_init, [0]*n)
	sol = hqp.solve_HQPl1(params_init, [0]*n, gamma_init = 200.0)
	enablePrint()

	x_dot_sol = sol.value(x_dot)
	# print(x_dot_sol)
	con_viol2 = []
	x_dot_sol2 = sol_chqp[pl - 1].value(x_dot)
	# print(x_dot_sol2)
	con_viol = []

	geq_constraints_satisfied = True
	same_constraints_satisfied = True

	verbose = True
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
		if sum(satisfied_con_hqp) < sum(satisfied_con_chqp):
			geq_constraints_satisfied = False

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





	print(con_viol)
	print(con_viol2)
	print("diff between solutions = " + str(max(cs.fabs(x_dot_sol - x_dot_sol2).full())))

	if max(cs.fabs(x_dot_sol - x_dot_sol2).full()) <= 1e-4:
		print("Identical solution by both methods!!")
		count_identical_solution += 1
		count_geq_constraints += 1
		count_same_constraints += 1
	
	elif same_constraints_satisfied:
		print("same constraints are satisfied")
		count_same_constraints += 1
		count_geq_constraints += 1
	
	elif geq_constraints_satisfied:
		print("The same of greater number of constriants satisfied at each level")
		count_geq_constraints += 1

	else:
		print("hierarchy failed!!!!!!!!!!!!!!!!")
		count_hierarchy_failue += 1

print("hierarchy failed " + str(count_hierarchy_failue))
print("Identical solution " + str(count_identical_solution))
print("Same constraints satisfied " + str(count_same_constraints))
print("Geq constraints satisfied " + str(count_geq_constraints))

# print(sol_chqp[pl - 1].value(hqp.constraints[1]))
# print(hqp.constraint_options_lb[1])
# print(hqp.constraint_options_ub[1])

