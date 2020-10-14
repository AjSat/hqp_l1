#code to test the hierarchical task specification on random tasks for the correctness
import sys
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

hqp = hqp_p.hqp()

n = 150
pl = 25 #priority levels

n_eq_per_level = 3
n_ineq_per_level = 3
# pl = 3
# n_eq_per_level = 25
# n_ineq_per_level = 25

x, x_dot = hqp.create_variable(n, 1e-3)

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
np.random.seed(451)
for i in range(pl):
	if i <= 100:
		A_eq[i] = cs.DM(np.random.randn(n_eq_per_level, n))
	elif i > 0:
		A_eq[i] = cs.DM(np.random.randn(n_eq_per_level-1, n))
		A_eq[i] = (cs.vertcat(A_eq[i], A_eq[i][0,:] + A_eq[i][1,:]))*0.5
	b_eq[i] = np.random.randn(n_eq_per_level)
	A_eq_opti[i] = hqp.opti.parameter(n_eq_per_level, n)
	b_eq_opti[i] = hqp.opti.parameter(n_eq_per_level, 1)
	params = cs.vertcat(params, cs.vec(A_eq_opti[i]), b_eq_opti[i])
	params_init = cs.vertcat(params_init, cs.vec(A_eq[i]), b_eq[i])
	hqp.create_constraint(cs.mtimes(A_eq_opti[i],x_dot) - b_eq_opti[i], 'equality', priority = i)
	# hqp.create_constraint(cs.mtimes(A_eq[i],x_dot) - b_eq[i], 'equality', priority = i)

	
	print(A_eq_opti[i])
	

	A_ineq[i] = np.random.randn(n_ineq_per_level, n)
	b_upper[i] = np.random.randn(n_ineq_per_level)
	b_lower[i] = b_upper[i] - 100
	A_ineq_opti[i] = hqp.opti.parameter(n_ineq_per_level, n)
	b_upper_opti[i] = hqp.opti.parameter(n_ineq_per_level)
	hqp.create_constraint(A_ineq_opti[i]@x_dot, 'lub', priority = i, options = {'lb':b_upper_opti[i] - 1, 'ub':b_upper_opti[i]})
	# hqp.create_constraint(A_ineq[i]@x_dot, 'lub', priority = i, options = {'lb':b_upper[i] - 10, 'ub':b_upper[i]})
	params = cs.vertcat(params, cs.vec(A_ineq_opti[i]), b_upper_opti[i])
	params_init = cs.vertcat(params_init, cs.vec(A_ineq[i]), b_upper[i])

print(A_eq_opti[i])
# hqp.opti.set_value(A_eq_opti[1], A_eq[1])
# p_opts = {"expand":True}
# s_opts = {"max_iter": 100}#, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-6}
# hqp.opti.solver('ipopt', p_opts, s_opts)
hqp.opti.solver("sqpmethod", {"expand":True, "qpsol": 'qpoases', 'print_iteration': True, 'print_header': True, 'print_status': True, "print_time":True, 'max_iter': 1000})

hqp.variables0 = params
hqp.configure()

# hqp.setup_cascadedQP()
# hqp.solve_cascadedQP(params_init, [0]*n)
sol = hqp.solve_HQPl1(params_init, [0]*n)

x_dot_sol = sol.value(x_dot)
print(x_dot_sol)
con_viol = []
for i in range(1,pl):
	con_viol.append(cs.norm_1(sol.value(hqp.slacks[i])))
print(con_viol)
#using this for warm starting

# hqp.opti.solver("sqpmethod", {"expand":True, "qpsol": 'qpoases', 'print_iteration': True, 'print_header': True, 'print_status': True, "print_time":True, 'max_iter': 1000})
# hqp.opti.subject_to()
pert_fac = 1e-2 #perturbation factor
tic = time.time()
for j in range(10):
	params_init = [0]*n
	for i in range(pl):
		if i <= 100:
			A_eq[i] = A_eq[i] + pert_fac*np.random.randn(n_eq_per_level, n)
		elif i > 0:
			A_eq[i][0:-1,:] = A_eq[i][0:-1,:] + pert_fac*np.random.randn(n_eq_per_level-1, n)
			A_eq[i][-1,:] =  (A_eq[i][0,:] + A_eq[i][1,:])*0.5
		b_eq[i] = b_eq[i] + pert_fac*np.random.randn(n_eq_per_level)
		params_init = cs.vertcat(params_init, cs.vec(A_eq[i]), b_eq[i])
	
		A_ineq[i] = A_ineq[i] + pert_fac*np.random.randn(n_ineq_per_level, n)
		b_upper[i] = b_upper[i] + pert_fac*np.random.randn(n_ineq_per_level)
		params_init = cs.vertcat(params_init, cs.vec(A_ineq[i]), b_upper[i])

# p_opts = {"expand":True}
# s_opts = {"max_iter": 100}#, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-6}
# hqp.opti.solver('ipopt', p_opts, s_opts)
# # hqp.opti.solver("sqpmethod", {"expand":True, "qpsol": 'qpoases', 'print_iteration': True, 'print_header': True, 'print_status': True, "print_time":True, 'max_iter': 1000})

# hqp.configure()
	sol = hqp.solve_HQPl1(params_init, x_dot_sol)
	x_dot_sol = sol.value(x_dot)

print(x_dot_sol)
con_viol = []
for i in range(1,pl):
	print(sol.value(hqp.slacks[i]))
	con_viol.append(cs.norm_1(sol.value(hqp.slacks[i])))
print(con_viol)
print(time.time() - tic)
