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

hqp = hqp_p.hqp()

n = 50
pl = 20 #priority levels

n_eq_per_level = 2
n_ineq_per_level = 3
np.random.seed(451)


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

for i in range(pl):

	A_eq[i] = cs.DM(np.random.randn(n_eq_per_level, n))
	b_eq[i] = np.random.randn(n_eq_per_level)
	hqp.create_constraint(cs.mtimes(A_eq[i],x_dot) - b_eq[i], 'equality', priority = i)
	

	A_ineq[i] = np.random.randn(n_ineq_per_level, n)
	b_upper[i] = np.random.randn(n_ineq_per_level)
	b_lower[i] = b_upper[i] - 1
	hqp.create_constraint(A_ineq[i]@x_dot, 'lub', priority = i, options = {'lb':b_upper[i] - 10, 'ub':b_upper[i]})

p_opts = {"expand":True}
s_opts = {"max_iter": 100}#, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-6}
hqp.opti.solver('ipopt', p_opts, s_opts)
# hqp.opti.solver("sqpmethod", {"expand":True, "qpsol": 'qpoases', 'print_iteration': False, 'print_header': False, 'print_status': False, "print_time":False, 'max_iter': 1000})

hqp.variables0 = params
hqp.configure()

hqp.setup_cascadedQP()
sol_chqp = hqp.solve_cascadedQP(params_init, [0]*n)
sol = hqp.solve_HQPl1(params_init, [0]*n)

x_dot_sol = sol.value(x_dot)
print(x_dot_sol)
con_viol = []
for i in range(1,pl):
	con_viol.append(cs.norm_1(sol.value(hqp.slacks[i])))
print(con_viol)

print(x_dot_sol)
con_viol = []
for i in range(1,pl):
	print(sol.value(hqp.slacks[i]))
	con_viol.append(cs.norm_1(sol.value(hqp.slacks[i])))
print(con_viol)
print(time.time() - tic)