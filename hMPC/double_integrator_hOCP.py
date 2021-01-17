import sys
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
from tasho.utils import geometry

if __name__ == "__main__":

	horizon_size = 20;
	t_mpc = 0.1;
	tc = tp.task_context(horizon_size*t_mpc)

	x = tc.create_expression('x', 'state', (1,1))
	x_dot = tc.create_expression('x_dot', 'state', (1,1))
	x_ddot = tc.create_expression('x_ddot', 'control', (1,1))

	tc.set_dynamics(x, x_dot)
	tc.set_dynamics(x_dot, x_ddot)

	x_init_con = {'expression':x, 'reference':0}
	x_dot_init_con = {'expression':x_dot, 'reference':0}
	init_constraints = {'initial_constraints':[x_init_con, x_dot_init_con]}

	x_dot_bounds = {'lub':True, 'hard':True, 'expression':x_dot, 'upper_limits':2, 'lower_limits':-2}
	x_ddot_bounds = {'lub':True, 'hard':True, 'expression':x_ddot, 'upper_limits':5, 'lower_limits':-5}
	x_ddot_ref = {'hard':False, 'equality':True, 'expression':x_ddot, 'reference':0, 'gain':0.02}
	x_dot_ref = {'hard':False, 'equality':True, 'expression':x_dot, 'reference':1, 'gain':1, 'norm':'L1'}
	x_ref = {'hard':False, 'equality':True, 'expression':x, 'reference':1, 'gain':10, 'norm':'L1'}

	path_constraints = {'path_constraints':[x_dot_bounds, x_ddot_bounds, x_dot_ref, x_ref, x_ddot_ref]}

	tc.add_task_constraint(init_constraints)
	tc.add_task_constraint(path_constraints)

	tc.set_ocp_solver('ipopt', {'ipopt':{"max_iter": 1000, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-3}})
	disc_settings = {'discretization method': 'multiple shooting', 'horizon size': horizon_size, 'order':1, 'integration':'rk'}
	tc.set_discretization_settings(disc_settings)
	sol = tc.solve_ocp()

	print(sol.sample(x, grid = 'control'))