#Code where I specify an instantaneous tasks of different levels of priority
#and check which of these methods work well.
#Benchmark the computation times and the accuracy of the L1 methods against
#the number of constraints. Benchmarking is on a linearized sytem.
#So, it is L1 vs QP vs augmented PI (with nullspace projection)

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


if __name__ == '__main__':

	robot = rob.Robot('iiwa7')
	max_joint_acc = 180*3.14159/180
	max_joint_vel = 90*3.14159/180
	position = [0.0, 0.0, 0.0]
	orientation = [0.0, 0.0, 0.0, 1.0]
	robot.set_joint_acceleration_limits(lb = -max_joint_acc, ub = max_joint_acc)
	robot.set_joint_velocity_limits(lb = -max_joint_vel, ub = max_joint_vel)
	robot.set_joint_torque_limits(lb = -100, ub = 100)
	jac_fun_rob = robot.set_kinematic_jacobian('jac_fun', 6)

	ts = 0.005
	T = 1.000
	#task1 : x_direction 0.1 m/s velocity
	#task2 : y_direction 0.1 m/s velocity
	#task3 : x_direction -0.05 m/s velocity
	#task4 (soft constraint) : maximize manipulability (check later)
	q0 = [0.6967786678678314, 1.0571249256028108, 0.14148034853277666, -1.270205899164967, 0.24666659678004457, 0.7847437220601475, 0.41090241207031053]

	#Implementing with my L1 norm method

	opti = cs.Opti()

	q = opti.parameter(7,1)
	J = jac_fun_rob(q)
	J = cs.vertcat(J[0], J[1])

	q_dot = opti.variable(7, 1)
	
	#Adding the motion constraints
	c1 = J[0:2,:]@q_dot - cs.MX([0.1, 0.045])
	slack1 = opti.variable(2, 1)
	opti.subject_to(-slack1 <= (c1 <= slack1))

	c2 = J[1:4,:]@q_dot - cs.MX([0.02, 0.055, 0.0])
	slack2 = opti.variable(3, 1)
	opti.subject_to(-slack2 <= (c2 <= slack2))

	c3 = J[3:6,:]@q_dot - cs.MX([0.1, 0.04, 0.03])
	slack3 = opti.variable(3, 1)
	opti.subject_to(-slack3 <= (c3 <= slack3))

	slg1 = opti.parameter( 2 , 1)
	slg2 = opti.parameter(3, 1)
	slg3 = opti.parameter(3, 1)
	objective = slg1.T@slack1 + slg2.T@slack2 + slg3.T@slack3 + q_dot.T@q_dot*1e-3

	opti.subject_to(robot.joint_lb <= (q + q_dot*ts <= robot.joint_ub))
	opti.subject_to(-max_joint_vel <= (q_dot <= max_joint_vel))
	# objective = slg[0]*slack1*slack1 + slg[1]*slack2*slack2 + slg[2]*slack3*slack3 + q_dot.T@q_dot*0.001

	# opti.subject_to(cs.vertcat(slack1, slack2, slack3) >= 0)
	
	opti.minimize(objective)

	p_opts = {"expand":True}

	# s_opts = {"max_iter": 100, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-6}

	kkt_tol_pr = 1e-6
	kkt_tol_du = 1e-6
	min_step_size = 1e-6
	max_iter = 2
	max_iter_ls = 3
	qpsol_options = {'constr_viol_tol': kkt_tol_pr, 'dual_inf_tol': kkt_tol_du, 'verbose' : False, 'print_iter': False, 'print_header': False, 'dump_in': False, "error_on_fail" : False}
	solver_options = {'qpsol': 'qrqp', 'qpsol_options': qpsol_options, 'verbose': False, 'tol_pr': kkt_tol_pr, 'tol_du': kkt_tol_du, 'min_step_size': min_step_size, 'max_iter': max_iter, 'max_iter_ls': max_iter_ls, 'print_iteration': True, 'print_header': False, 'print_status': False, 'print_time': True}
	opti.solver("sqpmethod", {"qpsol": 'qpoases', 'print_iteration': False, 'print_header': False, 'print_status': False, "print_time":False})

	q_opt = cs.DM(q0)
	q_opt_history = q_opt

	c1_jac = cs.jacobian(c1, q_dot)
	c2_jac = cs.jacobian(c2, q_dot)
	c3_jac = cs.jacobian(c3, q_dot)

	tic = time.time()
	opti.set_value(slg1, [100, 100])
	opti.set_value(slg2, [10, 10, 10])
	opti.set_value(slg3, [1, 1, 1])
	x_dot = cs.DM([0, 0, 0, 0, 0, 0])
	for i in range(math.ceil(T/ts)):

		opti.set_value(q, q_opt.full())
		sol = opti.solve()

		q_dot_sol = sol.value(q_dot)
		J_sol = sol.value(J)

		x_dot = cs.horzcat(x_dot, cs.mtimes(J_sol, q_dot_sol))

		print(x_dot[:,-1])

		q_opt += q_dot_sol*ts
		q_opt_history = cs.horzcat(q_opt_history, q_opt)

		c1_jac_sol = sol.value(c1_jac)
		c2_jac_sol = sol.value(c2_jac)
		c3_jac_sol = sol.value(c3_jac)

		# print(c3_jac_sol)
		# print(cs.mtimes(c3_jac_sol, c2_jac_sol.T))

		#compute the gains for the lowest priority constraints 
		alpha3 = cs.DM([1, 1, 1])
		for j in range(1,3):
			alpha3[j] = alpha3[0]*cs.norm_1(c3_jac_sol[0,:])/cs.norm_1(c3_jac_sol[j,:])

		#computing the gains for the next highest priority constraints
		alpha2 = cs.DM([1, 1, 1])
		alpha2[0] = 1.1*3*alpha3[0]*cs.norm_1(c3_jac_sol[0,:])/cs.norm_1(c2_jac_sol[0,:])
		for j in range(1, 3):
			alpha2[j] = alpha2[0]*cs.norm_1(c2_jac_sol[0,:])/cs.norm_1(c2_jac_sol[j,:])

		alpha1 = cs.DM([1, 1])
		alpha1[0] = 1.1*3*alpha3[0]*cs.norm_1(c3_jac_sol[0,:])/cs.norm_1(c1_jac_sol[0,:]) 
		alpha1[0] += 1.1*3*alpha2[0]*cs.norm_1(c2_jac_sol[0,:])/cs.norm_1(c1_jac_sol[0,:])
		for j in range(1, 2):
			alpha1[j] = alpha1[0]*cs.norm_1(c1_jac_sol[0,:])/cs.norm_1(c1_jac_sol[j,:])
		print("alphas are " + str(alpha1) + "  " + str(alpha2) + "  " + str(alpha3))
		opti.set_value(slg1, alpha1)
		opti.set_value(slg2, alpha2)
		opti.set_value(slg3, alpha3)
		opti.set_initial(q_dot, q_dot_sol)
		# print(q_opt)


	time_l1opt = time.time() - tic

	# print("Nullspace Projection took " + str(time_nsp) +"s")
	print("L1 optimization method took " + str(time_l1opt) +"s")

	# print(q_pi_history[-1])
	print(q_opt_history[:,-1])

	#Implementing solution by Hierarchical QP
	figure()
	plot(list(range(201)), q_opt_history.full().T, 'r', label = 'L1-opt')
	# plot(list(range(3001)), q_pi_history.full().T, 'g', label = 'PI NSP')
	# plot(horizon_sizes, list(average_solver_time_average_L2.values()), label = 'L2 penalty')
	# legend()
	# title("Average solver time per MPC iteration vs Horizon size")
	# xlabel("No of samples in the horizon (sampling time = 0.05s)")
	# ylabel('Seconds')
	

	figure()
	plot(list(range(201)), x_dot[0,:].full().T, label = 'x')
	plot(list(range(201)), x_dot[1,:].full().T, label = 'y')
	plot(list(range(201)), x_dot[2,:].full().T, label = 'z')
	plot(list(range(201)), x_dot[3,:].full().T, label = 'w_x')
	plot(list(range(201)), x_dot[4,:].full().T, label = 'w_y')
	plot(list(range(201)), x_dot[5,:].full().T, label = 'w_z')
	legend()
	show(block=True)