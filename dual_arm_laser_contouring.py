#Implementing the lex2opt on the dual arm laser contouring task

import sys
from lexopt_mosek import  lexopt_mosek
import robot as rob
import casadi as cs
from casadi import pi, cos, sin
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import json
import os
# import tkinter
from pylab import *

def circle_path(s, centre, radius, rot_mat = np.eye(3)):

	T_goal = cs.vertcat(centre[0] + radius*(cos(s)-1), centre[1] + radius*sin(s), centre[2])

	return T_goal

def inv_T_matrix(T):

	T_inv = cs.horzcat(cs.horzcat(T[0:3, 0:3].T, cs.mtimes(-T[0:3, 0:3].T, T[0:3,3])).T, [0, 0, 0, 1]).T

	return T_inv

if __name__ == '__main__':

	max_joint_vel = 500*3.14159/180
	max_joint_acc = 60*3.14159/180
	rob_settings = {'n_dof' : 18, 'no_links' : 20, 'q_min' : np.array([-2.9409, -2.5045, -2.9409, -2.1555, -5.0615, -1.5359, -3.9968, -0.1, -0.1, -2.9409, -2.5045, -2.9409, -2.1555, -5.0615, -1.5359, -3.9968, -0.1, -0.1]).T, 'q_max' : np.array([2.9409, 0.7592, 2.9409, 1.3963, 5.0615, 2.4086, 3.9968, 0.025, 0.025, 2.9409, 0.7592, 2.9409, 1.3963, 5.0615, 2.4086, 3.9968, 0.025, 0.025]).T }
	robot = rob.Robot('yumi')
	robot.set_joint_velocity_limits(lb = -max_joint_vel, ub = max_joint_vel)
	robot.set_joint_acceleration_limits(lb = -max_joint_acc, ub = max_joint_acc)

	jac_fun_rob = robot.set_kinematic_jacobian('jac_fun', 7)
	jac_fun_rob2 = robot.set_kinematic_jacobian('jac_fun', 17)

	ts = 0.005
	T = 1.000*5

	q0 = np.array([-1.35488912e+00, -8.72846052e-01, 2.18411843e+00,  6.78786296e-01,
  2.08696971e+00, -9.76390128e-01, -1.71721329e+00,  1.65969745e-03,
  1.65969745e-03,  1.47829337e+00, -5.24943547e-01, -1.95134781e+00,
  5.30517837e-01, -2.69960026e+00, -8.14070355e-01,  1.17172289e+00,
  2.06459136e-03,  2.06462524e-03]).T

	hlp = lexopt_mosek()
	max_joint_vel = np.array([max_joint_vel]*14)
	max_joint_acc = np.array([max_joint_acc]*14)
	q1, q_dot1 = hlp.create_variable(14, -max_joint_vel, max_joint_vel)
	s_1, s_dot1 = hlp.create_variable(1, cs.vertcat(-1), cs.vertcat(1))
	q_full = cs.vertcat(q1[0:7], 0, 0, q1[7:14], 0, 0)
	J1 = jac_fun_rob(q_full)
	J1 = cs.vertcat(J1[0], J1[1])
	J2 = jac_fun_rob2(q_full)
	J2 = cs.vertcat(J2[0], J2[1])

	fk_vals1 = robot.fk(q_full)[7] #forward kinematics first robot
	fk_vals2 = robot.fk(q_full)[17] #forward kinematics second robot

	#computing the point of projection of laser on the plane
	fk_relative = cs.mtimes(inv_T_matrix(fk_vals2), fk_vals1)
	Ns = np.array([0, 0, 1], ndmin=2).T
	sc = -0.3
	Nl = fk_relative[0:3,2]
	P = fk_relative[0:3,3]
	a = (-sc - cs.mtimes(Ns.T, P))/cs.mtimes(Ns.T, Nl)
	Ps = P + a*Nl

	centre = [0.0, 0.0, -sc]
	radius = 0.1
	p_des = circle_path(s_1, centre, radius)

	#Adding bounds on the acceleration
	q_dot1_prev = hlp.create_parameter(14)# hqp.opti.parameter(7, 1)
	hlp.create_constraint(q_dot1 - q_dot1_prev, 'lub', priority = 0, options = {'lb':-max_joint_acc*ts, 'ub':max_joint_acc*ts})
	hlp.create_constraint(s_1 + s_dot1*ts, 'lub',  0, options = {'lb':0, 'ub':6*pi})
	hlp.create_constraint(q1 + q_dot1*ts, 'lub', 0, options = {'lb':cs.vertcat(robot.joint_lb[0:7], robot.joint_lb[9:16]), 'ub':cs.vertcat(robot.joint_ub[0:7], robot.joint_ub[9:16])})
	x = cs.vertcat(q1, s_1)
	x_dot = cs.vertcat(q_dot1, s_dot1)
	#slack priority 2
	K_stay_path = 1
	Jacsp = cs.jacobian(Ps[0:2] - p_des[0:2], x)
	hlp.create_constraint(Jacsp@x_dot + K_stay_path*(Ps[0:2] - p_des[0:2]), 'eq', priority = 1, options = {'b':cs.vcat([0,0])})

	#slack priority 3
	K_depth = 1
	Jacdepth = cs.jacobian(a, x)
	hlp.create_constraint(Jacdepth@x_dot + K_depth*a, 'lub', priority = 2, options = {'lb':0.01, 'ub':0.03})
	dot_prod_ee_workpiece = -cs.mtimes(Ns.T, Nl)
	angle_limit = cos(10*3.14159/180)
	Jacang = cs.jacobian(dot_prod_ee_workpiece, x)
	hlp.create_constraint(-(Jacang@x_dot + dot_prod_ee_workpiece), 'ub', priority = 2, options = {'ub':-angle_limit})

	#slack priority 4
	s1_dot_rate_ff = 1.0
	hlp.create_constraint(s_dot1 - s1_dot_rate_ff, 'eq', 3, {'b':0.0})

	#slack priority 5
	hlp.create_constraint(Jacdepth@x_dot + K_depth*a, 'eq', priority = 4, options = {'b':0.02})
	hlp.create_constraint(-(Jacang@x_dot + dot_prod_ee_workpiece), 'eq', priority = 4, options = {'b':-1})

	#slack priority 6
	hlp.create_constraint(q_dot1, 'eq', 5, {"b":cs.vcat([0]*14)})

	q_opt = cs.vertcat(q0[0:7], q0[9:16], 0, cs.DM([0]*14))
	q_opt_history = q_opt
	q_dot_opt_history = cs.DM.zeros(15,1)
	hlp.configure_constraints()
	hlp.compute_matrices(cs.DM([0]*15), q_opt)
	hlp.configure_weighted_problem()
	hlp.configure_sequential_problem()

	hlp.solve_weighted_method([1,1,1,1])
	hlp.solve_sequential_problem()

	visualizationBullet = True
	counter = 1
	if visualizationBullet:

		import world_simulator
		import pybullet as p

		obj = world_simulator.world_simulator(bullet_gui = False)
		position = [0.0, 0.0, 0.0]
		orientation = [0.0, 0.0, 0.0, 1.0]

		yumiID = obj.add_robot(position, orientation, 'yumi')

		#correspondence between joint numbers in bullet and OCP determined after reading joint info of YUMI
		#from the world simulator
		joint_indices = [11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 2, 3, 4, 5, 6, 7, 8, 9]
		obj.resetJointState(yumiID, joint_indices, q0)
		joint_indices = [11, 12, 13, 14, 15, 16, 17, 1, 2, 3, 4, 5, 6, 7]

	cool_off_counter = 0
	comp_time = []
	simplex_iters = []
	constraint_violations = cs.DM([0, 0, 0, 0])
	max_err = 0;
	no_times_exceeded = 0
	comp_time.append(hlp.Mw.getSolverDoubleInfo("simTime"))
	simplex_iters.append(hlp.Mw.getSolverIntInfo("simPrimalIter"))
	print("First solve time is = " + str(hlp.Mw.getSolverDoubleInfo("simTime")))
	q_zero_integral = 0
	q_1_integral = 0
	q_2_integral = 0
	sequential_method = False

	for i in range(1000): #range(math.ceil(T/ts)):
		counter += 1
		# hqp.time_taken = 0
		print("iter :" + str(i))
		print(hlp.Mw_dict[4]['eq_b'].getValue())
		if not sequential_method:
			hlp.solve_weighted_method([1,1,1,1])
			var_dot_sol = hlp.Mw_dict['x'].level()


			# sol = hqp.solve_HQPl1(q_opt, q_dot_opt, gamma_init = 10.0)
			q_dot1_sol = var_dot_sol[0:14]
			s_dot1_sol = var_dot_sol[14]

			con_viol1 = cs.norm_1(hlp.Mw_dict[1]['eq_slack'].level())
			con_viol2 = cs.norm_1(hlp.Mw_dict[2]['lub_slack'].level()) + cs.norm_1(hlp.Mw_dict[2]['ub_slack'].level())
			con_viol3 = cs.norm_1(hlp.Mw_dict[3]['eq_slack'].level())
			con_viol4 = cs.norm_1(hlp.Mw_dict[4]['eq_slack'].level())
			# con_viols = sol.value(cs.vertcat(hqp.slacks[1], hqp.slacks[2], hqp.slacks[3], hqp.slacks[4]))
			constraint_violations = cs.horzcat(constraint_violations, cs.vertcat(con_viol1, con_viol2, con_viol3, con_viol4))
			# enablePrint()
			# sol_time = hlp.Mw.getSolverDoubleInfo("optimizerTime")
			sol_time = hlp.Mw.getSolverDoubleInfo("simPrimalTime") + hlp.Mw.getSolverDoubleInfo("simDualTime")
			simplex_iters.append(hlp.Mw.getSolverIntInfo("simPrimalIter") + hlp.Mw.getSolverIntInfo("simDualIter"))
			comp_time.append(sol_time)
			# blockPrint()


		#sol = hqp.solve_adaptive_hqp2(q_opt, q_dot_opt, gamma_init = 0.2)
		if sequential_method:
			hlp.solve_sequential_problem()
			var_dot_sol = hlp.Mdict_seq[5]['x'].level()
			q_dot1_sol = var_dot_sol[0:14]
			s_dot1_sol = var_dot_sol[14]

			print("Solver time is = " + str(hlp.time_taken))
			comp_time.append(hlp.time_taken)
			con_viol1 = hlp.optimal_slacks[1]
			con_viol2 = hlp.optimal_slacks[2]
			con_viol3 = hlp.optimal_slacks[3]
			con_viol4 = hlp.optimal_slacks[4]
			constraint_violations = cs.horzcat(constraint_violations, cs.vertcat(con_viol1, con_viol2, con_viol3, con_viol4))

		number_non_zero = sum(cs.fabs(q_dot1_sol).full() >=  1e-3)
		q_zero_integral += number_non_zero*ts
		q_2_integral += (cs.sumsqr(q_dot1_sol))*ts
		q_1_integral += sum(cs.fabs(q_dot1_sol).full() )*ts
		#compute q_1_integral

		#Update all the variables
		q_dot_opt = cs.vertcat(q_dot1_sol, s_dot1_sol)
		q_opt[0:15] += ts*q_dot_opt

		s1_opt = q_opt[14]
		if s1_opt >=1:
			print("Robot1 reached it's goal. Terminating")
			cool_off_counter += 1


		if visualizationBullet:
			obj.resetJointState(yumiID, joint_indices, q_opt[0:14])
			# time.sleep(ts)

		q_opt_history = cs.horzcat(q_opt_history, q_opt)
		q_dot_opt_history = cs.horzcat(q_dot_opt_history, q_dot_opt)

		#When bounds on acceleration
		q_opt = cs.vertcat(q_opt[0:15], q_dot1_sol)
		hlp.compute_matrices(cs.DM([0]*15), q_opt)


	if visualizationBullet:
		obj.end_simulation()

	print("Total average number of actuators used = " + str(q_zero_integral/5))
	print("Total average number of L2  = " + str(q_2_integral/5))
	print("Total average number of L1 = " + str(q_1_integral/5))

	#Implementing solution by Hierarchical QP
	figure()
	plot(list(range(counter)), constraint_violations[0,:].full().T, label = '2nd priority')
	plot(list(range(counter)), constraint_violations[1,:].full().T, label = '3rd priority')
	plot(list(range(counter)), constraint_violations[2,:].full().T, label = '4th priority')
	plot(list(range(counter)), constraint_violations[3,:].full().T, label = '5th priority')
	title("Constraint violations")
	xlabel("Time step (Control sampling time = 0.005s)")
	legend()

	figure()
	semilogy(list(range(counter)), comp_time)
	# # plot(horizon_sizes, list(average_solver_time_average_L2.values()), label = 'L2 penalty')
	# #
	title("Computation times")
	xlabel("No of samples in the horizon (sampling time = 0.05s)")
	ylabel('Time (s)')

	figure()
	plot(list(range(counter)), q_dot_opt_history[0:14,:].full().T)
	# plot(horizon_sizes, list(average_solver_time_average_L2.values()), label = 'L2 penalty')
	# #
	title("joint_velocities")
	xlabel("No of samples in the horizon (sampling time = 0.05s)")
	ylabel('rad/s')

	show(block=True)
