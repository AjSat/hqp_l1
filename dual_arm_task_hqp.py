#Code where I specify an instantaneous tasks of different levels of priority
#and check which of these methods work well.
#Benchmark the computation times and the accuracy of the L1 methods against
#the number of constraints. Benchmarking is on a linearized sytem.
#So, it is L1 vs QP vs augmented PI (with nullspace projection)

import sys
# sys.path.insert(0, "/home/ajay/Desktop/hqp_l1")
from hqp import  hqp
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
# Disable
def blockPrint():
	sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
	sys.stdout = sys.__stdout__

if __name__ == '__main__':


	max_joint_acc = 150*3.14159/180
	max_joint_vel = 50*3.14159/180
	gamma = 1.4

	robot = rob.Robot('iiwa7')
	robot.set_joint_acceleration_limits(lb = -max_joint_acc, ub = max_joint_acc)
	robot.set_joint_velocity_limits(lb = -max_joint_vel, ub = max_joint_vel)
	robot.set_joint_torque_limits(lb = -100, ub = 100)

	#creating a second robot arm
	robot2 = rob.Robot('iiwa7')
	robot2.set_joint_acceleration_limits(lb = -max_joint_acc, ub = max_joint_acc)
	robot2.set_joint_velocity_limits(lb = -max_joint_vel, ub = max_joint_vel)
	robot2.set_joint_torque_limits(lb = -100, ub = 100)

	jac_fun_rob = robot.set_kinematic_jacobian('jac_fun', 6)

	ts = 0.005
	T = 1.000*5

	q0_1 = [-5.53636820e-01, 1.86726808e-01, -1.32319806e-01, -2.06761360e+00, 3.12421835e-02,  8.89043596e-01, -7.03329152e-01]
	q0_2 = [ 0.36148756, 0.19562711, 0.34339407,-2.06759027, -0.08427634, 0.89133467, 0.75131025]

	#Implementing with my L1 norm method

	hqp = hqp()

	#decision variables for robot 1
	q1, q_dot1 = hqp.create_variable(7, 1e-6)
	J1 = jac_fun_rob(q1)
	J1 = cs.vertcat(J1[0], J1[1])
	#progress variable for robot 1
	s_1, s_dot1 = hqp.create_variable(1, 1e-6)

	fk_vals1 = robot.fk(q1)[6] #forward kinematics first robot

	#decision variables for robot 2
	q2, q_dot2 = hqp.create_variable(7, 1e-6)
	J2 = jac_fun_rob(q2)
	J2 = cs.vertcat(J2[0], J2[1])
	#progress variables for robot 2
	s_2, s_dot2 = hqp.create_variable(1, 1e-6)

	fk_vals2 = robot.fk(q2)[6] #forward kinematics second robot
	fk_vals2[1,3] += 0.3 #accounting for the base offset in the y-direction

	print(robot.fk(q0_1)[6])
	print(robot2.fk(q0_2)[6])

	#specifying the cartesian trajectory of the two robots as a function
	#of the progress variable

	rob1_start_pose = cs.DM([0.3, -0.24, 0.4])
	rob1_end_pose = cs.DM([0.6, 0.2, 0.25])
	traj1 = rob1_start_pose + cs.fmin(s_1, 1)*(rob1_end_pose - rob1_start_pose)

	rob2_start_pose = cs.DM([0.3, 0.54, 0.4])
	rob2_end_pose = cs.DM([0.6, 0.1, 0.25])
	traj2 = rob2_start_pose + cs.fmin(s_2, 1)*(rob2_end_pose - rob2_start_pose)

	#Highest priority: Hard constraints on joint velocity and joint position limits
	max_joint_vel = np.array([max_joint_vel]*7)
	max_joint_acc = np.array([max_joint_acc]*7)
	hqp.create_constraint(q_dot1, 'lub', priority = 0, options = {'lb':-max_joint_vel, 'ub':max_joint_vel})
	hqp.create_constraint(q_dot2, 'lub', priority = 0, options = {'lb':-max_joint_vel, 'ub':max_joint_vel})
	hqp.create_constraint(q1 + q_dot1*ts, 'lub', priority = 0, options = {'lb':robot.joint_lb, 'ub':robot.joint_ub})
	hqp.create_constraint(q2 + q_dot2*ts, 'lub', priority = 0, options = {'lb':robot.joint_lb, 'ub':robot.joint_ub})

	#Adding bounds on the acceleration
	q_dot1_prev = hqp.create_parameter(7)# hqp.opti.parameter(7, 1)
	q_dot2_prev = hqp.create_parameter(7)#hqp.opti.parameter(7, 1)
	# hqp.opti.set_value(q_dot1_prev, cs.DM([0]*7))
	# hqp.opti.set_value(q_dot2_prev, cs.DM([0]*7))
	hqp.create_constraint(q_dot1 - q_dot1_prev, 'lub', priority = 0, options = {'lb':-max_joint_acc*ts, 'ub':max_joint_acc*ts})
	hqp.create_constraint(q_dot2 - q_dot2_prev, 'lub', priority = 0, options = {'lb':-max_joint_acc*ts, 'ub':max_joint_acc*ts})

	#2nd priority: Orientation constraints are fixed, And also the TODO: obstacle avoidance

	# slack_priority2 = opti.variable(6, 1)
	hqp.create_constraint(cs.vertcat(J1[3:6, :]@q_dot1, J2[3:6, :]@q_dot2), 'equality', priority = 1)

	dist_ees = -cs.sqrt((fk_vals1[0:3,3] - fk_vals2[0:3, 3]).T@(fk_vals1[0:3,3] - fk_vals2[0:3, 3])) + 0.3
	Jac_dist_con = cs.jacobian(dist_ees, cs.vertcat(q1, q2))
	K_coll_avoid = 1
	hqp.create_constraint(Jac_dist_con@cs.vertcat(q_dot1, q_dot2) + K_coll_avoid*dist_ees, 'ub', priority = 1, options = {'ub':np.zeros((1,1))})

	#3rd highest priority. Stay on the path for both the robots
	#for robot 1
	stay_path1 = fk_vals1[0:3, 3] - traj1
	K_stay_path1 = 1
	Jac_sp1 = cs.jacobian(stay_path1, cs.vertcat(q1, s_1))
	hqp.create_constraint(Jac_sp1@cs.vertcat(q_dot1, s_dot1) + K_stay_path1*stay_path1, 'equality', priority = 2)

	#for robot 2
	stay_path2 = fk_vals2[0:3, 3] - traj2
	K_stay_path2 = 1
	Jac_sp2 = cs.jacobian(stay_path2, cs.vertcat(q2, s_2))
	hqp.create_constraint(Jac_sp2@cs.vertcat(q_dot2, s_dot2) + K_stay_path2*stay_path2, 'equality', priority = 2)

	#4th priority: First robot reaches the goal
	s1_dot_rate_ff = 0.25
	hqp.create_constraint(s_dot1 - s1_dot_rate_ff, 'equality', priority = 3)

	#5th priority: Second robot reaches the goal
	s2_dot_rate_ff = 0.5
	hqp.create_constraint(s_dot2 - s2_dot_rate_ff, 'equality', priority = 4)

	hqp.create_constraint(q_dot1, 'equality', priority = 5)
	hqp.create_constraint(q_dot2, 'equality', priority = 5)

	p_opts = {"expand":True}

	# # s_opts = {"max_iter": 100, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-6}

	# kkt_tol_pr = 1e-6
	# kkt_tol_du = 1e-6
	# min_step_size = 1e-6
	# max_iter = 2
	# max_iter_ls = 3
	# qpsol_options = {'constr_viol_tol': kkt_tol_pr, 'dual_inf_tol': kkt_tol_du, 'verbose' : False, 'print_iter': False, 'print_header': False, 'dump_in': False, "error_on_fail" : False}
	# solver_options = {'qpsol': 'qpoases', 'qpsol_options': qpsol_options, 'verbose': False, 'tol_pr': kkt_tol_pr, 'tol_du': kkt_tol_du, 'min_step_size': min_step_size, 'max_iter': max_iter, 'max_iter_ls': max_iter_ls, 'print_iteration': True, 'print_header': False, 'print_status': False, 'print_time': True}
	hqp.opti.solver("sqpmethod", {"expand":True, "qpsol": 'qpoases', 'print_iteration': True, 'print_header': True, 'print_status': True, "print_time":True, "record_time":True, 'max_iter': 1000})
	# hqp.opti.solver("ipopt", {"expand":True, 'ipopt':{'tol':1e-6, 'print_level':0}})

	q_opt = cs.vertcat(cs.DM(q0_1), 0, cs.DM(q0_2), 0, cs.DM.zeros(14,1))
	q_opt_history = q_opt
	q_dot_opt = cs.DM([0]*16)
	q_dot_opt_history = q_dot_opt
	hqp.configure()
	hqp.time_taken = 0
	tic = time.time()

	constraint_violations = cs.DM([0, 0, 0, 0])

	#Setup for visualization
	visualizationBullet = True
	counter = 1
	if visualizationBullet:

		import world_simulator
		import pybullet as p

		obj = world_simulator.world_simulator(bullet_gui=True)

		position = [0.0, 0.0, 0.0]
		orientation = [0.0, 0.0, 0.0, 1.0]

		kukaID = obj.add_robot(position, orientation, 'iiwa7')
		position = [0.0, 0.3, 0.0]
		kukaID2 = obj.add_robot(position, orientation, 'iiwa7')

		joint_indices = [0, 1, 2, 3, 4, 5, 6]
		obj.resetJointState(kukaID, joint_indices, q0_1)
		obj.resetJointState(kukaID2, joint_indices, q0_2)
		obj.physics_ts = ts

	# time.sleep(5)
	cool_off_counter = 0
	comp_time = []
	max_err = 0;
	hqp.once_solved = False
	no_times_exceeded = 0
	sol, hierarchy_failure = hqp.solve_adaptive_hqp3(q_opt, q_dot_opt, gamma_init = 0.1, iter_lim = 10)
	comp_time.append(hqp.time_taken)
	for i in range(500):#math.ceil(T/ts)):
		counter += 1
		hqp.time_taken = 0
		# sol, hierarchy_failure = hqp.solve_adaptive_hqp3(q_opt, q_dot_opt, gamma_init = 0.1, iter_lim = 5)
		# # sol = hqp.solve_HQPl1(q_opt, q_dot_opt, gamma_init = 2.0)
		# enablePrint()
		# print(hqp.time_taken)
		# comp_time.append(hqp.time_taken)
		# blockPrint()
		# q_dot1_sol = sol.value(q_dot1)
		# q_dot2_sol = sol.value(q_dot2)
		# s_dot1_sol = sol.value(s_dot1)
		# s_dot2_sol = sol.value(s_dot2)
		# con_viols = sol.value(cs.vertcat(hqp.slacks[1], hqp.slacks[2], hqp.slacks[3], hqp.slacks[4]))
		# constraint_violations = cs.horzcat(constraint_violations, cs.vertcat(cs.norm_1(sol.value(hqp.slacks[1])), cs.norm_1(sol.value(hqp.slacks[2])), cs.norm_1(sol.value(hqp.slacks[3])), cs.norm_1(sol.value(hqp.slacks[4]))))


		#sol = hqp.solve_adaptive_hqp2(q_opt, q_dot_opt, gamma_init = 0.2)
		q_opt = q_opt.full()
		q_dot_opt = q_dot_opt.full()
		# sol_cqp, chqp_optis = hqp.solve_cascadedQP3(q_opt, q_dot_opt)
		# sol_cqp = hqp.solve_cascadedQP5(q_opt, q_dot_opt, warm_start = True)#, solver = 'ipopt')
		sol_cqp = hqp.solve_cascadedQP4(q_opt, q_dot_opt, warm_start = True)
		# # sol_h = hqp.solve_HQPl1(q_opt, q_dot_opt, gamma_init = 10.0)
		# sol = sol_cqp[4]
		sol = sol_cqp[5]

		# # print(var_dot.shape)
		# # var_dot = chqp_optis[4][3]
		# var_dot_sol = sol.value(hqp.cHQP_xdot[4])
		var_dot_sol = sol.value(hqp.cHQP_xdot[5])
		enablePrint()
		# print(var_dot_sol)
		# print(var_dot_sol2)

		print(hqp.time_taken)
		# # comp_time.append(hqp.time_taken)
		# # blockPrint()
		q_dot1_sol = var_dot_sol[0:7]
		q_dot2_sol = var_dot_sol[8:15]
		s_dot1_sol = var_dot_sol[7]
		s_dot2_sol = var_dot_sol[15]

		# sol_h = hqp.solve_HQPl1(q_opt, q_dot_opt, gamma_init = 1.5)
		# max_err = cs.fmax(max_err, cs.norm_1(sol_h.value(cs.vertcat(q_dot1, s_dot1, q_dot2, s_dot2)) - var_dot_sol))
		# enablePrint()
		# print(max_err)
		# if cs.norm_1(sol_h.value(cs.vertcat(q_dot1, s_dot1, q_dot2, s_dot2)) - var_dot_sol) >= 1e-4:
		# 	no_times_exceeded += 1
		# blockPrint()
		#Computing the constraint violations

		# print(con_viols)
		# print(q_opt)
		# print(s2_opt)

		#Update all the variables
		q_dot_opt = cs.vertcat(q_dot1_sol, s_dot1_sol, q_dot2_sol, s_dot2_sol)
		q_opt[0:16] += ts*q_dot_opt

		s1_opt = q_opt[7]
		if s1_opt >=1:
			print("Robot1 reached it's goal. Terminating")
			cool_off_counter += 1
			if cool_off_counter >= 100: #ts*100 cooloff period for the robot to exactly reach it's goal
				break

		# fk_vals1_sol = sol.value(fk_vals1)
		# fk_vals2_sol = sol.value(fk_vals2)
		# print(fk_vals1_sol)
		# print(fk_vals2_sol)

		if visualizationBullet:
			obj.setController(kukaID, "velocity", joint_indices, targetVelocities = q_dot1_sol)
			obj.setController(kukaID2, "velocity", joint_indices, targetVelocities = q_dot2_sol)
			obj.run_simulation(1)

	# 	J_sol = sol.value(J)

	# 	x_dot = cs.horzcat(x_dot, cs.mtimes(J_sol, q_dot_sol))

	# 	print(x_dot[:,-1])

	# 	q_opt += q_dot_sol*ts
		q_opt_history = cs.horzcat(q_opt_history, q_opt)
		q_dot_opt_history = cs.horzcat(q_dot_opt_history, q_dot_opt)
	# 	c1_jac_sol = sol.value(c1_jac)
	# 	c2_jac_sol = sol.value(c2_jac)
	# 	c3_jac_sol = sol.value(c3_jac)

	# 	# print(c3_jac_sol)
	# 	# print(cs.mtimes(c3_jac_sol, c2_jac_sol.T))


		#When bounds on acceleration
		q_opt = cs.vertcat(q_opt[0:16], q_dot1_sol, q_dot2_sol)
		# q_opt[16:23] =  q_dot1_sol#hqp.opti.set_value(q_dot1_prev, q_dot1_sol)
		# q_opt[23:30] = q_dot2_sol #hqp.opti.set_value(q_dot2_prev, q_dot2_sol)

	enablePrint()
	print("No of times exceeded   !" + str(no_times_exceeded))
	if visualizationBullet:
		obj.end_simulation()

	time_l1opt = time.time() - tic

	# # print("Nullspace Projection took " + str(time_nsp) +"s")
	print("L1 optimization method took " + str(time_l1opt) +"s")

	# # print(q_pi_history[-1])
	# print(q_opt_history[:,-1])

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
	plot(list(range(counter-1)), q_dot_opt_history.full().T)
	# # plot(horizon_sizes, list(average_solver_time_average_L2.values()), label = 'L2 penalty')
	# #
	title("joint_velocities")
	xlabel("No of samples in the horizon (sampling time = 0.05s)")
	ylabel('rad/s')

	figure()
	semilogy(list(range(counter)), comp_time)
	# # plot(horizon_sizes, list(average_solver_time_average_L2.values()), label = 'L2 penalty')
	# #
	title("Computation times")
	xlabel("No of samples in the horizon (sampling time = 0.05s)")
	ylabel('Time (s)')

	# figure()
	# plot(list(range(counter-1)), q_opt_history.full().T)
	# title("joint_positions")
	# xlabel("No of samples in the horizon (sampling time = 0.05s)")
	# ylabel('rad')


	# figure()
	# plot(list(range(201)), x_dot[0,:].full().T, label = 'x')
	# plot(list(range(201)), x_dot[1,:].full().T, label = 'y')
	# plot(list(range(201)), x_dot[2,:].full().T, label = 'z')
	# plot(list(range(201)), x_dot[3,:].full().T, label = 'w_x')
	# plot(list(range(201)), x_dot[4,:].full().T, label = 'w_y')
	# plot(list(range(201)), x_dot[5,:].full().T, label = 'w_z')
	# legend()
	# print(sol.value(hqp.opti.lam_g))

	temp = {'comp_time':comp_time, 'q_traj':q_opt_history.full().tolist(), 'q_dot_traj':q_dot_opt_history.full().tolist()}
	with open('../hqp_l1/comp_time.txt', 'w') as fp:
		json.dump(temp, fp)

	show(block=True)
