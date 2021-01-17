# Implementing the same instantaneous control problem now using MPC. Using fixed weights for this initial feasibility study.

import sys
# sys.path.insert(0, "/home/ajay/Desktop/hqp_l1")
import casadi as cs
from casadi import pi, cos, sin
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import json
import os
from tasho import task_prototype_rockit as tp
from tasho import input_resolution, world_simulator
from tasho import robot as rob

# import tkinter
from pylab import *
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

if __name__ == '__main__':

	
	max_joint_acc = 120*3.14159/180
	max_joint_vel = 60*3.14159/180
	horizon_size = 40
	t_mpc = 0.05

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

	tc = tp.task_context(horizon_size*t_mpc)


	#decision variables for robot 1
	q1, q_dot1, q_ddot1, q01, q_dot01 = input_resolution.acceleration_resolved(tc, robot, {})
	# q1, q_dot1, q_ddot1, tau, q01, q_dot01 = input_resolution.torque_resolved(tc, robot, {'forward_dynamics_constraints': False})
	#progress variable for robot 1
	s_1 = tc.create_expression('s1', 'state', (1,1))
	s_dot1 = tc.create_expression('s_dot1', 'state', (1,1))
	s_ddot1 = tc.create_expression('s_ddot1', 'control', (1,1))
	tc.ocp.subject_to(0 <= (s_1 <= 1))

	tc.set_dynamics(s_1, s_dot1)
	tc.set_dynamics(s_dot1, s_ddot1)

	fk_vals1 = robot.fk(q1)[6] #forward kinematics first robot

	#decision variables for robot 2
	q2, q_dot2, q_ddot2, q02, q_dot02 = input_resolution.acceleration_resolved(tc, robot, {})
	# q2, q_dot2, q_ddot2, tau2, q02, q_dot02 = input_resolution.torque_resolved(tc, robot2, {'forward_dynamics_constraints': False})
	#progress variable for robot 1
	s_2 = tc.create_expression('s2', 'state', (1,1))
	s_dot2 = tc.create_expression('s_dot2', 'state', (1,1))
	s_ddot2 = tc.create_expression('s_ddot2', 'control', (1,1))
	tc.ocp.subject_to(0 <= (s_2 <= 1))
	tc.set_dynamics(s_2, s_dot2)
	tc.set_dynamics(s_dot2, s_ddot2)

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
	#Already added by default by TaSHo

	#2nd priority: Orientation constraints are fixed, And also the TODO: obstacle avoidance

	# slack_priority2 = opti.variable(6, 1)
	orientation_rob1 = {'equality':True, 'hard':False, 'expression':fk_vals1, 'reference':robot.fk(q0_1)[6], 'type':'Frame', 'rot_gain':50, 'trans_gain':0, 'norm':'L1'}
	orientation_rob2 = {'equality':True, 'hard':False, 'expression':fk_vals2, 'reference':robot.fk(q0_2)[6], 'type':'Frame', 'rot_gain':50, 'trans_gain':0, 'norm':'L1'}

	dist_ees = -cs.sqrt((fk_vals1[0:3,3] - fk_vals2[0:3, 3]).T@(fk_vals1[0:3,3] - fk_vals2[0:3, 3])) + 0.2
	coll_avoid = {'inequality':True, 'hard':False, 'expression':dist_ees, 'upper_limits':0, 'gain':50, 'norm':'L1'}
	# Jac_dist_con = cs.jacobian(dist_ees, cs.vertcat(q1, q2))
	# K_coll_avoid = 1
	# hqp.create_constraint(Jac_dist_con@cs.vertcat(q_dot1, q_dot2) + K_coll_avoid*dist_ees, 'ub', priority = 1, options = {'ub':np.zeros((1,1))})

	#3rd highest priority. Stay on the path for both the robots
	#for robot 1
	stay_path1 = fk_vals1[0:3, 3] - traj1
	print(stay_path1.shape)
	trans_rob1x = {'equality':True, 'hard':False, 'expression':stay_path1[0], 'reference':0, 'gain':20, 'norm':'L1'}
	trans_rob1y = {'equality':True, 'hard':False, 'expression':stay_path1[1], 'reference':0, 'gain':20, 'norm':'L1'}
	trans_rob1z = {'equality':True, 'hard':False, 'expression':stay_path1[2], 'reference':0, 'gain':20, 'norm':'L1'}
	# sl1 = tc.ocp.control(1,1)
	# tc.ocp.subject_to(-sl1 <= (stay_path1[0] <= sl1))
	# tc.ocp.add_objective(tc.ocp.integral(sl1)*2)
	#for robot 2
	stay_path2 = fk_vals2[0:3, 3] - traj2
	trans_rob2x = {'equality':True, 'hard':False, 'expression':stay_path2[0], 'reference':0, 'gain':100, 'norm':'L1'}
	trans_rob2y = {'equality':True, 'hard':False, 'expression':stay_path2[1], 'reference':0, 'gain':100, 'norm':'L1'}
	trans_rob2z = {'equality':True, 'hard':False, 'expression':stay_path2[2], 'reference':0, 'gain':100, 'norm':'L1'}

	#4th priority: First robot reaches the goal
	s1_dot_rate_ff = 0.25
	s1_dot_con = {'equality':True, 'hard':False, 'expression':s_1, 'reference':1, 'gain':1, 'norm':'L1'}

	#5th priority: Second robot reaches the goal
	s2_dot_rate_ff = 0.5
	s2_dot_con = {'equality':True, 'hard':False, 'expression':s_2, 'reference':1, 'gain':1.2, 'norm':'L1'}

	path_con = {'path_constraints': [orientation_rob1, orientation_rob2, trans_rob1x, trans_rob1y, trans_rob1z, trans_rob2y, trans_rob2z, trans_rob2x,s1_dot_con, s2_dot_con, coll_avoid]}
	tc.add_task_constraint(path_con)

	acc_cost1 = {'equality':True, 'hard':False, 'expression':q_ddot1, 'reference':0, 'gain':0.01}
	acc_cost2 = {'equality':True, 'hard':False, 'expression':q_ddot2, 'reference':0, 'gain':0.01}
	s_ddot1_cost = {'equality':True, 'hard':False, 'expression':s_ddot1, 'reference':0, 'gain':0.01}
	s_ddot2_cost = {'equality':True, 'hard':False, 'expression':s_ddot2, 'reference':0, 'gain':0.01}
	regularization = {'path_constraints':[acc_cost1, acc_cost2, s_ddot2_cost, s_ddot1_cost]}
	tc.add_task_constraint(regularization)

	#Adding terminal constraints
	q_dot1_fin = {'equality':True, 'hard':True, 'expression':q_dot1, 'reference':0}
	q_dot2_fin = {'equality':True, 'hard':True, 'expression':q_dot2, 'reference':0}
	s_dot1_fin = {'equality':True, 'hard':True, 'expression':s_dot1, 'reference':0}
	s_dot2_fin = {'equality':True, 'hard':True, 'expression':s_dot2, 'reference':0}
	terminal_constraints = {'final_constraints':[q_dot1_fin, q_dot2_fin, s_dot2_fin, s_dot1_fin]}
	tc.add_task_constraint(terminal_constraints)
	p_opts = {"expand":True}

	# # s_opts = {"max_iter": 100, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-6}

	# kkt_tol_pr = 1e-6
	# kkt_tol_du = 1e-6
	# min_step_size = 1e-6
	# max_iter = 2
	# max_iter_ls = 3
	# qpsol_options = {'constr_viol_tol': kkt_tol_pr, 'dual_inf_tol': kkt_tol_du, 'verbose' : False, 'print_iter': False, 'print_header': False, 'dump_in': False, "error_on_fail" : False}
	# solver_options = {'qpsol': 'qrqp', 'qpsol_options': qpsol_options, 'verbose': False, 'tol_pr': kkt_tol_pr, 'tol_du': kkt_tol_du, 'min_step_size': min_step_size, 'max_iter': max_iter, 'max_iter_ls': max_iter_ls, 'print_iteration': True, 'print_header': False, 'print_status': False, 'print_time': True}
	# hqp.opti.solver("sqpmethod", {"expand":True, "qpsol": 'qpoases', 'print_iteration': True, 'print_header': True, 'print_status': True, "print_time":True, "record_time":True, 'max_iter': 1000})
	tc.set_ocp_solver("ipopt", {"expand":True, 'ipopt':{'tol':1e-6, 'print_level':5}})
	# tc.set_ocp_solver('ipopt', {'ipopt':{"max_iter": 1000, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-3}})
	disc_settings = {'discretization method': 'multiple shooting', 'horizon size': horizon_size, 'order':1, 'integration':'rk'}
	tc.set_discretization_settings(disc_settings)

	tc.ocp.set_value(q01, q0_1)
	tc.ocp.set_value(q02, q0_2)
	tc.ocp.set_value(q_dot01, 0)
	tc.ocp.set_value(q_dot02, 0)

	sol = tc.solve_ocp()

	# begin the visualization of applying OCP solution in open loop

	import pybullet as p

	obj = world_simulator.world_simulator()

	position = [0.0, 0.0, 0.0]
	orientation = [0.0, 0.0, 0.0, 1.0]

	kukaID = obj.add_robot(position, orientation, 'iiwa7')
	position = [0.0, 0.3, 0.0]
	kukaID2 = obj.add_robot(position, orientation, 'iiwa7')

	joint_indices = [0, 1, 2, 3, 4, 5, 6]
	obj.resetJointState(kukaID, joint_indices, q0_1)
	obj.resetJointState(kukaID2, joint_indices, q0_2)
	obj.physics_ts = ts
	_, qdd_sol1 = sol.sample(q_ddot1, grid="control")
	# _, tau_sol = sol.sample(tau, grid="control")
	# print(tau_sol)
	# time.sleep(5)
	_, q_sol = sol.sample(q1, grid="control")
	_, q_sol2 = sol.sample(q2, grid="control")
	print(robot.fk(q_sol2[-1])[6])
	print(q_sol[-1])
	ts, q_dot_sol1 = sol.sample(q_dot1, grid="control")
	ts, q_dot_sol2 = sol.sample(q_dot2, grid="control")
	print(q_dot_sol1)
	obj.resetJointState(kukaID, joint_indices, q0_1)
	obj.resetJointState(kukaID2, joint_indices, q0_2)
	obj.setController(kukaID, "velocity", joint_indices, targetVelocities = q_dot_sol1[0])
	obj.setController(kukaID2, "velocity", joint_indices, targetVelocities = q_dot_sol2[0])
	
	no_samples = int(t_mpc/obj.physics_ts)
	for i in range(horizon_size):
		q_vel_current1 = 0.5*(q_dot_sol1[i] + q_dot_sol1[i+1])
		obj.setController(kukaID, "velocity", joint_indices, targetVelocities = q_vel_current1)
		q_vel_current2 = 0.5*(q_dot_sol2[i] + q_dot_sol2[i+1])
		obj.setController(kukaID2, "velocity", joint_indices, targetVelocities = q_vel_current2)
		obj.run_simulation(no_samples)
	obj.setController(kukaID, "velocity", joint_indices, targetVelocities = q_dot_sol1[-1])
	obj.setController(kukaID2, "velocity", joint_indices, targetVelocities = q_dot_sol2[-1])
	obj.run_simulation(100)
	obj.end_simulation()



	t_s, obs_avoidance = sol.sample(cs.fmax(dist_ees, 0), grid="control")
	t_s, stay_on_path = sol.sample((cs.vertcat(stay_path1, stay_path2)), grid="control")
	stay_on_path_norm = []
	for i in range(len(t_s)):
		stay_on_path_norm.append(cs.norm_1(stay_on_path[i, :]))
	t_s, s_1_sol = sol.sample(cs.fabs(s_1-1), grid = 'control')
	t_s, s_2_sol = sol.sample(cs.fabs(s_2-1), grid = 'control')


	figure()
	plot(t_s, obs_avoidance, label = 'obstacle avoidance (2)')
	plot(t_s, stay_on_path_norm, label = 'stay on path constraint (3)')
	plot(t_s, s_2_sol, label = 'distance from goal robot 1 (4)')
	plot(t_s, s_1_sol, label = 'distance from goal robot 2 (5)')
	# plot(list(range(counter)), constraint_violations[3,:].full().T, label = '5th priority')
	title("Constraint violations")
	xlabel("Time step (Control sampling time = 0.005s)")
	legend()
	show(block=True)

	# tic = time.time()

	# constraint_violations = cs.DM([0, 0, 0, 0])

	# #Setup for visualization
	# visualizationBullet = True
	# counter = 1
	# if visualizationBullet:

	# 	import world_simulator
	# 	import pybullet as p

	# 	obj = world_simulator.world_simulator()

	# 	position = [0.0, 0.0, 0.0]
	# 	orientation = [0.0, 0.0, 0.0, 1.0]

	# 	kukaID = obj.add_robot(position, orientation, 'iiwa7')
	# 	position = [0.0, 0.3, 0.0]
	# 	kukaID2 = obj.add_robot(position, orientation, 'iiwa7')

	# 	joint_indices = [0, 1, 2, 3, 4, 5, 6]
	# 	obj.resetJointState(kukaID, joint_indices, q0_1)
	# 	obj.resetJointState(kukaID2, joint_indices, q0_2)
	# 	obj.physics_ts = ts

	# cool_off_counter = 0
	# comp_time = []
	# max_err = 0;
	# hqp.once_solved = False
	# no_times_exceeded = 0
	# for i in range(math.ceil(T/ts)):
	# 	counter += 1
	# 	hqp.time_taken = 0
	# 	sol = hqp.solve_HQPl1(q_opt, q_dot_opt, gamma_init = 50.0)
	# 	enablePrint()
	# 	print(hqp.time_taken)
	# 	comp_time.append(hqp.time_taken)
	# 	blockPrint()
	# 	q_dot1_sol = sol.value(q_dot1)
	# 	q_dot2_sol = sol.value(q_dot2)
	# 	s_dot1_sol = sol.value(s_dot1)
	# 	s_dot2_sol = sol.value(s_dot2)
	# 	con_viols = sol.value(cs.vertcat(hqp.slacks[1], hqp.slacks[2], hqp.slacks[3], hqp.slacks[4]))
	# 	constraint_violations = cs.horzcat(constraint_violations, cs.vertcat(cs.norm_1(sol.value(hqp.slacks[1])), cs.norm_1(sol.value(hqp.slacks[2])), cs.norm_1(sol.value(hqp.slacks[3])), cs.norm_1(sol.value(hqp.slacks[4]))))


	# 	# sol = hqp.solve_adaptive_hqp2(q_opt, q_dot_opt, gamma_init = 0.2)
	# 	# q_opt = q_opt.full()
	# 	# q_dot_opt = q_dot_opt.full()
	# 	# # sol_cqp, chqp_optis = hqp.solve_cascadedQP3(q_opt, q_dot_opt)
	# 	# sol_cqp = hqp.solve_cascadedQP5(q_opt, q_dot_opt, warm_start = True)#, solver = 'ipopt')
	# 	# # sol_h = hqp.solve_HQPl1(q_opt, q_dot_opt, gamma_init = 10.0)
	# 	# sol = sol_cqp[4]
	# 	# # print(var_dot.shape)
	# 	# # var_dot = chqp_optis[4][3]
	# 	# var_dot_sol = sol.value(hqp.cHQP_xdot[4])
	# 	# # enablePrint()
	# 	# # # print(hqp.time_taken)
	# 	# # comp_time.append(hqp.time_taken)
	# 	# # blockPrint()
	# 	# q_dot1_sol = var_dot_sol[0:7]
	# 	# q_dot2_sol = var_dot_sol[8:15]
	# 	# s_dot1_sol = var_dot_sol[7]
	# 	# s_dot2_sol = var_dot_sol[15]

	# 	# sol_h = hqp.solve_HQPl1(q_opt, q_dot_opt, gamma_init = 1.5)
	# 	# max_err = cs.fmax(max_err, cs.norm_1(sol_h.value(cs.vertcat(q_dot1, s_dot1, q_dot2, s_dot2)) - var_dot_sol))
	# 	# enablePrint()
	# 	# print(max_err)
	# 	# if cs.norm_1(sol_h.value(cs.vertcat(q_dot1, s_dot1, q_dot2, s_dot2)) - var_dot_sol) >= 1e-4:
	# 	# 	no_times_exceeded += 1
	# 	# blockPrint()
	# 	#Computing the constraint violations
		
	# 	# print(con_viols)
	# 	# print(q_opt)
	# 	# print(s2_opt)

	# 	#Update all the variables
	# 	q_dot_opt = cs.vertcat(q_dot1_sol, s_dot1_sol, q_dot2_sol, s_dot2_sol)
	# 	q_opt[0:16] += ts*q_dot_opt

	# 	s1_opt = q_opt[7]
	# 	if s1_opt >=1:
	# 		print("Robot1 reached it's goal. Terminating")
	# 		cool_off_counter += 1
	# 		if cool_off_counter >= 100: #ts*100 cooloff period for the robot to exactly reach it's goal
	# 			break

	# 	# fk_vals1_sol = sol.value(fk_vals1)
	# 	# fk_vals2_sol = sol.value(fk_vals2)
	# 	# print(fk_vals1_sol)
	# 	# print(fk_vals2_sol)

	# 	if visualizationBullet:
	# 		obj.setController(kukaID, "velocity", joint_indices, targetVelocities = q_dot1_sol)
	# 		obj.setController(kukaID2, "velocity", joint_indices, targetVelocities = q_dot2_sol)
	# 		obj.run_simulation(1)

	# # 	J_sol = sol.value(J)

	# # 	x_dot = cs.horzcat(x_dot, cs.mtimes(J_sol, q_dot_sol))

	# # 	print(x_dot[:,-1])

	# # 	q_opt += q_dot_sol*ts
	# 	q_opt_history = cs.horzcat(q_opt_history, q_opt)
	# 	q_dot_opt_history = cs.horzcat(q_dot_opt_history, q_dot_opt)
	# # 	c1_jac_sol = sol.value(c1_jac)
	# # 	c2_jac_sol = sol.value(c2_jac)
	# # 	c3_jac_sol = sol.value(c3_jac)

	# # 	# print(c3_jac_sol)
	# # 	# print(cs.mtimes(c3_jac_sol, c2_jac_sol.T))


	# 	#When bounds on acceleration
	# 	q_opt = cs.vertcat(q_opt[0:16], q_dot1_sol, q_dot2_sol)
	# 	# q_opt[16:23] =  q_dot1_sol#hqp.opti.set_value(q_dot1_prev, q_dot1_sol)
	# 	# q_opt[23:30] = q_dot2_sol #hqp.opti.set_value(q_dot2_prev, q_dot2_sol)

	# enablePrint()
	# print("No of times exceeded   !" + str(no_times_exceeded))
	# if visualizationBullet:
	# 	obj.end_simulation()

	# time_l1opt = time.time() - tic

	# # # print("Nullspace Projection took " + str(time_nsp) +"s")
	# print("L1 optimization method took " + str(time_l1opt) +"s")

	# # # print(q_pi_history[-1])
	# # print(q_opt_history[:,-1])

	# #Implementing solution by Hierarchical QP
	# figure()
	# plot(list(range(counter)), constraint_violations[0,:].full().T, label = '2nd priority')
	# plot(list(range(counter)), constraint_violations[1,:].full().T, label = '3rd priority')
	# plot(list(range(counter)), constraint_violations[2,:].full().T, label = '4th priority')
	# plot(list(range(counter)), constraint_violations[3,:].full().T, label = '5th priority')
	# title("Constraint violations")
	# xlabel("Time step (Control sampling time = 0.005s)")
	# legend()

	# # figure()
	# # plot(list(range(counter-1)), q_dot_opt_history.full().T)
	# # # # plot(horizon_sizes, list(average_solver_time_average_L2.values()), label = 'L2 penalty')
	# # # # 
	# # title("joint_velocities")
	# # xlabel("No of samples in the horizon (sampling time = 0.05s)")
	# # ylabel('rad/s')

	# # figure()
	# # plot(list(range(counter-1)), q_opt_history.full().T)
	# # title("joint_positions")
	# # xlabel("No of samples in the horizon (sampling time = 0.05s)")
	# # ylabel('rad')
	

	# # figure()
	# # plot(list(range(201)), x_dot[0,:].full().T, label = 'x')
	# # plot(list(range(201)), x_dot[1,:].full().T, label = 'y')
	# # plot(list(range(201)), x_dot[2,:].full().T, label = 'z')
	# # plot(list(range(201)), x_dot[3,:].full().T, label = 'w_x')
	# # plot(list(range(201)), x_dot[4,:].full().T, label = 'w_y')
	# # plot(list(range(201)), x_dot[5,:].full().T, label = 'w_z')
	# # legend()
	# # print(sol.value(hqp.opti.lam_g))

	# show(block=True)



