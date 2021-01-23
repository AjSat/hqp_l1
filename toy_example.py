#A toy example applying 2 cartesian velocity constraints on a KUKA robot.
#Mainly to illustrate the differences between HLP and HQP and also to show the
#importance of bounding acceleration

#Code where I specify an instantaneous tasks of different levels of priority
#and check which of these methods work well.
#Benchmark the computation times and the accuracy of the L1 methods against
#the number of constraints. Benchmarking is on a linearized sytem.
#So, it is L1 vs QP vs augmented PI (with nullspace projection)

import sys
# sys.path.insert(0, "/home/ajay/Desktop/hqp_l1")
from hqp import  hqp as hqp_class
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


	max_joint_acc = 60*3.14159/180
	max_joint_vel = 50*3.14159/180
	gamma = 1.4

	robot = rob.Robot('iiwa7')
	robot.set_joint_acceleration_limits(lb = -max_joint_acc, ub = max_joint_acc)
	robot.set_joint_velocity_limits(lb = -max_joint_vel, ub = max_joint_vel)
	robot.set_joint_torque_limits(lb = -100, ub = 100)

	jac_fun_rob = robot.set_kinematic_jacobian('jac_fun', 6)

	ts = 0.005
	T = 1.000*5

	# q0_1 = [-5.53636820e-01, 1.86726808e-01, -1.32319806e-01, -2.06761360e+00, 3.12421835e-02,  8.89043596e-01, -7.03329152e-01]
	# q0_1 = [ 0.36148756, 0.19562711, 0.34339407,-2.06759027, -0.08427634, 0.89133467, 0.75131025]
	# q0_1 = [0.1,0.1,0.1,0.1,0.1,0.1,0]
	q0_1 = [ 1.13126454, -0.46297661, -0.44533078,  1.91223516,  2.01132728,
       -0.69395674,  1.54762597]
	q0_1 = [-0.26906598, -0.47057001, -0.73600268, -0.66816414, -0.42619447,
       -1.25024347, -0.93842125];
	q0_1 = [0.5, 1.86726808e-01, -1.32319806e-01, -2.06761360e+00, 3.12421835e-02,  8.89043596e-01, -7.03329152e-01]

	L1_regularization = True #L2 regularization if false
	HLP = False #Implement cascadedQP if false
	acceleration_limit = False #enforce hard bounds on acceleration if true

	## Defining the first set of tasks
	hqp = hqp_class()
	#decision variables for robot 1
	if not L1_regularization:
		q1, q_dot1 = hqp.create_variable(7, 1e-6)
	else:
		q1, q_dot1 = hqp.create_variable(7, 1e-7)

	J1 = jac_fun_rob(q1)
	J1 = cs.vertcat(J1[0], J1[1])
	fk_vals1 = robot.fk(q1)[6] #forward kinematics first robot
	#Highest priority: Hard constraints on joint velocity and joint position limits
	max_joint_vel = np.array([max_joint_vel]*7)
	max_joint_acc = np.array([max_joint_acc]*7)
	hqp.create_constraint(q_dot1, 'lub', priority = 0, options = {'lb':-max_joint_vel, 'ub':max_joint_vel})
	# hqp.create_constraint(q1 + q_dot1*ts, 'lub', priority = 0, options = {'lb':robot.joint_lb, 'ub':robot.joint_ub})
	#Adding bounds on the acceleration
	q_dot1_prev = hqp.create_parameter(7)# hqp.opti.parameter(7, 1)
	if acceleration_limit:
		hqp.create_constraint(q_dot1 - q_dot1_prev, 'lub', priority = 0, options = {'lb':-max_joint_acc*ts, 'ub':max_joint_acc*ts})
	#1st priority constraint
	hqp.create_constraint(J1[0:2,:]@cs.vertcat(q_dot1) - cs.vertcat(0.2,0.2), 'equality', priority = 1)
	#2nd priority
	hqp.create_constraint(J1[0:2,:]@cs.vertcat(q_dot1) - cs.vertcat(-0.10,-0.10), 'equality', priority = 2)
	#3rd priority
	hqp.create_constraint(J1[0:2,:]@cs.vertcat(q_dot1) - cs.vertcat(-0.00,0.00), 'equality', priority = 3)

	if L1_regularization:
		hqp.create_constraint(q_dot1 - 0, 'equality', priority = 3)

	hqp.opti.solver("sqpmethod", {"expand":True, "qpsol": 'qpoases', 'print_iteration': True, 'print_header': True, 'print_status': True, "print_time":True, "record_time":True, 'max_iter': 1000})

	q_opt = cs.vertcat(cs.DM(q0_1), cs.DM.zeros(7,1))
	ee_vel_history = cs.vertcat(cs.DM.zeros(2,1))
	q_opt_history = q_opt
	q_dot_opt = cs.DM([0]*7)
	q_dot_opt_history = q_dot_opt
	hqp.configure()
	hqp.time_taken = 0
	tic = time.time()

	constraint_violations = cs.DM([0, 0, 0])

	#Setup for visualization
	visualizationBullet = True
	counter = 1
	if visualizationBullet:

		import world_simulator
		import pybullet as p

		obj = world_simulator.world_simulator()
		position = [0.0, 0.0, 0.0]
		orientation = [0.0, 0.0, 0.0, 1.0]
		kukaID = obj.add_robot(position, orientation, 'iiwa7')
		joint_indices = [0, 1, 2, 3, 4, 5, 6]
		obj.resetJointState(kukaID, joint_indices, q0_1)
		q_sensor = obj.readJointState(kukaID, joint_indices)
		print(q_sensor)
		obj.physics_ts = ts

	# time.sleep(5)
	cool_off_counter = 0
	comp_time = []
	max_err = 0;
	hqp.once_solved = False
	no_times_exceeded = 0
	sol, hierarchy_failure = hqp.solve_adaptive_hqp3(q_opt, q_dot_opt, gamma_init = 10.0, iter_lim = 10)
	comp_time.append(hqp.time_taken)
	for i in range(200):
		counter += 1
		hqp.time_taken = 0
		if HLP:
			sol, hierarchy_failure = hqp.solve_adaptive_hqp3(q_opt, q_dot_opt, gamma_init = 10.0, iter_lim = 5)
			# sol = hqp.solve_HQPl1(q_opt, q_dot_opt, gamma_init = 50.0)
			# enablePrint()
			print(hqp.time_taken)
			comp_time.append(hqp.time_taken)
			# blockPrint()
			q_dot1_sol = sol.value(q_dot1)
			con_viols = sol.value(cs.vertcat(hqp.slacks[1]))#, hqp.slacks[2], hqp.slacks[3]))
			ee_vel = cs.vec(sol.value(J1[0:2,:]@q_dot1))
			# constraint_violations = cs.horzcat(constraint_violations, cs.vertcat(cs.norm_1(sol.value(hqp.slacks[1]))))#, cs.norm_1(sol.value(hqp.slacks[2]))))#, cs.norm_1(sol.value(hqp.slacks[3]))))

		else:
			q_opt = q_opt.full()
			q_dot_opt = q_dot_opt.full()
			# sol_cqp = hqp.solve_cascadedQP5(q_opt, q_dot_opt, warm_start = True)#, solver = 'ipopt')
			sol_cqp = hqp.solve_cascadedQP_L2 (q_opt, q_dot_opt)#, solver = 'ipopt')
			sol = sol_cqp[3]
			# # # print(var_dot.shape)
			# # # var_dot = chqp_optis[4][3]
			var_dot_sol = sol.value(hqp.cHQP_xdot[3])
			# enablePrint()
			# print(hqp.time_taken)
			# # # comp_time.append(hqp.time_taken)
			# # # blockPrint()
			q_dot1_sol = var_dot_sol
			J1 = jac_fun_rob(q_opt[0:7])
			J1 = cs.vertcat(J1[0], J1[1])
			ee_vel = cs.vec(sol.value(J1[0:2,:]@q_dot1_sol))

		#Update all the variables
		q_dot_opt = cs.vertcat(q_dot1_sol)
		q_opt[0:7] += ts*q_dot_opt

		s1_opt = q_opt[7]
		if s1_opt >=1:
			print("Robot1 reached it's goal. Terminating")
			cool_off_counter += 1
			if cool_off_counter >= 100: #ts*100 cooloff period for the robot to exactly reach it's goal
				break

		if visualizationBullet:
			obj.setController(kukaID, "velocity", joint_indices, targetVelocities = q_dot1_sol)
			obj.run_simulation(1)
			q_sensor = []
			jointsInfo = obj.readJointState(kukaID, joint_indices)
			for jointInfo in jointsInfo:
				q_sensor.append(jointInfo[0])
			print(q_sensor)
			q_opt[0:7] = cs.vcat(q_sensor)


		enablePrint()
		print("EE vel is")
		print(ee_vel.shape)
		ee_vel_history = cs.horzcat(ee_vel_history, ee_vel)

	# 	q_opt += q_dot_sol*ts
		q_opt_history = cs.horzcat(q_opt_history, q_opt)
		q_dot_opt_history = cs.horzcat(q_dot_opt_history, q_dot_opt)
		#When bounds on acceleration
		q_opt = cs.vertcat(q_opt[0:7], q_dot1_sol)


	#defining the second set of tasks
	hqp2 = hqp_class()
	if not L1_regularization:
		q1, q_dot1 = hqp2.create_variable(7, 1e-6)
	else:
		q1, q_dot1 = hqp2.create_variable(7, 0)

	J1 = jac_fun_rob(q1)
	J1 = cs.vertcat(J1[0], J1[1])
	fk_vals1 = robot.fk(q1)[6] #forward kinematics first robot
	#Highest priority: Hard constraints on joint velocity and joint position limits
	hqp2.create_constraint(q_dot1, 'lub', priority = 0, options = {'lb':-max_joint_vel, 'ub':max_joint_vel})
	# hqp.create_constraint(q1 + q_dot1*ts, 'lub', priority = 0, options = {'lb':robot.joint_lb, 'ub':robot.joint_ub})
	#Adding bounds on the acceleration
	q_dot1_prev = hqp2.create_parameter(7)# hqp.opti.parameter(7, 1)
	if acceleration_limit:
		hqp2.create_constraint(q_dot1 - q_dot1_prev, 'lub', priority = 0, options = {'lb':-max_joint_acc*ts, 'ub':max_joint_acc*ts})
	#1st priority constraint
	hqp2.create_constraint(J1[0:2,:]@cs.vertcat(q_dot1) - cs.vertcat(0.2,0.2), 'equality', priority = 2)
	#2nd priority
	hqp2.create_constraint(J1[0:2,:]@cs.vertcat(q_dot1) - cs.vertcat(-0.10,-0.10), 'equality', priority = 1)
	#3rd priority
	hqp2.create_constraint(J1[0:2,:]@cs.vertcat(q_dot1) - cs.vertcat(-0.05,-0.05), 'equality', priority = 3)

	if L1_regularization:
		hqp2.create_constraint(q_dot1 - 0, 'equality', priority = 4)

	hqp2.opti.solver("sqpmethod", {"expand":True, "qpsol": 'qpoases', 'print_iteration': True, 'print_header': True, 'print_status': True, "print_time":True, "record_time":True, 'max_iter': 1000})
	hqp2.configure()
	for i in range(200):
		counter += 1
		hqp2.time_taken = 0
		if HLP:
			sol, hierarchy_failure = hqp2.solve_adaptive_hqp3(q_opt, q_dot_opt, gamma_init = 10.0, iter_lim = 5)
			# sol = hqp.solve_HQPl1(q_opt, q_dot_opt, gamma_init = 50.0)
			# enablePrint()
			print(hqp2.time_taken)
			comp_time.append(hqp2.time_taken)
			# blockPrint()
			q_dot1_sol = sol.value(q_dot1)
			con_viols = sol.value(cs.vertcat(hqp2.slacks[1]))#, hqp.slacks[2], hqp.slacks[3]))
			ee_vel = cs.vec(sol.value(J1[0:2,:]@q_dot1))
			# constraint_violations = cs.horzcat(constraint_violations, cs.vertcat(cs.norm_1(sol.value(hqp.slacks[1]))))#, cs.norm_1(sol.value(hqp.slacks[2]))))#, cs.norm_1(sol.value(hqp.slacks[3]))))

		else:
			q_opt = q_opt.full()
			q_dot_opt = q_dot_opt.full()
			# sol_cqp = hqp2.solve_cascadedQP5(q_opt, q_dot_opt, warm_start = True)#, solver = 'ipopt')
			sol_cqp = hqp2.solve_cascadedQP_L2_warmstart(q_opt, q_dot_opt)#, solver = 'ipopt')
			if L1_regularization:
				sol = sol_cqp[4]
				var_dot_sol = sol.value(hqp2.cHQP_xdot[4])
			else:
				sol = sol_cqp[3]
				var_dot_sol = sol.value(hqp2.cHQP_xdot[3])
			# # # print(var_dot.shape)
			# # # var_dot = chqp_optis[4][3]

			# enablePrint()
			# print(hqp.time_taken)
			# # # comp_time.append(hqp.time_taken)
			# # # blockPrint()
			q_dot1_sol = var_dot_sol
			J1 = jac_fun_rob(q_opt[0:7])
			J1 = cs.vertcat(J1[0], J1[1])
			ee_vel = cs.vec(sol.value(J1[0:2,:]@q_dot1_sol))

		#Update all the variables
		q_dot_opt = cs.vertcat(q_dot1_sol)
		q_opt[0:7] += ts*q_dot_opt

		s1_opt = q_opt[7]
		if s1_opt >=1:
			print("Robot1 reached it's goal. Terminating")
			cool_off_counter += 1
			if cool_off_counter >= 100: #ts*100 cooloff period for the robot to exactly reach it's goal
				break

		if visualizationBullet:
			obj.setController(kukaID, "velocity", joint_indices, targetVelocities = q_dot1_sol)
			obj.run_simulation(1)
			q_sensor = []
			jointsInfo = obj.readJointState(kukaID, joint_indices)
			for jointInfo in jointsInfo:
				q_sensor.append(jointInfo[0])
			print(q_sensor)
			q_opt[0:7] = cs.vcat(q_sensor)

	# 	q_opt += q_dot_sol*ts
		ee_vel_history = cs.horzcat(ee_vel_history, ee_vel)
		q_opt_history = cs.horzcat(q_opt_history, q_opt)
		q_dot_opt_history = cs.horzcat(q_dot_opt_history, q_dot_opt)
		#When bounds on acceleration
		q_opt = cs.vertcat(q_opt[0:7], q_dot1_sol)


	#defining the third set of tasks
	hqp3 = hqp_class()
	if not L1_regularization:
		q1, q_dot1 = hqp3.create_variable(7, 1e-6)
	else:
		q1, q_dot1 = hqp3.create_variable(7, 0)

	J1 = jac_fun_rob(q1)
	J1 = cs.vertcat(J1[0], J1[1])
	fk_vals1 = robot.fk(q1)[6] #forward kinematics first robot
	#Highest priority: Hard constraints on joint velocity and joint position limits
	hqp3.create_constraint(q_dot1, 'lub', priority = 0, options = {'lb':-max_joint_vel, 'ub':max_joint_vel})
	# hqp.create_constraint(q1 + q_dot1*ts, 'lub', priority = 0, options = {'lb':robot.joint_lb, 'ub':robot.joint_ub})
	#Adding bounds on the acceleration
	q_dot1_prev = hqp3.create_parameter(7)# hqp.opti.parameter(7, 1)
	if acceleration_limit:
		hqp3.create_constraint(q_dot1 - q_dot1_prev, 'lub', priority = 0, options = {'lb':-max_joint_acc*ts, 'ub':max_joint_acc*ts})
	#1st priority constraint
	hqp3.create_constraint(J1[0:2,:]@cs.vertcat(q_dot1) - cs.vertcat(0.2,0.2), 'equality', priority = 1)
	#2nd priority
	hqp3.create_constraint(J1[0:2,:]@cs.vertcat(q_dot1) - cs.vertcat(-0.10,-0.10), 'equality', priority = 1)
	#3rd priority
	hqp3.create_constraint(J1[0:2,:]@cs.vertcat(q_dot1) - cs.vertcat(-0.05,-0.05), 'equality', priority = 2)

	if L1_regularization:
		hqp3.create_constraint(q_dot1 - 0, 'equality', priority = 3)

	hqp3.opti.solver("sqpmethod", {"expand":True, "qpsol": 'qpoases', 'print_iteration': True, 'print_header': True, 'print_status': True, "print_time":True, "record_time":True, 'max_iter': 1000})
	hqp3.configure()
	for i in range(200):
		counter += 1
		hqp3.time_taken = 0
		if HLP:
			sol, hierarchy_failure = hqp3.solve_adaptive_hqp3(q_opt, q_dot_opt, gamma_init = 10.0, iter_lim = 5)
			# sol = hqp.solve_HQPl1(q_opt, q_dot_opt, gamma_init = 50.0)
			# enablePrint()
			print(hqp3.time_taken)
			comp_time.append(hqp3.time_taken)
			# blockPrint()
			q_dot1_sol = sol.value(q_dot1)
			con_viols = sol.value(cs.vertcat(hqp3.slacks[1]))#, hqp.slacks[2], hqp.slacks[3]))
			ee_vel = cs.vec(sol.value(J1[0:2,:]@q_dot1))
			# constraint_violations = cs.horzcat(constraint_violations, cs.vertcat(cs.norm_1(sol.value(hqp.slacks[1]))))#, cs.norm_1(sol.value(hqp.slacks[2]))))#, cs.norm_1(sol.value(hqp.slacks[3]))))

		else:
			q_opt = q_opt.full()
			q_dot_opt = q_dot_opt.full()
			# sol_cqp = hqp3.solve_cascadedQP5(q_opt, q_dot_opt, warm_start = True)#, solver = 'ipopt')
			sol_cqp = hqp3.solve_cascadedQP_L2(q_opt, q_dot_opt)#, solver = 'ipopt')
			if not L1_regularization:
				sol = sol_cqp[2]
				var_dot_sol = sol.value(hqp3.cHQP_xdot[2])
			else:
				sol = sol_cqp[3]
				var_dot_sol = sol.value(hqp3.cHQP_xdot[3])
			# # # print(var_dot.shape)
			# # # var_dot = chqp_optis[4][3]

			# enablePrint()
			# print(hqp.time_taken)
			# # # comp_time.append(hqp.time_taken)
			# # # blockPrint()
			q_dot1_sol = var_dot_sol
			J1 = jac_fun_rob(q_opt[0:7])
			J1 = cs.vertcat(J1[0], J1[1])
			ee_vel = cs.vec(sol.value(J1[0:2,:]@q_dot1_sol))

		#Update all the variables
		q_dot_opt = cs.vertcat(q_dot1_sol)
		q_opt[0:7] += ts*q_dot_opt

		s1_opt = q_opt[7]
		if s1_opt >=1:
			print("Robot1 reached it's goal. Terminating")
			cool_off_counter += 1
			if cool_off_counter >= 100: #ts*100 cooloff period for the robot to exactly reach it's goal
				break

		if visualizationBullet:
			obj.setController(kukaID, "velocity", joint_indices, targetVelocities = q_dot1_sol)
			obj.run_simulation(1)
			q_sensor = []
			jointsInfo = obj.readJointState(kukaID, joint_indices)
			for jointInfo in jointsInfo:
				q_sensor.append(jointInfo[0])
			print(q_sensor)
			q_opt[0:7] = cs.vcat(q_sensor)

		# 	q_opt += q_dot_sol*ts
		ee_vel_history = cs.horzcat(ee_vel_history, ee_vel)
		q_opt_history = cs.horzcat(q_opt_history, q_opt)
		q_dot_opt_history = cs.horzcat(q_dot_opt_history, q_dot_opt)
		#When bounds on acceleration
		q_opt = cs.vertcat(q_opt[0:7], q_dot1_sol)

	#Implement the final hqp task that will drive the robot to singularity
	hqp4 = hqp_class()
	if not L1_regularization:
		q1, q_dot1 = hqp4.create_variable(7, 1e-6)
	else:
		q1, q_dot1 = hqp4.create_variable(7, 0)

	J1 = jac_fun_rob(q1)
	J1 = cs.vertcat(J1[0], J1[1])
	fk_vals1 = robot.fk(q1)[6] #forward kinematics first robot
	#Highest priority: Hard constraints on joint velocity and joint position limits
	hqp4.create_constraint(q_dot1, 'lub', priority = 0, options = {'lb':-max_joint_vel, 'ub':max_joint_vel})
	# hqp.create_constraint(q1 + q_dot1*ts, 'lub', priority = 0, options = {'lb':robot.joint_lb, 'ub':robot.joint_ub})
	#Adding bounds on the acceleration
	q_dot1_prev = hqp4.create_parameter(7)# hqp.opti.parameter(7, 1)
	if acceleration_limit:
		hqp4.create_constraint(q_dot1 - q_dot1_prev, 'lub', priority = 0, options = {'lb':-max_joint_acc*ts, 'ub':max_joint_acc*ts})
	#1st priority constraint
	hqp4.create_constraint(J1[1,:]@cs.vertcat(q_dot1) - cs.vertcat(0.1), 'equality', priority = 1)
	hqp4.create_constraint(J1[0,:]@cs.vertcat(q_dot1) - cs.vertcat(1.0), 'equality', priority = 2)

	if L1_regularization:
		hqp4.create_constraint(q_dot1 - 0, 'equality', priority = 3)

	hqp4.opti.solver("sqpmethod", {"expand":True, "qpsol": 'qpoases', 'print_iteration': True, 'print_header': True, 'print_status': True, "print_time":True, "record_time":True, 'max_iter': 1000})
	hqp4.configure()
	for i in range(500):
		counter += 1
		hqp4.time_taken = 0
		if HLP:
			sol, hierarchy_failure = hqp4.solve_adaptive_hqp3(q_opt, q_dot_opt, gamma_init = 10.0, iter_lim = 5)
			# sol = hqp.solve_HQPl1(q_opt, q_dot_opt, gamma_init = 50.0)
			# enablePrint()
			print(hqp4.time_taken)
			comp_time.append(hqp4.time_taken)
			# blockPrint()
			q_dot1_sol = sol.value(q_dot1)
			con_viols = sol.value(cs.vertcat(hqp4.slacks[1]))#, hqp.slacks[2], hqp.slacks[3]))
			ee_vel = cs.vec(sol.value(J1[0:2,:]@q_dot1))
			# constraint_violations = cs.horzcat(constraint_violations, cs.vertcat(cs.norm_1(sol.value(hqp.slacks[1]))))#, cs.norm_1(sol.value(hqp.slacks[2]))))#, cs.norm_1(sol.value(hqp.slacks[3]))))

		else:
			q_opt = q_opt.full()
			q_dot_opt = q_dot_opt.full()
			# sol_cqp = hqp4.solve_cascadedQP5(q_opt, q_dot_opt, warm_start = True)#, solver = 'ipopt')
			sol_cqp = hqp4.solve_cascadedQP_L2(q_opt, q_dot_opt, solver = 'ipopt')
			if not L1_regularization:
				sol = sol_cqp[1]
				var_dot_sol = sol.value(hqp4.cHQP_xdot[1])
			else:
				sol = sol_cqp[2]
				var_dot_sol = sol.value(hqp4.cHQP_xdot[2])
			# # # print(var_dot.shape)
			# # # var_dot = chqp_optis[4][3]

			# enablePrint()
			# print(hqp.time_taken)
			# # # comp_time.append(hqp.time_taken)
			# # # blockPrint()
			q_dot1_sol = var_dot_sol
			J1 = jac_fun_rob(q_opt[0:7])
			J1 = cs.vertcat(J1[0], J1[1])
			ee_vel = cs.vec(sol.value(J1[0:2,:]@q_dot1_sol))

		#Update all the variables
		q_dot_opt = cs.vertcat(q_dot1_sol)
		q_opt[0:7] += ts*q_dot_opt

		if visualizationBullet:
			obj.setController(kukaID, "velocity", joint_indices, targetVelocities = q_dot1_sol)
			obj.run_simulation(1)
			q_sensor = []
			jointsInfo = obj.readJointState(kukaID, joint_indices)
			for jointInfo in jointsInfo:
				q_sensor.append(jointInfo[0])
			print(q_sensor)
			q_opt[0:7] = cs.vcat(q_sensor)

		ee_vel_history = cs.horzcat(ee_vel_history, ee_vel)
		q_opt_history = cs.horzcat(q_opt_history, q_opt)
		q_dot_opt_history = cs.horzcat(q_dot_opt_history, q_dot_opt)
		#When bounds on acceleration
		q_opt = cs.vertcat(q_opt[0:7], q_dot1_sol)

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
	# figure()
	# plot(list(range(counter)), constraint_violations[0,:].full().T, label = '2nd priority')
	# plot(list(range(counter)), constraint_violations[1,:].full().T, label = '3rd priority')
	# plot(list(range(counter)), constraint_violations[2,:].full().T, label = '4th priority')
	# # plot(list(range(counter)), constraint_violations[3,:].full().T, label = '5th priority')
	# title("Constraint violations")
	# xlabel("Time step (Control sampling time = 0.005s)")
	# legend()

	figure()
	subplot(2,1,1)
	plot(list(np.array(range(counter))*ts), q_dot_opt_history[0,:].full().T, label = 'q1')
	plot(list(np.array(range(counter))*ts), q_dot_opt_history[1,:].full().T, label = 'q2')
	plot(list(np.array(range(counter))*ts), q_dot_opt_history[2,:].full().T, label = 'q3')
	plot(list(np.array(range(counter))*ts), q_dot_opt_history[3,:].full().T, label = 'q4')
	plot(list(np.array(range(counter))*ts), q_dot_opt_history[4,:].full().T, label = 'q5')
	plot(list(np.array(range(counter))*ts), q_dot_opt_history[5,:].full().T, label = 'q6')
	plot(list(np.array(range(counter))*ts), q_dot_opt_history[6,:].full().T, label = 'q7')
	title("HQP joint velocities")
	xlabel("Time (s)")
	ylabel('Joint velocities (rad/s)')
	plt.grid()
	legend()

	subplot(2,1,2)
	plot(list(np.array(range(counter))*ts), ee_vel_history[0,:].full().T, label = 'x-direction')
	plot(list(np.array(range(counter))*ts), ee_vel_history[1,:].full().T, label = 'y-direction')
	# # plot(horizon_sizes, list(average_solver_time_average_L2.values()), label = 'L2 penalty')
	# #
	xlabel("Time (s)")
	ylabel('End effector velocities (m/s)')
	plt.grid()
	legend()

	# figure()
	# semilogy(list(range(counter)), comp_time)
	# # # plot(horizon_sizes, list(average_solver_time_average_L2.values()), label = 'L2 penalty')
	# # #
	# title("Computation times")
	# xlabel("No of samples in the horizon (sampling time = 0.05s)")
	# ylabel('Time (s)')

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
