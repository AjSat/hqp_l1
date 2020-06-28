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

	
	max_joint_acc = 90*3.14159/180
	max_joint_vel = 30*3.14159/180
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

	opti = cs.Opti()

	#decision variables for robot 1
	q1 = opti.parameter(7,1)
	J1 = jac_fun_rob(q1)
	J1 = cs.vertcat(J1[0], J1[1])
	q_dot1 = opti.variable(7, 1)
	#progress variable for robot 1
	s_1 = opti.parameter()
	s_dot1 = opti.variable()

	fk_vals1 = robot.fk(q1)[6] #forward kinematics first robot

	#decision variables for robot 2
	q2 = opti.parameter(7,1)
	J2 = jac_fun_rob(q2)
	J2 = cs.vertcat(J2[0], J2[1])
	q_dot2 = opti.variable(7, 1)
	#progress variables for robot 2
	s_2 = opti.parameter()
	s_dot2 = opti.variable()

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
	opti.subject_to(-max_joint_vel <= (q_dot1 <= max_joint_vel))
	opti.subject_to(-max_joint_vel <= (q_dot2 <= max_joint_vel))
	opti.subject_to(robot.joint_lb <= (q1 + q_dot1*ts <= robot.joint_ub))
	opti.subject_to(robot.joint_lb <= (q2 + q_dot2*ts <= robot.joint_ub))

	#2nd priority: Orientation constraints are fixed, And also the TODO: obstacle avoidance

	# slack_priority2 = opti.variable(6, 1)
	slack_priority2 = opti.variable(7, 1)
	opti.subject_to(-slack_priority2[0:3] <= (J1[3:6, :]@q_dot1 <= slack_priority2[0:3]))
	opti.subject_to(-slack_priority2[3:6] <= (J2[3:6, :]@q_dot2 <= slack_priority2[3:6]))

	dist_ees = -cs.sqrt((fk_vals1[0:3,3] - fk_vals2[0:3, 3]).T@(fk_vals1[0:3,3] - fk_vals2[0:3, 3])) + 0.3
	Jac_dist_con = cs.jacobian(dist_ees, cs.vertcat(q1, q2))
	K_coll_avoid = 1
	opti.subject_to(Jac_dist_con@cs.vertcat(q_dot1, q_dot2) + K_coll_avoid*dist_ees <= slack_priority2[6])
	opti.subject_to(slack_priority2[6] >= 0)

	slp2_gains = opti.parameter(7, 1)

	#3rd highest priority. Stay on the path for both the robots
	#for robot 1
	stay_path1 = fk_vals1[0:3, 3] - traj1
	K_stay_path1 = 1
	Jac_sp1 = cs.jacobian(stay_path1, cs.vertcat(q1, s_1))
	sp1_slack = opti.variable(3,1)
	opti.subject_to(-sp1_slack <= (Jac_sp1@cs.vertcat(q_dot1, s_dot1) + K_stay_path1*stay_path1 <= sp1_slack))

	#for robot 2
	stay_path2 = fk_vals2[0:3, 3] - traj2
	K_stay_path2 = 1
	Jac_sp2 = cs.jacobian(stay_path2, cs.vertcat(q2, s_2))
	sp2_slack = opti.variable(3,1)
	opti.subject_to(-sp2_slack <= (Jac_sp2@cs.vertcat(q_dot2, s_dot2) + K_stay_path2*stay_path2 <= sp2_slack))

	slack_priority3 = cs.vertcat(sp1_slack, sp2_slack)
	slp3_gains = opti.parameter(6,1)

	#4th priority: First robot reaches the goal
	slack_priority4 = opti.variable()
	s1_dot_rate_ff = 0.25
	opti.subject_to(-slack_priority4 <= (s_dot1 - s1_dot_rate_ff <= slack_priority4))

	slp4_gains = opti.parameter()

	#5th priority: Second robot reaches the goal
	slack_priority5 = opti.variable()
	s2_dot_rate_ff = 0.5
	opti.subject_to(-slack_priority5 <= (s_dot2 - s2_dot_rate_ff <= slack_priority5))

	slp5_gains = opti.parameter()

	objective = slp3_gains.T@slack_priority3 + slp4_gains@slack_priority4 + slp2_gains.T@slack_priority2 + slp5_gains@slack_priority5
	objective += 1e-3*(q_dot1.T@q_dot1) + 1e-3*(q_dot2.T@q_dot2) + 1e-3*(s_dot1*s_dot1 + s_dot2*s_dot2) 
	# slack_q_dot = opti.variable(7,1)
	# opti.subject_to(-slack_q_dot <=  (q_dot1 <= slack_q_dot) )
	# slack_q_dot2 = opti.variable(7,1)
	# opti.subject_to(-slack_q_dot2 <=  (q_dot2 <= slack_q_dot2) )

	# objective += cs.mtimes(cs.DM([1e-3]*7).T, slack_q_dot)
	# objective += cs.mtimes(cs.DM([1e-3]*7).T, slack_q_dot2)

	#Adding bounds on the acceleration
	q_dot1_prev = opti.parameter(7, 1)
	opti.set_value(q_dot1_prev, cs.DM([0]*7))
	opti.subject_to(-max_joint_acc*ts <= (q_dot1 - q_dot1_prev <= max_joint_acc*ts))
	opti.minimize(objective)

	q_dot2_prev = opti.parameter(7, 1)
	opti.set_value(q_dot2_prev, cs.DM([0]*7))
	opti.subject_to(-max_joint_acc*ts <= (q_dot2 - q_dot2_prev <= max_joint_acc*ts))
	opti.minimize(objective)

	# #Adding the motion constraints
	# c1 = J[0:2,:]@q_dot - cs.MX([0.1, 0.045])
	# slack1 = opti.variable(2, 1)
	# opti.subject_to(-slack1 <= (c1 <= slack1))

	# c2 = J[1:4,:]@q_dot - cs.MX([0.02, 0.055, 0.0])
	# slack2 = opti.variable(3, 1)
	# opti.subject_to(-slack2 <= (c2 <= slack2))

	# c3 = J[3:6,:]@q_dot - cs.MX([0.1, 0.04, 0.03])
	# slack3 = opti.variable(3, 1)
	# opti.subject_to(-slack3 <= (c3 <= slack3))

	# slg1 = opti.parameter( 2 , 1)
	# slg2 = opti.parameter(3, 1)
	# slg3 = opti.parameter(3, 1)
	# objective = slg1.T@slack1 + slg2.T@slack2 + slg3.T@slack3 + q_dot.T@q_dot*1e-3

	p_opts = {"expand":True}

	# # s_opts = {"max_iter": 100, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-6}

	# kkt_tol_pr = 1e-6
	# kkt_tol_du = 1e-6
	# min_step_size = 1e-6
	# max_iter = 2
	# max_iter_ls = 3
	# qpsol_options = {'constr_viol_tol': kkt_tol_pr, 'dual_inf_tol': kkt_tol_du, 'verbose' : False, 'print_iter': False, 'print_header': False, 'dump_in': False, "error_on_fail" : False}
	# solver_options = {'qpsol': 'qrqp', 'qpsol_options': qpsol_options, 'verbose': False, 'tol_pr': kkt_tol_pr, 'tol_du': kkt_tol_du, 'min_step_size': min_step_size, 'max_iter': max_iter, 'max_iter_ls': max_iter_ls, 'print_iteration': True, 'print_header': False, 'print_status': False, 'print_time': True}
	opti.solver("sqpmethod", {"qpsol": 'qpoases', 'print_iteration': True, 'print_header': True, 'print_status': True, "print_time":True, 'max_iter': 1000})

	q_opt = cs.vertcat(cs.DM(q0_1), cs.DM(q0_2))
	q_opt_history = q_opt
	q_dot_opt_history = cs.DM([0]*14)
	s1_opt = cs.DM(0)
	s2_opt = cs.DM(0)
	# c1_jac = cs.jacobian(c1, q_dot)
	# c2_jac = cs.jacobian(c2, q_dot)
	# c3_jac = cs.jacobian(c3, q_dot)

	tic = time.time()
	opti.set_value(slp2_gains, [100, 100, 100, 100, 100, 100, 100])
	opti.set_value(slp3_gains, [10, 10, 10, 10, 10, 10])
	opti.set_value(slp4_gains, 5/49.9)
	opti.set_value(slp5_gains, 0.1)

	# opti.set_value(slg2, [10, 10, 10])
	# opti.set_value(slg3, [1, 1, 1])
	# x_dot = cs.DM([0, 0, 0, 0, 0, 0])

	constraint_violations = cs.DM([0, 0, 0, 0])

	#Setup for visualization
	visualizationBullet = True
	counter = 1
	if visualizationBullet:

		from tasho import world_simulator
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

	cool_off_counter = 0
	for i in range(math.ceil(T/ts)):
		counter += 1
		opti.set_value(cs.vertcat(q1, q2), q_opt.full())
		opti.set_value(s_1, s1_opt)
		opti.set_value(s_2, s2_opt)
		sol = opti.solve()

		q_dot1_sol = sol.value(q_dot1)
		q_dot2_sol = sol.value(q_dot2)
		s_dot1_sol = sol.value(s_dot1)
		s_dot2_sol = sol.value(s_dot2)

		#Computing the constraint violations
		con_viols = sol.value(cs.vertcat(slack_priority2, slack_priority3, slack_priority4, slack_priority5))
		constraint_violations = cs.horzcat(constraint_violations, cs.vertcat(cs.norm_1(con_viols[0:7]), cs.norm_1(con_viols[7:13]), cs.norm_1(con_viols[13]), cs.norm_1(con_viols[14])))
		print(con_viols)
		print(q_opt)
		print(s2_opt)

		#Update all the variables
		q_opt += ts*cs.vertcat(q_dot1_sol, q_dot2_sol)
		s1_opt += s_dot1_sol*ts
		s2_opt += s_dot2_sol*ts


		if s1_opt >=1:
			print("Robot1 reached it's goal. Terminating")
			cool_off_counter += 1
			if cool_off_counter >= 100: #ts*100 cooloff period for the robot to exactly reach it's goal
				break

		fk_vals1_sol = sol.value(fk_vals1)
		fk_vals2_sol = sol.value(fk_vals2)
		print(fk_vals1_sol)
		print(fk_vals2_sol)

		if visualizationBullet:
			obj.setController(kukaID, "velocity", joint_indices, targetVelocities = q_dot1_sol)
			obj.setController(kukaID2, "velocity", joint_indices, targetVelocities = q_dot2_sol)
			obj.run_simulation(1)

	# 	J_sol = sol.value(J)

	# 	x_dot = cs.horzcat(x_dot, cs.mtimes(J_sol, q_dot_sol))

	# 	print(x_dot[:,-1])

	# 	q_opt += q_dot_sol*ts
		q_opt_history = cs.horzcat(q_opt_history, q_opt)
		q_dot_opt_history = cs.horzcat(q_dot_opt_history, cs.vertcat(q_dot1_sol, q_dot2_sol))
	# 	c1_jac_sol = sol.value(c1_jac)
	# 	c2_jac_sol = sol.value(c2_jac)
	# 	c3_jac_sol = sol.value(c3_jac)

	# 	# print(c3_jac_sol)
	# 	# print(cs.mtimes(c3_jac_sol, c2_jac_sol.T))

		#compute the gains for the 5th priority
		alpha5 = cs.DM([1])

		#compute gains for 4th priority
		alpha4 = cs.DM(gamma*alpha5)

		#compute gains for the 3rd priority
		alpha3 = cs.DM([1, 1, 1, 1, 1, 1])
		Jac_sp1_sol = sol.value(Jac_sp1)
		Jac_sp2_sol = sol.value(Jac_sp2)
		alpha3[0] = gamma*((alpha5 + alpha4)/cs.norm_1(Jac_sp1_sol[0, :]))
		for j in range(1,3):
			alpha3[j] = alpha3[0]*cs.norm_1(Jac_sp1_sol[0,:])/cs.norm_1(Jac_sp1_sol[j,:])
		for j in range(3, 6):
			alpha3[j] = alpha3[0]*cs.norm_1(Jac_sp1_sol[0,:])/cs.norm_1(Jac_sp2_sol[j-3,:])

		#computing the gains for the next highest priority constraints
		alpha2 = cs.DM([1, 1, 1, 1, 1, 1, 1])
		J1_sol = sol.value(J1)
		J2_sol = sol.value(J2)
		alpha2[0] = gamma*(6*alpha3[0]*cs.norm_1(Jac_sp1_sol[0,:]) + alpha4 + alpha5)/cs.norm_1(J1_sol[3,:])
		for j in range(1, 3):
			alpha2[j] = alpha2[0]*cs.norm_1(J1_sol[3,:])/cs.norm_1(J1_sol[3+j,:])
		for j in range(3, 6):
			alpha2[j] = alpha2[0]*cs.norm_1(J1_sol[3,:])/cs.norm_1(J2_sol[j,:])

		Jac_dist_con_sol = sol.value(Jac_dist_con)
		alpha2[6] = alpha2[0]*cs.norm_1(J1_sol[3,:])/cs.norm_1(Jac_dist_con_sol)

		print("alphas are " + str(alpha2) + "  " + str(alpha3) + "  " + str(alpha4) + "  " + str(alpha5))
		opti.set_value(slp5_gains, alpha5)
		opti.set_value(slp4_gains, alpha4)
		opti.set_value(slp3_gains, alpha3)
		opti.set_value(slp2_gains, alpha2)

		#When bounds on acceleration
		opti.set_value(q_dot1_prev, q_dot1_sol)
		opti.set_value(q_dot2_prev, q_dot2_sol)
	# 	opti.set_value(slg2, alpha2)
	# 	opti.set_value(slg3, alpha3)
		opti.set_initial(cs.vertcat(q_dot1, q_dot2), cs.vertcat(q_dot1_sol, q_dot2_sol))
		opti.set_initial(s_dot1, s_dot1_sol)
		opti.set_initial(s_dot2, s_dot2_sol)
	# 	# print(q_opt)

	obj.end_simulation()

	time_l1opt = time.time() - tic

	# # print("Nullspace Projection took " + str(time_nsp) +"s")
	print("L1 optimization method took " + str(time_l1opt) +"s")

	# # print(q_pi_history[-1])
	# print(q_opt_history[:,-1])

	# #Implementing solution by Hierarchical QP
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
	title("Trajectory")
	xlabel("No of samples in the horizon (sampling time = 0.05s)")
	ylabel('Seconds')
	

	# figure()
	# plot(list(range(201)), x_dot[0,:].full().T, label = 'x')
	# plot(list(range(201)), x_dot[1,:].full().T, label = 'y')
	# plot(list(range(201)), x_dot[2,:].full().T, label = 'z')
	# plot(list(range(201)), x_dot[3,:].full().T, label = 'w_x')
	# plot(list(range(201)), x_dot[4,:].full().T, label = 'w_y')
	# plot(list(range(201)), x_dot[5,:].full().T, label = 'w_z')
	# legend()
	show(block=True)