# Code containing all the routines for L1-HQP

import sys, os
import casadi as cs
from casadi import pi, cos, sin
import numpy as np
import matplotlib.pyplot as plt
import math
import time
# import tkinter
from pylab import *
import copy

class hqp:
	"""
	Class contains all the functions for implemnting L1-HQP
	"""
	def __init__(self, ts = 0.005):

		self.variables_dot = [] #derivative of the variables
		self.slacks = {} #Slacks for the constraints
		self.slack_weights = {} #the weights for the slack variables
		self.constraints = {}
		self.constraints_numbers = {} #stores the correspondence between constraint id in opti.g
		#and the constraints of a particular priority.
		self.constraint_type = {}
		self.constraint_options_lb = {}
		self.constraint_options_ub = {}
		self.variables0 = [] #Expression for the current value of the variables
		self.cascadedHQP_active = False
		self.opti = cs.Opti() #creating CasADi's optistack
		self.obj = 0 #the variable to be minimized.
		self.constraint_counter = 0
		self.cHQP_optis = {}
		self.cHQP_slacks = {}
		self.once_solved = False
		self.chqp_warm_start_x = {}
		self.chqp_warm_start_lam = {}
	def activate_cascadedHQP(priority_levels):
		""" Should be activated before creating any variables or adding any constraints"""
		cHQP = []
		for i in range(priority_levels):
			cHQP.append(cs.Opti())

		self.cascadedHQP_active = True



	def create_variable(self, size, weight = 0):

		opti = self.opti
		var0 = opti.parameter(size, 1)
		var_dot = opti.variable(size, 1)
		self.variables0 = cs.vertcat(self.variables0, var0)
		self.variables_dot = cs.vertcat(self.variables_dot, var_dot)
		#Adding to the objective the regularization cost
		self.obj += weight*cs.sumsqr(var_dot)

		return var0, var_dot

	def create_parameter(self, size, weight = 0):

		opti = self.opti
		var0 = opti.parameter(size, 1)
		self.variables0 = cs.vertcat(self.variables0, var0)
		return var0

	def set_value(self, var, val):
		self.opti.set_value(var, val)

	def create_constraint(self, expression, ctype, priority = 0, options = {}):

		opti = self.opti
		shape = expression.shape[0]
		if priority >= 1:
			if priority not in self.slacks:
				self.slacks[priority] = []
				self.slack_weights[priority] = []
				self.constraints[priority] = []
				self.constraint_type[priority] = []
				self.constraint_options_lb[priority]= []
				self.constraint_options_ub[priority] = []
				self.constraints_numbers[priority] = []


			slack_var = opti.variable(shape, 1)
			self.obj += cs.sumsqr(slack_var)*1e-6*0 #regularization
			slack_weights = opti.parameter(shape, 1)

			if ctype == 'equality':
				opti.subject_to(-slack_var <= (expression <= slack_var))
				for i in range(shape):
					self.constraints_numbers[priority].append([self.constraint_counter, self.constraint_counter + 1])
					self.constraint_counter += 2

			elif ctype == 'lub':
				opti.subject_to(-slack_var + options['lb'] <= (expression <= slack_var + options['ub']))
				opti.subject_to(slack_var >= 0)
				for i in range(shape):
					self.constraints_numbers[priority].append([self.constraint_counter, self.constraint_counter + 1, self.constraint_counter + 2])
					self.constraint_counter += 3

			elif ctype == 'ub':
				opti.subject_to(expression <= slack_var + options['ub'])
				opti.subject_to(slack_var >= 0)
				for i in range(shape):
					self.constraints_numbers[priority].append([self.constraint_counter, self.constraint_counter + 1])
					self.constraint_counter += 2

			elif ctype == 'lb':
				opti.subject_to(-slack_var + options['lb'] <= expression)
				opti.subject_to(slack_var >= 0)
				for i in range(shape):
					self.constraints_numbers[priority].append([self.constraint_counter, self.constraint_counter + 1])
					self.constraint_counter += 2

			self.obj += cs.mtimes(slack_weights.T, slack_var)



			self.slacks[priority] = cs.vertcat(self.slacks[priority], slack_var)
			self.slack_weights[priority] = cs.vertcat(self.slack_weights[priority], slack_weights)
			self.constraints[priority] = cs.vertcat(self.constraints[priority], expression)
			self.constraint_type[priority] = self.constraint_type[priority] +  [ctype]*expression.shape[0]
			if 'lb' in options:
				if options['lb'].shape[0] != 1:
					self.constraint_options_lb[priority] = cs.vertcat(self.constraint_options_lb[priority], options['lb'])
				else:
					self.constraint_options_lb[priority] = cs.vertcat(self.constraint_options_lb[priority], options['lb']*expression.shape[0])
			else:
				# print(expression.shape)
				self.constraint_options_lb[priority] = cs.vertcat(self.constraint_options_lb[priority], cs.DM.ones(expression.shape[0])*(-cs.inf))

			if 'ub' in options:
				if options['ub'].shape[0] != 1:
					self.constraint_options_ub[priority] = cs.vertcat(self.constraint_options_ub[priority], options['ub'])
				else:
					self.constraint_options_ub[priority] = cs.vertcat(self.constraint_options_ub[priority], options['ub']*expression.shape[0])
			else:
				self.constraint_options_ub[priority] = cs.vertcat(self.constraint_options_ub[priority], cs.DM.ones(expression.shape[0])*(cs.inf))

		elif priority == 0:
			if priority not in self.constraints:
				self.constraints[priority] = []
				self.constraints_numbers[priority] = []
				self.constraint_type[priority] = []
				self.constraint_options_lb[priority]= []
				self.constraint_options_ub[priority] = []
			self.constraint_type[priority] = self.constraint_type[priority] +  [ctype]*expression.shape[0]
			if ctype == 'equality':
				opti.subject_to(expression == 0)
				for i in range(shape):
					self.constraints_numbers[priority].append([self.constraint_counter])
					self.constraint_counter += 1

			elif ctype == 'lub':
				opti.subject_to( options['lb'] <= (expression <= options['ub']))
				for i in range(shape):
					self.constraints_numbers[priority].append([self.constraint_counter])
					self.constraint_counter += 1

			elif ctype == 'ub':
				opti.subject_to(expression <= options['ub'])
				for i in range(shape):
					self.constraints_numbers[priority].append([self.constraint_counter])
					self.constraint_counter += 1

			elif ctype == 'lb':
				opti.subject_to(options['lb'] <= expression)
				for i in range(shape):
					self.constraints_numbers[priority].append([self.constraint_counter])
					self.constraint_counter += 1

			if 'lb' in options:
				if options['lb'].shape[0] != 1:
					self.constraint_options_lb[priority] = cs.vertcat(self.constraint_options_lb[priority], options['lb'])
				else:
					self.constraint_options_lb[priority] = cs.vertcat(self.constraint_options_lb[priority], options['lb']*expression.shape[0])
			else:
				# print(expression.shape)
				self.constraint_options_lb[priority] = cs.vertcat(self.constraint_options_lb[priority], cs.DM.ones(expression.shape[0])*(-cs.inf))

			if 'ub' in options:
				if options['ub'].shape[0] != 1:
					self.constraint_options_ub[priority] = cs.vertcat(self.constraint_options_ub[priority], options['ub'])
				else:
					self.constraint_options_ub[priority] = cs.vertcat(self.constraint_options_ub[priority], options['ub']*expression.shape[0])
			else:
				self.constraint_options_ub[priority] = cs.vertcat(self.constraint_options_ub[priority], cs.DM.ones(expression.shape[0])*(cs.inf))



			self.constraints[priority] = cs.vertcat(self.constraints[priority], expression)


	def configure(self):

		#Create functions to compute the constraint expressions
		self.number_priorities = len(self.slacks)
		self.constraint_funs = {}
		self.constraint_expression_funs = {}

		for i in range(0, self.number_priorities + 1):
			self.constraint_funs[i] = cs.Function('cons' + str(i), [self.variables0, self.variables_dot], [cs.jacobian(self.constraints[i], self.variables_dot)])
			self.constraint_expression_funs[i] = cs.Function('cons' + str(i), [self.variables0, self.variables_dot], [self.constraints[i]])
		self.opti.minimize(self.obj)

		self.inf_but_optimal = {}
		self.constraints_violated = {}
		for i in range(1, self.number_priorities+1):
			self.inf_but_optimal[i] = []
			self.constraints_violated[i] = 0

		self.gamma_initialized = False

	def setup_cascadedQP(self):
		number_priorities = len(self.slacks)
		opti = self.opti
		cHQP = {}
		cHQP_slackparams = {}
		for i in range(1, number_priorities + 1):
			#create a separate opti instance of each priority level
			opti2 = copy.deepcopy(opti)
			cHQP_slackparams[i] = {}
			# print("i is = "+ str(i))
			for j in range(1,i):
				#create parameters for the slack variables from the lower levels
				# print(self.slacks[j].shape[0])
				cHQP_slackparams[i][j] = opti2.parameter(self.slacks[j].shape[0], 1)
				opti2.subject_to(cHQP_slackparams[i][j] == self.slacks[j])
				# print("j is = " + str(j))
				#Impose constraints from all lower levels
			cHQP[i] = opti2

		self.cHQP = cHQP
		self.cHQP_slackparams = cHQP_slackparams

	def solve_cascadedQP(self, variable_values, variable_dot_values):

		sol_cqp = {}
		gain = 1000 #just set some value for the L1 penalty on task constraints
		number_priorities = len(self.slacks)
		# print(self.slacks)
		for priority in range(0, number_priorities + 1):
			# print("solving for priority level = " + str(priority))
			opti = copy.deepcopy(self.opti)
			opti.set_value(self.variables0, variable_values)
			opti.set_initial(self.variables_dot, variable_dot_values)

			#set weights for all constraints to zero
			for j in range(1, number_priorities + 1):
				opti.set_value(self.slack_weights[j], [0]*self.slack_weights[j].shape[0])

			if priority >= 1:
				#set weights only for the constraints of this particular priority level
				constraints = self.constraint_funs[priority](variable_values, variable_dot_values)
				for j in range(constraints.shape[0]):
					weight = gain/cs.norm_1(constraints[j, :])
					opti.set_value(self.slack_weights[priority][j], weight)

				#create the hard constraits for the constraints from the previous levels
				for j in range(1,priority):
					#compute solution of constraints from this level from
					#the most recent qp sol
					constraints = self.constraints[j]
					constraints_sol = sol_cqp[priority-1].value(constraints)
					constraint_type = self.constraint_type[j]
					constraint_options_ub = self.constraint_options_ub[j]
					constraint_options_lb = self.constraint_options_lb[j]
					#check if the constraint is active and make it a hard constraint
					for k in range(constraints.shape[0]):
						#if equality constraint, always active
						if constraint_type[k] == 'equality':
							opti.subject_to(constraints_sol[k] == constraints[k])
						#else check if active
						else:
							# print(constraints_sol)
							#if violated add as equality constraint
							# print("constraint sol is = " + str(constraints_sol[k]))
							if constraints_sol[k] - constraint_options_lb[k] <= -1e-6:
								opti.subject_to(constraints_sol[k] == constraints[k])
							#if just active add as inequality constraint
							else:# constraints_sol[k] - constraint_options_lb[k] <= 1e-6:
								opti.subject_to(constraint_options_lb[k] <= constraints[k])
							if constraints_sol[k] - constraint_options_ub[k] >= 1e-6:
								# print("forcing " + str(k)+"th value of " + str(j) + "th priority constraint to be equal to constraint val" + str(constraints_sol[k]))
								opti.subject_to(constraints_sol[k] == constraints[k])
							else:# constraints_sol[k] - constraint_options_ub[k] >= -1e-6:
								opti.subject_to(constraint_options_ub[k] >= constraints[k])
								#opti.subject_to(constraints_sol[k])

			#solve the QP for this priority level
			# print(opti.p.shape)
			# print(self.opti.p.shape)
			try:
				sol = opti.solve()
			except:
				return False
				break
			# cHQP[priority] = opti
			sol_cqp[priority] = sol

		return sol_cqp

	#following the convention of Kanoun 2011
	def solve_cascadedQP2(self, variable_values, variable_dot_values):

		self.time_taken = 0
		sol_cqp = {}
		gain = 1000 #just set some value for the L1 penalty on task constraints
		number_priorities = len(self.slacks)
		# print(self.slacks)
		for priority in range(0, number_priorities + 1):
			# print("solving for priority level = " + str(priority))
			opti = copy.deepcopy(self.opti)
			opti.set_value(self.variables0, variable_values)
			opti.set_initial(self.variables_dot, variable_dot_values)

			#set weights for all constraints to zero
			for j in range(1, number_priorities + 1):
				opti.set_value(self.slack_weights[j], [0]*self.slack_weights[j].shape[0])

			if priority >= 1:
				#set weights only for the constraints of this particular priority level
				constraints = self.constraint_funs[priority](variable_values, variable_dot_values)
				for j in range(constraints.shape[0]):
					weight = gain/cs.norm_1(constraints[j, :])
					opti.set_value(self.slack_weights[priority][j], weight)

				#create the hard constraits for the constraints from the previous levels
				for j in range(1,priority):
					#compute solution of constraints from this level from
					#the most recent qp sol
					constraints = self.constraints[j]
					constraints_sol = sol_cqp[priority-1].value(constraints)
					slacks_sol = sol_cqp[priority-1].value(self.slacks[j])
					constraint_type = self.constraint_type[j]
					constraint_options_ub = self.constraint_options_ub[j]
					constraint_options_lb = self.constraint_options_lb[j]
					#check if the constraint is active and make it a hard constraint
					for k in range(constraints.shape[0]):
						opti.subject_to(self.slacks[j][k] <= slacks_sol[k])

			#solve the QP for this priority level
			# print(opti.p.shape)
			# print(self.opti.p.shape)
			tic = time.time()
			try:
				sol = opti.solve()
				self.time_taken += time.time() - tic
			except:
				return False
				break
			# cHQP[priority] = opti
			sol_cqp[priority] = sol

		return sol_cqp

	def solve_cascadedQP3(self, variable_values, variable_dot_values, solver = 'qpoases'):
		blockPrint()
		sol_cqp = {}
		gain = 1000 #just set some value for the L1 penalty on task constraints
		number_priorities = len(self.slacks)
		self.time_taken = 0
		self.cHQP_xdot = {}
		casc_QP_slack_sols = {}
		# print(self.slacks)
		for priority in range(0, number_priorities + 1):
			# print("solving for priority level = " + str(priority))
			opti = cs.Opti()
			variables_dot = opti.variable(len(variable_dot_values), 1)
			variables0 = opti.parameter(len(variable_values), 1)
			opti.set_value(variables0, variable_values)
			opti.set_initial(variables_dot, variable_dot_values)

			#Adding the highest priority constraints
			constraint_expression = self.constraint_expression_funs[0](variables0, variables_dot)
			constraint_type = self.constraint_type[0]
			constraint_options_ub = self.constraint_options_ub[0]
			constraint_options_lb = self.constraint_options_lb[0]
			for k in range(constraint_expression.shape[0]):
				#if equality constraint, always active
				if constraint_type[k] == 'equality':
					opti.subject_to(constraint_expression[k] == 0)
				elif constraint_type[k] == 'lb':
					opti.subject_to(constraint_options_lb[k] <= constraint_expression[k])
				elif constraint_type[k] == 'ub':
					opti.subject_to(constraint_expression[k] <= constraint_options_ub[k])
				else:
					opti.subject_to(constraint_options_lb[k] <= (constraint_expression[k] <= constraint_options_ub[k]))

			if priority >= 1:

				#create the hard constraits for the constraints from the previous levels
				for j in range(1,priority):
					#compute solution of constraints from this level from
					#the most recent qp sol
					constraint_expression = self.constraint_expression_funs[j](variables0, variables_dot)
					constraint_type = self.constraint_type[j]
					constraint_options_ub = self.constraint_options_ub[j]
					constraint_options_lb = self.constraint_options_lb[j]
					constraints_sol = casc_QP_slack_sols[j]
					#check if the constraint is active and make it a hard constraint
					for k in range(constraint_expression.shape[0]):
						#if equality constraint, always active
						if constraint_type[k] == 'equality':
							if constraints_sol[k] < 0:
								constraints_sol[k] = -constraints_sol[k]
							opti.subject_to(-constraints_sol[k] <= (constraint_expression[k] <= constraints_sol[k]))
						#else check if active
						elif constraint_type[k] == 'lb':
							opti.subject_to(constraint_options_lb[k] - constraints_sol[k] <= constraint_expression[k])
						elif constraint_type[k] == 'ub':
							opti.subject_to(constraint_expression[k] <= constraint_options_ub[k] + constraints_sol[k])
						else:
							opti.subject_to(constraint_options_lb[k] - constraints_sol[k] <= (constraint_expression[k] <= constraint_options_ub[k] + constraints_sol[k]))

				#creating slack variables and setting constraints for this level
				constraint_expression = self.constraint_expression_funs[priority](variables0, variables_dot)
				constraint_type = self.constraint_type[priority]
				constraint_options_ub = self.constraint_options_ub[priority]
				constraint_options_lb = self.constraint_options_lb[priority]
				slacks_now = opti.variable(constraint_expression.shape[0], 1)
				opti.subject_to(slacks_now >= 0)
				for k in range(constraint_expression.shape[0]):
					#if equality constraint, always active
					if constraint_type[k] == 'equality':
						opti.subject_to(-slacks_now[k] <= (constraint_expression[k] <= slacks_now[k]))
					#else check if active
					elif constraint_type[k] == 'lb':
						opti.subject_to(constraint_options_lb[k] - slacks_now[k] <= constraint_expression[k])
					elif constraint_type[k] == 'ub':
						opti.subject_to(constraint_expression[k] <= constraint_options_ub[k] + slacks_now[k])
					else:
						opti.subject_to(constraint_options_lb[k] - slacks_now[k] <= (constraint_expression[k] <= constraint_options_ub[k] + slacks_now[k]))
				#set weights only for the constraints of this particular priority level
				constraints = self.constraint_funs[priority](variable_values, variable_dot_values)
				obj = 0
				for j in range(constraints.shape[0]):
					weight = gain/cs.norm_1(constraints[j, :])
					obj += weight*slacks_now[j]

				opti.minimize(obj + cs.sumsqr(variables_dot)*1e-6*0)

			#solve the QP for this priority level
			# print(opti.p.shape)
			# print(self.opti.p.shape)
			# enablePrint()
			if solver == 'qpoases':
				qpsol_options = {'error_on_fail':False}
				opti.solver("sqpmethod", {"expand":True, "qpsol": 'qpoases', 'qpsol_options': qpsol_options,  'print_iteration': True, 'print_header': True, 'print_status': True, "print_time":True, 'max_iter': 1000})
			else:
				opti.solver("ipopt")
			tic = time.time()
			# sol = opti.solve()
			try:

				sol = opti.solve()
				self.time_taken += sol.stats()['t_wall_total']#time.time() - tic
				if priority >= 1:
					casc_QP_slack_sols[priority] = sol.value(slacks_now)
					self.cHQP_slacks[priority] = slacks_now
					self.cHQP_xdot[priority] = variables_dot
			except:
				return False
				break
			# cHQP[priority] = opti
			sol_cqp[priority] = sol
		enablePrint()
		return sol_cqp

	def solve_cascadedQP5(self, variable_values, variable_dot_values, solver = 'qpoases', warm_start = False):
		blockPrint()
		sol_cqp = {}
		gain = 1 #just set some value for the L1 penalty on task constraints
		number_priorities = len(self.slacks)
		chqp_warm_start_x = self.chqp_warm_start_x
		chqp_warm_start_lam = self.chqp_warm_start_lam
		self.time_taken = 0
		self.cHQP_xdot = {}
		casc_QP_slack_sols = {}
		# print(self.slacks)
		for priority in range(1, number_priorities + 1):
			# print("solving for priority level = " + str(priority))
			opti = cs.Opti()
			variables_dot = opti.variable(len(variable_dot_values), 1)
			variables0 = opti.parameter(len(variable_values), 1)
			opti.set_value(variables0, variable_values)
			opti.set_initial(variables_dot, variable_dot_values)

			#Adding the highest priority constraints
			constraint_expression = self.constraint_expression_funs[0](variables0, variables_dot)
			constraint_type = self.constraint_type[0]
			constraint_options_ub = self.constraint_options_ub[0]
			constraint_options_lb = self.constraint_options_lb[0]
			for k in range(constraint_expression.shape[0]):
				#if equality constraint, always active
				if constraint_type[k] == 'equality':
					opti.subject_to(constraint_expression[k] == 0)
				elif constraint_type[k] == 'lb':
					opti.subject_to(constraint_options_lb[k] <= constraint_expression[k])
				elif constraint_type[k] == 'ub':
					opti.subject_to(constraint_expression[k] <= constraint_options_ub[k])
				else:
					opti.subject_to(constraint_options_lb[k] <= (constraint_expression[k] <= constraint_options_ub[k]))

			if priority >= 1:

				#create the hard constraits for the constraints from the previous levels
				for j in range(1,priority):
					#compute solution of constraints from this level from
					#the most recent qp sol
					constraint_expression = self.constraint_expression_funs[j](variables0, variables_dot)
					constraint_type = self.constraint_type[j]
					constraint_options_ub = self.constraint_options_ub[j]
					constraint_options_lb = self.constraint_options_lb[j]
					constraints_sol = cs.DM(casc_QP_slack_sols[j])
					slacks_now = opti.variable(constraint_expression.shape[0], 1)
					opti.subject_to(slacks_now >= 0)
					#check if the constraint is active and make it a hard constraint
					for k in range(constraint_expression.shape[0]):
						#if equality constraint, always active
						if constraint_type[k] == 'equality':
							if constraints_sol[k] < 0:
								constraints_sol[k] = -constraints_sol[k]
							opti.subject_to(-slacks_now[k] <= (constraint_expression[k] <= slacks_now[k]))
						#else check if active
						elif constraint_type[k] == 'lb':
							opti.subject_to(constraint_options_lb[k] - slacks_now[k] <= constraint_expression[k])
						elif constraint_type[k] == 'ub':
							opti.subject_to(constraint_expression[k] <= constraint_options_ub[k] + slacks_now[k])
						else:
							opti.subject_to(constraint_options_lb[k] - slacks_now[k] <= (constraint_expression[k] <= constraint_options_ub[k] + slacks_now[k]))
					opti.subject_to(np.ones(slacks_now.shape).T@slacks_now <= np.ones(slacks_now.shape).T@constraints_sol)


				#creating slack variables and setting constraints for this level
				constraint_expression = self.constraint_expression_funs[priority](variables0, variables_dot)
				constraint_type = self.constraint_type[priority]
				constraint_options_ub = self.constraint_options_ub[priority]
				constraint_options_lb = self.constraint_options_lb[priority]
				slacks_now = opti.variable(constraint_expression.shape[0], 1)
				opti.subject_to(slacks_now >= 0)
				for k in range(constraint_expression.shape[0]):
					#if equality constraint, always active
					if constraint_type[k] == 'equality':
						opti.subject_to(-slacks_now[k] <= (constraint_expression[k] <= slacks_now[k]))
					#else check if active
					elif constraint_type[k] == 'lb':
						opti.subject_to(constraint_options_lb[k] - slacks_now[k] <= constraint_expression[k])
					elif constraint_type[k] == 'ub':
						opti.subject_to(constraint_expression[k] <= constraint_options_ub[k] + slacks_now[k])
					else:
						opti.subject_to(constraint_options_lb[k] - slacks_now[k] <= (constraint_expression[k] <= constraint_options_ub[k] + slacks_now[k]))
				#set weights only for the constraints of this particular priority level
				constraints = self.constraint_funs[priority](variable_values, variable_dot_values)
				obj = 0
				for j in range(constraints.shape[0]):
					weight = gain/cs.norm_1(constraints[j, :])
					obj += weight*slacks_now[j]

				if priority == number_priorities:
					opti.minimize(obj + cs.sumsqr(variables_dot)*1e-12*0) #sufficient to add regularization only at the last level
				else:
					opti.minimize(obj)

			#solve the QP for this priority level
			# print(opti.p.shape)
			# print(self.opti.p.shape)
			# enablePrint()
			if solver == 'qpoases':
				qpsol_options = {'error_on_fail':False}
				opti.solver("sqpmethod", {"expand":True, "qpsol": 'qpoases', 'qpsol_options': qpsol_options,  'print_iteration': True, 'print_header': True, 'print_status': True, "print_time":True, 'max_iter': 1000})
			else:
				opti.solver("ipopt",{"expand":True, 'ipopt':{'tol':1e-6, 'print_level':0}})
			tic = time.time()
			# sol = opti.solve()
			if warm_start and self.once_solved:
				opti.set_initial(opti.x, chqp_warm_start_x[priority])
				opti.set_initial(opti.lam_g, chqp_warm_start_lam[priority])
			try:

				sol = opti.solve()
				if warm_start:
					chqp_warm_start_x[priority] = sol.value(opti.x)
					chqp_warm_start_lam[priority] = sol.value(opti.lam_g)
				self.time_taken += sol.stats()['t_wall_total']#time.time() - tic
				if priority >= 1:
					casc_QP_slack_sols[priority] = sol.value(slacks_now)
					self.cHQP_slacks[priority] = slacks_now
					self.cHQP_xdot[priority] = variables_dot
			except:
				return False
				break
			# cHQP[priority] = opti
			sol_cqp[priority] = sol
		enablePrint()
		self.once_solved = True
		return sol_cqp

	def solve_cascadedQP_L2(self, variable_values, variable_dot_values, solver = 'qpoases'):
		blockPrint()
		sol_cqp = {}
		gain = 1000 #just set some value for the L1 penalty on task constraints
		number_priorities = len(self.slacks)
		self.time_taken = 0
		self.cHQP_xdot = {}
		casc_QP_slack_sols = {}
		# print(self.slacks)
		for priority in range(0, number_priorities + 1):
			# print("solving for priority level = " + str(priority))
			opti = cs.Opti()
			variables_dot = opti.variable(len(variable_dot_values), 1)
			variables0 = opti.parameter(len(variable_values), 1)
			opti.set_value(variables0, variable_values)
			opti.set_initial(variables_dot, variable_dot_values)

			#Adding the highest priority constraints
			constraint_expression = self.constraint_expression_funs[0](variables0, variables_dot)
			constraint_type = self.constraint_type[0]
			constraint_options_ub = self.constraint_options_ub[0]
			constraint_options_lb = self.constraint_options_lb[0]
			for k in range(constraint_expression.shape[0]):
				#if equality constraint, always active
				if constraint_type[k] == 'equality':
					opti.subject_to(constraint_expression[k] == 0)
				elif constraint_type[k] == 'lb':
					opti.subject_to(constraint_options_lb[k] <= constraint_expression[k])
				elif constraint_type[k] == 'ub':
					opti.subject_to(constraint_expression[k] <= constraint_options_ub[k])
				else:
					opti.subject_to(constraint_options_lb[k] <= (constraint_expression[k] <= constraint_options_ub[k]))

			if priority >= 1:

				#create the hard constraits for the constraints from the previous levels
				for j in range(1,priority):
					#compute solution of constraints from this level from
					#the most recent qp sol
					constraint_expression = self.constraint_expression_funs[j](variables0, variables_dot)
					constraint_type = self.constraint_type[j]
					constraint_options_ub = self.constraint_options_ub[j]
					constraint_options_lb = self.constraint_options_lb[j]
					constraints_sol = casc_QP_slack_sols[j]
					constraints_sol = cs.DM(casc_QP_slack_sols[j])
					#check if the constraint is active and make it a hard constraint
					for k in range(constraint_expression.shape[0]):
						#if equality constraint, always active
						if constraint_type[k] == 'equality':
							if constraints_sol[k] < 0:
								constraints_sol[k] = -constraints_sol[k]
							opti.subject_to(-constraints_sol[k] <= (constraint_expression[k] <= constraints_sol[k]))
						#else check if active
						elif constraint_type[k] == 'lb':
							opti.subject_to(constraint_options_lb[k] - constraints_sol[k] <= constraint_expression[k])
						elif constraint_type[k] == 'ub':
							opti.subject_to(constraint_expression[k] <= constraint_options_ub[k] + constraints_sol[k])
						else:
							opti.subject_to(constraint_options_lb[k] - constraints_sol[k] <= (constraint_expression[k] <= constraint_options_ub[k] + constraints_sol[k]))

				#creating slack variables and setting constraints for this level
				constraint_expression = self.constraint_expression_funs[priority](variables0, variables_dot)
				constraint_type = self.constraint_type[priority]
				constraint_options_ub = self.constraint_options_ub[priority]
				constraint_options_lb = self.constraint_options_lb[priority]
				slacks_now = opti.variable(constraint_expression.shape[0], 1)
				# opti.subject_to(slacks_now >= 0)
				for k in range(constraint_expression.shape[0]):
					#if equality constraint, always active
					if constraint_type[k] == 'equality':
						opti.subject_to(-slacks_now[k] <= (constraint_expression[k] <= slacks_now[k]))
					#else check if active
					elif constraint_type[k] == 'lb':
						opti.subject_to(constraint_options_lb[k] - slacks_now[k] <= constraint_expression[k])
					elif constraint_type[k] == 'ub':
						opti.subject_to(constraint_expression[k] <= constraint_options_ub[k] + slacks_now[k])
					else:
						opti.subject_to(constraint_options_lb[k] - slacks_now[k] <= (constraint_expression[k] <= constraint_options_ub[k] + slacks_now[k]))
				#set weights only for the constraints of this particular priority level
				constraints = self.constraint_funs[priority](variable_values, variable_dot_values)
				obj = 0
				for j in range(constraints.shape[0]):
					obj += cs.sumsqr(slacks_now[j])

				# if priority == number_priorities:
				opti.minimize(obj + cs.sumsqr(variables_dot)*1e-6*0)

			#solve the QP for this priority level
			# print(opti.p.shape)
			# print(self.opti.p.shape)
			# enablePrint()
			if solver == 'qpoases':
				qpsol_options = {'error_on_fail':False}
				opti.solver("sqpmethod", {"expand":True, "qpsol": 'qpoases', 'qpsol_options': qpsol_options,  'print_iteration': True, 'print_header': True, 'print_status': True, "print_time":True, 'max_iter': 1000})
			else:
				opti.solver("ipopt",{"expand":True, 'ipopt':{'tol':1e-12, 'print_level':5}})
			tic = time.time()
			# sol = opti.solve()
			try:
				# enablePrint()
				sol = opti.solve()
				# time.sleep(1)
				# blockPrint()
				self.time_taken += sol.stats()['t_wall_total']#time.time() - tic
				if priority >= 1:
					casc_QP_slack_sols[priority] = sol.value(slacks_now)
					self.cHQP_slacks[priority] = slacks_now
					self.cHQP_xdot[priority] = variables_dot
			except:
				return False
				break
			# cHQP[priority] = opti
			sol_cqp[priority] = sol
		enablePrint()
		return sol_cqp

	# The same as cascadedQP3, but now the optimization problem is generated only once. And warm starting is used,
	# to get a good idea of the difference in the computational effort.

	def solve_cascadedQP4(self, variable_values, variable_dot_values):

		blockPrint()
		sol_cqp = {}
		gain = 1000 #just set some value for the L1 penalty on task constraints
		number_priorities = len(self.slacks)
		self.time_taken = 0
		self.cHQP_xdot = {}
		casc_QP_slack_sols = {}
		# print(self.slacks)
		for priority in range(1, number_priorities + 1):
			# print("solving for priority level = " + str(priority))
			if priority not in self.cHQP_optis:
				opti = cs.Opti()
				variables_dot = opti.variable(variable_dot_values.shape[0], 1)
				variables0 = opti.parameter(variable_values.shape[0], 1)
				opti.set_value(variables0, variable_values)
				opti.set_initial(variables_dot, variable_dot_values)

				#Adding the highest priority constraints
				constraint_expression = self.constraint_expression_funs[0](variables0, variables_dot)
				constraint_type = self.constraint_type[0]
				constraint_options_ub = self.constraint_options_ub[0]
				constraint_options_lb = self.constraint_options_lb[0]
				for k in range(constraint_expression.shape[0]):
					#if equality constraint, always active
					if constraint_type[k] == 'equality':
						opti.subject_to(constraint_expression[k] == 0)
					elif constraint_type[k] == 'lb':
						opti.subject_to(constraint_options_lb[k] <= constraint_expression[k])
					elif constraint_type[k] == 'ub':
						opti.subject_to(constraint_expression[k] <= constraint_options_ub[k])
					else:
						opti.subject_to(constraint_options_lb[k] <= (constraint_expression[k] <= constraint_options_ub[k]))

				constraints_sols_params_dict = {} #store the MX symbolic variables for the slack variable relaxation parameters
				if priority >= 1:

					#create the hard constraits for the constraints from the previous levels

					for j in range(1,priority):
						#compute solution of constraints from this level from
						#the most recent qp sol
						constraint_expression = self.constraint_expression_funs[j](variables0, variables_dot)
						constraint_type = self.constraint_type[j]
						constraint_options_ub = self.constraint_options_ub[j]
						constraint_options_lb = self.constraint_options_lb[j]
						casc_QP_slack_sols[j] = cs.DM(casc_QP_slack_sols[j])
						constraints_sol_params = opti.parameter(cs.DM(casc_QP_slack_sols[j]).shape[0])
						constraints_sols_params_dict[j] = constraints_sol_params
						#check if the constraint is active and make it a hard constraint
						for k in range(constraint_expression.shape[0]):
							#if equality constraint, always active
							if constraint_type[k] == 'equality':
								opti.subject_to(-constraints_sol_params[k] <= (constraint_expression[k] <= constraints_sol_params[k]))
							#else check if active
							elif constraint_type[k] == 'lb':
								opti.subject_to(constraint_options_lb[k] - constraints_sol_params[k] <= constraint_expression[k])
							elif constraint_type[k] == 'ub':
								opti.subject_to(constraint_expression[k] <= constraint_options_ub[k] + constraints_sol_params[k])
							else:
								opti.subject_to(constraint_options_lb[k] - constraints_sol_params[k] <= (constraint_expression[k] <= constraint_options_ub[k] + constraints_sol_params[k]))

					#creating slack variables and setting constraints for this level
					constraint_expression = self.constraint_expression_funs[priority](variables0, variables_dot)
					constraint_type = self.constraint_type[priority]
					constraint_options_ub = self.constraint_options_ub[priority]
					constraint_options_lb = self.constraint_options_lb[priority]
					slacks_now = opti.variable(constraint_expression.shape[0], 1)
					opti.subject_to(slacks_now >= 0)
					for k in range(constraint_expression.shape[0]):
						#if equality constraint, always active
						if constraint_type[k] == 'equality':
							opti.subject_to(-slacks_now[k] <= (constraint_expression[k] <= slacks_now[k]))
						#else check if active
						elif constraint_type[k] == 'lb':
							opti.subject_to(constraint_options_lb[k] - slacks_now[k] <= constraint_expression[k])
						elif constraint_type[k] == 'ub':
							opti.subject_to(constraint_expression[k] <= constraint_options_ub[k] + slacks_now[k])
						else:
							opti.subject_to(constraint_options_lb[k] - slacks_now[k] <= (constraint_expression[k] <= constraint_options_ub[k] + slacks_now[k]))

					#set weights only for the constraints of this particular priority level
					constraints = self.constraint_funs[priority](variable_values, variable_dot_values)
					obj = 0
					for j in range(constraints.shape[0]):
						weight = gain/cs.norm_1(constraints[j, :])
						obj += 1000*slacks_now[j]

					opti.minimize(obj + cs.sumsqr(variables_dot)*1e-6)
					#solve the QP for this priority level
					# print(opti.p.shape)
					# print(self.opti.p.shape)
					# enablePrint()
				# p_opts = {"expand":True}
				# s_opts = {"max_iter": 100, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-6}
				# opti.solver("ipopt", p_opts, s_opts)
				qpsol_options = {'error_on_fail':False}
				opti.solver("sqpmethod", {"expand":True, "qpsol": 'qpoases', 'qpsol_options': qpsol_options,  'print_iteration': True, 'print_header': True, 'print_status': True, "print_time":True, 'max_iter': 1000})

				self.cHQP_optis[priority] = [opti, slacks_now, constraints_sols_params_dict, variables_dot]

			enablePrint()
			print("\n\n\n Solving for priority level " + str(priority) + " \n\n\n")
			opti = self.cHQP_optis[priority][0]
			variables_dot = self.cHQP_optis[priority][3]
			slacks_now = self.cHQP_optis[priority][1]

			if priority >= 2:
				constraints_sols_params_dict = self.cHQP_optis[priority][2]
				for j in range(1,priority):
					opti.set_value(constraints_sols_params_dict[j], casc_QP_slack_sols[j])
			tic = time.time()
			sol = opti.solve()
			self.time_taken += time.time() - tic
			if priority >= 1:
				casc_QP_slack_sols[priority] = sol.value(slacks_now)
				self.cHQP_slacks[priority] = slacks_now
				self.cHQP_xdot[priority] = variables_dot
			blockPrint()

			try:
				opti = self.cHQP_optis[priority][0]
				variables_dot = self.cHQP_optis[priority][3]
				slacks_now = self.cHQP_optis[priority][1]
				#set weights only for the constraints of this particular priority level
				constraints = self.constraint_funs[priority](variable_values, variable_dot_values)
				obj = 0
				for j in range(constraints.shape[0]):
					weight = gain/cs.norm_1(constraints[j, :])
					obj += weight*slacks_now[j]

				opti.minimize(obj + cs.sumsqr(variables_dot)*1e-6)
				if priority >= 2:
					constraints_sols_params_dict = self.cHQP_optis[priority][2]
					for j in constraints_sols_params_dict:
						opti.set_value(constraints_sols_params_dict[j], casc_QP_slack_sols[j])
				tic = time.time()
				# sol = opti.solve()
				self.time_taken += time.time() - tic
				if priority >= 1:
					slacks_now = self.cHQP_optis[priority][1]
					casc_QP_slack_sols[priority] = cs.fabs(sol.value(slacks_now))
					self.cHQP_slacks[priority] = slacks_now
					self.cHQP_xdot[priority] = variables_dot
			except:
				return False
				break
			# cHQP[priority] = opti
			sol_cqp[priority] = sol

		enablePrint()
		return sol_cqp, self.cHQP_optis



	def solve_HQPl1(self, variable_values, variable_dot_values, gamma_init = None, gamma_limit = None):

		opti = self.opti
		opti.set_value(self.variables0, variable_values)
		opti.set_initial(self.variables_dot, variable_dot_values)

		#compute slack gains
		# enablePrint()
		gain_least_priority = 1
		if gamma_init == None:
			gamma = [0.25]*(self.number_priorities-1)
		else:
			if isinstance(gamma_init, float):
				gamma = [gamma_init]*(self.number_priorities-1)
			else:
				gamma = gamma_init
		cumulative_weight = 0
		for i in range(0,self.number_priorities):
			priority = self.number_priorities - i
			constraints = self.constraint_funs[priority](variable_values, variable_dot_values)
			if i == 0:
				for j in range(constraints.shape[0]):
					weight = gain_least_priority/cs.norm_1(constraints[j, :])
					# print(weight)
					cumulative_weight += weight*cs.norm_1(constraints[j, :])
					opti.set_value(self.slack_weights[priority][j], weight)
			else:
				gain = gamma[priority-1]*cumulative_weight
				for j in range(constraints.shape[0]):
					weight = gain/cs.norm_1(constraints[j, :])
					# print(weight)
					cumulative_weight += weight*cs.norm_1(constraints[j, :])
					# print("weight at priority_level " + str(priority) + " is " + str(weight))
					opti.set_value(self.slack_weights[priority][j], weight)

		if gamma_limit != None:
			for i in range(gamma_limit, self.number_priorities):
				opti.set_value(self.slack_weights[i+1], 0)
		# opti.minimize(opti.f) #TODOOO: remove, added just to eliminate warm start
		print("Cumulative weight is " + str(cumulative_weight))
		blockPrint()
		# enablePrint()
		tic = time.time()
		# sol = opti.solve()
		try:
			# opti.minimize(opti.f) #to prevent warmstart. REMOVE!!!!
			sol = opti.solve()
			self.time_taken += sol.stats()['t_wall_total'] # toc
			tewosdlkjf = 1
		except:
			sol = False
		toc = time.time() - tic

		# blockPrint()
		# enablePrint()

		# print(sol.value(opti.lam_g))
		# Jg = sol.value(cs.jacobian(opti.g, opti.x))
		# Jf = sol.value(cs.jacobian(opti.f, opti.x))
		# print(Jg.toarray())
		# print( Jf.T)
		# print("Constraint Jacobian shape is:" + str(Jg.shape))
		# print("Constraint numbers are : " + str(self.constraints_numbers))
		# opti2 = copy.deepcopy(opti)
		# opti2.set_value(self.variables0, variable_values)
		# p_opts = {"expand":True}
		# s_opts = {"max_iter": 100}#, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-6}
		# opti2.solver('ipopt', p_opts, s_opts)
		# opti2.solve()
		#TODO: add something that checks the satisfaction of the hierarchy
		# enablePrint()
		return sol

	#A method to adaptively change the values of gamma to better deal with the conditioning issues
	#and to detect and eliminate hierarchy violation
	def solve_adaptive_hqp(self, variable_values, variable_dot_values, gamma_init = None):
		opti = self.opti
		self.time_taken = 0
		gamma_init = [gamma_init]*(self.number_priorities-1)
		hierarcy_failed = True
		loop_counter = 0
		hierarchy_failure = {}
		while hierarcy_failed and loop_counter < 5:
			loop_counter += 1
			sol = self.solve_HQPl1(variable_values, variable_dot_values, gamma_init)
			if not sol:
				return sol, hierarchy_failure
			#check if the hierarchy fails
			#compute the constraint Jacobian
			Jg = sol.value(cs.jacobian(opti.g, opti.x))
			#compute the gradient of the objective
			Jf = sol.value(cs.jacobian(opti.f, opti.x))
			#compute the Lagrange multipliers
			lam_g = sol.value(opti.lam_g)
			infeasible_constraints = {}
			#compute grad_i: the sum of all gradients t.e.m the ith priority level
			grad_i = {}
			g_infeasible = {}
			for i in range(0, self.number_priorities):
				if i == 0:
					grad_i[i] = Jf.T
				else:
					grad_i[i] = grad_i[i-1]
				for constraints_numbers in self.constraints_numbers[i]:
					for c in constraints_numbers:
						grad_i[i] += Jg[c]*lam_g[c]

				#Find out which constriants are infeasible at each priority level
				pl = self.number_priorities - i #starting from the lowest priority constraint
				con_viols = sol.value(self.slacks[pl])
				print(con_viols)
				con_violated = con_viols >= 1e-7 #boolean array signifying which constraints are violated
				print(con_violated)
				con_violated = [j for j, s in enumerate(con_violated) if s]
				infeasible_constraints[pl] = con_violated

				#computing g_infeasible
				if i == 0:
					g_infeasible[pl] = np.array([0]*opti.x.shape[0])
				else:
					g_infeasible[pl] = g_infeasible[pl + 1]
				for con in con_violated:
					for c in self.constraints_numbers[pl][con]:
						g_infeasible[pl] += Jg[c]*lam_g[c]
			print("Infeasible constriants are " + str(infeasible_constraints))
			#Detect failure of hierarchy using the Lagrange multipliers
			hierarchy_failure = {} #stores which constraint at which level failed
			for pl in range(1, self.number_priorities):
				for c in infeasible_constraints[pl]:

					#compute the residual of gradient of Lagrangian
					grad = Jf*0
					for con in self.constraints_numbers[pl][c]:
						grad += Jg[con]*lam_g[con]
					# print("Gradient of the constraint" + str((pl, c)) + ":" + str(grad))
					# print("grad_i + g_infeasible : " + str(grad_i[pl] + g_infeasible[pl + 1]))
					residual = (grad_i[pl] + g_infeasible[pl + 1])@grad.T/cs.norm_1(grad)
					# print("Corresponding residual :" + str(residual))
					residual2 = (grad_i[pl])@grad.T/cs.norm_1(grad)
					# residual = (grad_i[pl] + g_infeasible[pl + 1])[0:self.variables_dot.shape[0]] / cs.norm_1(grad_i[pl] - grad_i[pl-1])
					print("Corresponding relative residual :" + str(cs.norm_1(residual)))
					print("Residual agnostic to feasibility of lower cons:" + str(cs.norm_1(residual2)/cs.norm_1(grad)))
					if cs.norm_1(residual)/cs.norm_1(grad) >= 1e-5:
					# if cs.norm_1(residual) >= 1e-3:
						hierarchy_failure[(pl, c)] = True


			# print(grad_i)
			# print(g_infeasible)
			print("Hierarchy failure is : " + str(hierarchy_failure))
			if len(hierarchy_failure) == 0:
				hierarcy_failed = False
			else:
				keys = hierarchy_failure.keys()
				for k in keys:
					gamma_init[k[0]-1] = 5*gamma_init[k[0]-1]
		enablePrint()
		print("Total time taken adaptive HQP 1 = " + str(self.time_taken))
		blockPrint()
		return sol, hierarchy_failure

	#An adaptive HQP method that uses a computationally demanding method to check the failure of hierarchy
	#with the idea that it eliminates false negatives
	def solve_adaptive_hqp2(self, variable_values, variable_dot_values, gamma_init = None):
		opti = self.opti
		self.time_taken = 0
		gamma_init = [gamma_init]*(self.number_priorities-1)
		hierarcy_failed = True
		once_ran = False
		loop_counter = 0
		hierarchy_failure = {}
		while hierarcy_failed and loop_counter < 10:
			enablePrint()
			loop_counter += 1
			print("loop counter is " + str(loop_counter))
			blockPrint()
			sol = self.solve_HQPl1(variable_values, variable_dot_values, gamma_init)
			if not sol:
				if once_ran:
					return sol_old, hierarchy_failure
				else:
					return sol, hierarchy_failure
			else:
				once_ran = True
				sol_old = sol
			#check if the hierarchy fails
			#compute the constraint Jacobian
			Jg = sol.value(cs.jacobian(opti.g, opti.x))
			#compute the gradient of the objective
			Jf = sol.value(cs.jacobian(opti.f, opti.x))
			#compute the Lagrange multipliers
			lam_g = sol.value(opti.lam_g)
			infeasible_constraints = {}

			#Compute all the gradients of the constraints upto a priority level.
			# Differentiated also between equality and inequality constraints
			# (because the sign matters for the inequality constriants)
			eq_grad = {}
			ineq_grad = {}
			# enablePrint()
			for i in range(0, self.number_priorities):
				# print("Priority level " + str(i))
				if i == 0:
					eq_grad[i] = []
					ineq_grad[i] = []
				else:
					eq_grad[i] = eq_grad[i-1]
					ineq_grad[i] = ineq_grad[i-1]
				constraint_type = self.constraint_type[i]
				j = 0
				for constraints_numbers in self.constraints_numbers[i]:
					grad = Jf*0
					for c in constraints_numbers:
						grad += Jg[c]*lam_g[c]
					grad = grad / (cs.norm_1(grad) + 1e-3)
					if constraint_type[j] == 'equality':
						eq_grad[i] = cs.vertcat(eq_grad[i], grad)
						# print(eq_grad)
					else:
						ineq_grad[i] = cs.vertcat(ineq_grad[i], grad)
						# print(ineq_grad)
					j += 1

				#Find out which constriants are infeasible at each priority level
				pl = self.number_priorities - i #starting from the lowest priority constraint
				con_viols = sol.value(self.slacks[pl])
				# print(con_viols)
				con_violated = con_viols >= 1e-7 #boolean array signifying which constraints are violated
				# print(con_violated)
				con_violated = [j for j, s in enumerate(con_violated) if s]
				infeasible_constraints[pl] = con_violated
			# print("Printting the first eq_grad")
			# print(eq_grad[0])
			# print("Infeasible constriants are " + str(infeasible_constraints))
			#Detect failure of hierarchy using the Lagrange multipliers
			hierarchy_failure = {} #stores which constraint at which level failed
			for pl in range(1, self.number_priorities):
				# for c in infeasible_constraints[pl]:
				if len(infeasible_constraints[pl]) > 0:
					#compute the residual of gradient of Lagrangian
					# grad = Jf*0
					# for con in self.constraints_numbers[pl][c]:
					# 	grad += Jg[con]*lam_g[con]
					# grad = grad / (cs.norm_1(grad) + 1e-3)
					# print("Gradient of the constraint" + str((pl, c)) + ":" + str(grad))
					# print("grad_i + g_infeasible : " + str(grad_i[pl] + g_infeasible[pl + 1]))

					grad = Jf*0
					for constraints_numbers in self.constraints_numbers[pl]:
						for c2 in constraints_numbers:
							grad += Jg[c2]*lam_g[c2]

					# eq_con_mat_feas_now = []
					# grad = Jf*0
					# for c in range(len(self.constraints_numbers[pl])):
					# 	if c in infeasible_constraints[pl]:
					# 	# for c2 in constraints_numbers:
					# 		for con in self.constraints_numbers[pl][c]:
					# 			# grad += Jg[c2]*lam_g[c2]
					# 			grad += Jg[con]*lam_g[con]

					# 	else:
					# 		grad_temp = Jf*0
					# 		for con in self.constraints_numbers[pl][c]:
					# 			# grad += Jg[c2]*lam_g[c2]
					# 			grad_temp += Jg[con]*lam_g[con]
					# 		grad_temp = grad_temp / (cs.norm_1(grad_temp) + 1e-3)
					# 		eq_con_mat_feas_now = cs.vertcat(eq_con_mat_feas_now, grad_temp)

					# j = 0
					# grad = Jf*0
					# eq_con_mat_feas_now = []
					# ineq_con_mat_feas_now = []
					# for constraints_numbers in self.constraints_numbers[pl]:
					# 	grad_temp = Jf*0
					# 	for c in constraints_numbers:
					# 		grad_temp += Jg[c]*lam_g[c]
					# 	grad_temp = grad_temp / (cs.norm_1(grad_temp) + 1e-3)
					# 	constraint_type = self.constraint_type[pl]
					# 	if constraint_type[j] == 'equality':
					# 		if j in infeasible_constraints[pl]:
					# 			grad += grad_temp.T
					# 		else:
					# 			eq_con_mat_feas_now = cs.vertcat(eq_con_mat_feas_now, grad_temp)
					# 	else:
					# 		if j in infeasible_constraints[pl]:
					# 			grad += grad_temp.T
					# 		else:
					# 			ineq_con_mat_feas_now = cs.vertcat(ineq_con_mat_feas_now, grad_temp)
					# 	j += 1

					grad = grad / (cs.norm_1(grad) + 1e-3)
					# print("Gradient of the constraint" + str((pl, c)) + ":" + str(grad))

					#Now creating a QP to check if the infeasibility can be caused solely by geq priority constraints
					eq_con_mat = eq_grad[pl - 1]
					ineq_con_mat = ineq_grad[pl - 1]

					# eq_con_mat = cs.vertcat(eq_con_mat, eq_con_mat_feas_now)
					# ineq_con_mat = cs.vertcat(ineq_con_mat, ineq_con_mat_feas_now)
					#Adding to these matrices, other constraints of the same priority level
					# j = 0
					# for constraints_numbers in self.constraints_numbers[pl]:
					# 	if j != c:
					# 		grad2 = Jf*0
					# 		for c2 in constraints_numbers:
					# 			grad2 += Jg[c2]*lam_g[c2]
					# 		grad2 = grad2 / (cs.norm_1(grad2) + 1e-3)
					# 		if constraint_type[j] == 'equality':
					# 			eq_con_mat = cs.vertcat(eq_con_mat, grad2)
					# 		else:
					# 			ineq_con_mat = cs.vertcat(ineq_con_mat, grad2)
					# 	j += 1
					# print("Ineq_con_mat is " + str(ineq_con_mat))
					# print("Eq_con_mat is " + str(eq_con_mat))
					#setting up the QP
					opti_ver = cs.Opti()
					lam = opti_ver.variable(eq_con_mat.shape[0])
					mu = opti_ver.variable(ineq_con_mat.shape[0])
					proj = eq_con_mat.T@lam + ineq_con_mat.T@mu
					n = self.variables_dot.shape[0]
					# print("Variable dot length = " + str(n))
					slack_ver = opti_ver.variable(n, 1)
					opti_ver.subject_to(-slack_ver <= (proj[0:n] + grad[0:n].T <= slack_ver))
					proj_err_orig = cs.DM.ones(1,n)@slack_ver
					proj_err = proj_err_orig + cs.sumsqr(lam)*1e-6*0 + cs.DM.ones(mu.shape[0]).T@mu*1e-6*0
					opti_ver.minimize(proj_err)
					opti_ver.subject_to(mu >= 0)
					blockPrint()
					# p_opts = {"expand":True}
					# s_opts = {"max_iter": 100, 'tol':1e-12}#, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5}
					# opti_ver.solver('ipopt', p_opts, s_opts)
					qpsol_options = {'error_on_fail':False, 'terminationTolerance': 1e-9}
					opti_ver.solver("sqpmethod", {"expand":True, "qpsol": 'qpoases', 'qpsol_options': qpsol_options, 'tol_pr': 1e-9, 'tol_du': 1e-9, 'print_iteration': False, 'print_header': True, 'print_status': True, "print_time":True, 'max_iter': 1000})
					# enablePrint()
					# sol_ver = opti_ver.solve()
					tic = time.time()
					try:
						sol_ver = opti_ver.solve()
						lkjhsdfjkl = 1
					except:
						sol = False
						break
					toc = time.time() - tic
					self.time_taken += toc
					blockPrint()
					# enablePrint()
					proj_err_sol = sol_ver.value(proj_err_orig)
					print("Proj err sol is = " + str(proj_err_sol))

					if proj_err_sol >= 1e-4:
						hierarchy_failure[(pl, 0)] = True


			# print(grad_i)
			# print(g_infeasible)
			print("Hierarchy failure is : " + str(hierarchy_failure))
			if len(hierarchy_failure) == 0:
				hierarcy_failed = False
			else:
				keys = hierarchy_failure.keys()
				for k in keys:
					gamma_init[k[0]-1] = 5*gamma_init[k[0]-1]

		enablePrint()
		print(gamma_init)
		print("Total time taken adaptive HQP2 = " + str(self.time_taken))
		blockPrint()
		return sol, hierarchy_failure

	#simply solve the sequential step with appropriate warm starting
	def solve_adaptive_hqp3(self, variable_values, variable_dot_values, gamma_init = None, iter_lim = 10):
		opti = self.opti
		self.time_taken = 0
		if not self.gamma_initialized:
			gamma_init = [gamma_init]*(self.number_priorities-1)
			self.gamma_initialized = True
			self.gamma_init = gamma_init
		else:
			gamma_init = self.gamma_init

		hierarcy_failed = True
		once_ran = False
		loop_counter = 0
		hierarchy_failure = {}
		infeasible_constraints = {}
		constraint_violations = {}
		while hierarcy_failed and loop_counter < iter_lim:
			enablePrint()
			loop_counter += 1
			print("loop counter is " + str(loop_counter))
			blockPrint()
			sol = self.solve_HQPl1(variable_values, variable_dot_values, gamma_init)
			if not sol:
				if once_ran:
					return sol_old, hierarchy_failure
				else:
					return sol, hierarchy_failure
			else:
				once_ran = True
				sol_old = sol
			#check if the hierarchy fails
			variable_dot_values = sol.value(self.variables_dot)

			for i in range(0, self.number_priorities):
				# print("Priority level " + str(i))

				#Find out which constriants are infeasible at each priority level
				pl = self.number_priorities - i #starting from the lowest priority constraint
				con_viols = sol.value(self.slacks[pl])
				# print(con_viols)
				con_violated = con_viols >= 1e-4 #boolean array signifying which constraints are violated

				if not isinstance(con_violated, np.ndarray):
					print(type(con_violated))
					con_violated = [con_violated]
				print(con_violated)
				con_violated = [j for j, s in enumerate(con_violated) if s]
				infeasible_constraints[pl] = con_violated
				constraint_violations[pl] = con_viols

			hierarchy_failure = {} #stores which constraint at which level failed
			for pl in range(1, self.number_priorities):
				# for c in infeasible_constraints[pl]:
				if len(infeasible_constraints[pl]) > 0:

					#if the violated constraints differ from infeasible but optimal constraints
					if not infeasible_constraints[pl] == self.inf_but_optimal[pl] or cs.norm_1(constraint_violations[pl])/(cs.norm_1(self.constraints_violated[pl]) + 1e-3) >= 2.0:
						print("pl level " + str(pl))
						print("Constraint violations from weighted")
						print(constraint_violations[pl])
						gamma_temp = copy.deepcopy(gamma_init)
						for p in range(pl + 1, self.number_priorities):
							gamma_temp[p-1] = 0 #set all the weights for lower priority constraints to zero
						print(gamma_temp)
						# sol_test = self.solve_HQPl1(variable_values, variable_dot_values, gamma_temp)
						sol_test = self.solve_HQPl1(variable_values, variable_dot_values, gamma_init, gamma_limit = pl)
						con_viol_test = np.array(sol_test.value(self.slacks[pl]))
						for pl_h in range(1,pl):
							if cs.norm_1(sol_test.value(self.slacks[pl_h])) >= cs.norm_1(constraint_violations[pl_h]) + 1e-6:
								gamma_init[pl_h] *= 5
						print("constraint violations from test")
						print(con_viol_test)
						con_violated = con_viol_test >= 1e-4 #boolean array signifying which constraints are violated
						if not isinstance(con_violated, np.ndarray):
							print(type(con_violated))
							con_violated = [con_violated]
						print(con_violated)
						con_violated = [j for j, s in enumerate(con_violated) if s]

						if cs.norm_1(constraint_violations[pl]) - cs.norm_1(con_viol_test) <= 1e-6:
							self.inf_but_optimal[pl] = infeasible_constraints[pl]
							self.constraints_violated[pl] = constraint_violations[pl]
						else:
							print(pl)
							gamma_init[pl-1] = 5*gamma_init[pl-1]
							self.gamma_init = gamma_init
							hierarchy_failure[pl] = True
							break

				else:
					self.inf_but_optimal[pl] = infeasible_constraints[pl]
					self.constraints_violated[pl] = constraint_violations[pl]

			if len(hierarchy_failure) == 0:
				hierarcy_failed = False
			print("Hierarchy failure is : " + str(hierarchy_failure))

		enablePrint()
		print(gamma_init)
		print("Total time taken adaptive HQP2 = " + str(self.time_taken))
		blockPrint()
		return sol, hierarchy_failure

# Disable
def blockPrint():
	# sys.stdout = sys.__stdout__
	sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
	sys.stdout = sys.__stdout__
