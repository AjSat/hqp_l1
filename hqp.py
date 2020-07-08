# Code containing all the routines for L1-HQP

import sys, os
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
		self.constraint_type = {}
		self.constraint_options_lb = {}
		self.constraint_options_ub = {}
		self.variables0 = [] #Expression for the current value of the variables
		self.cascadedHQP_active = False
		self.opti = cs.Opti() #creating CasADi's optistack
		self.obj = 0 #the variable to be minimized.

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

	def set_value(self, var, val):
		self.opti.set_value(var, val)

	def create_constraint(self, expression, ctype, priority = 0, options = {}):

		opti = self.opti
		if priority >= 1:
			shape = expression.shape[0]
			slack_var = opti.variable(shape, 1)
			slack_weights = opti.parameter(shape, 1)

			if ctype == 'equality':
				opti.subject_to(-slack_var <= (expression <= slack_var))

			elif ctype == 'lub':
				opti.subject_to(-slack_var + options['lb'] <= (expression <= slack_var + options['ub']))
				opti.subject_to(slack_var >= 0)

			elif ctype == 'ub':
				opti.subject_to(expression <= slack_var + options['ub'])
				opti.subject_to(slack_var >= 0)

			elif ctype == 'lb':
				opti.subject_to(-slack_var + options['lb'] <= expression)
				opti.subject_to(slack_var >= 0)

			self.obj += cs.mtimes(slack_weights.T, slack_var)

			if priority not in self.slacks:
				self.slacks[priority] = []
				self.slack_weights[priority] = []
				self.constraints[priority] = []
				self.constraint_type[priority] = []
				self.constraint_options_lb[priority]= []
				self.constraint_options_ub[priority] = []

			self.slacks[priority] = cs.vertcat(self.slacks[priority], slack_var)
			self.slack_weights[priority] = cs.vertcat(self.slack_weights[priority], slack_weights)
			self.constraints[priority] = cs.vertcat(self.constraints[priority], expression)
			self.constraint_type[priority].append((expression.shape[0], ctype))
			if 'lb' in options:
				self.constraint_options_lb[priority] = cs.vertcat(self.constraint_options_lb[priority], options['lb'])
			else:
				print(expression.shape)
				self.constraint_options_lb[priority] = cs.vertcat(self.constraint_options_lb[priority], cs.DM.zeros(expression.shape[0]))

			if 'ub' in options:
				self.constraint_options_ub[priority] = cs.vertcat(self.constraint_options_ub[priority], options['ub'])
			else:
				self.constraint_options_ub[priority] = cs.vertcat(self.constraint_options_ub[priority], cs.MX([0]*expression.shape[0]))

		elif priority == 0:
			if ctype == 'equality':
				opti.subject_to(expression == 0)

			elif ctype == 'lub':
				opti.subject_to( options['lb'] <= (expression <= options['ub']))

			elif ctype == 'ub':
				opti.subject_to(expression <= options['ub'])

			elif ctype == 'lb':
				opti.subject_to(options['lb'] <= expression)

			if priority not in self.constraints:
				self.constraints[priority] = []

			self.constraints[priority] = cs.vertcat(self.constraints[priority], expression)


	def configure(self):

		#Create functions to compute the constraint expressions
		self.number_priorities = len(self.slacks)
		self.constraint_funs = {}

		for i in range(0, self.number_priorities + 1):
			self.constraint_funs[i] = cs.Function('cons' + str(i), [self.variables0, self.variables_dot], [cs.jacobian(self.constraints[i], self.variables_dot)])

		self.opti.minimize(self.obj)

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
		gain = 1 #just set some value for the L1 penalty on task constraints
		number_priorities = len(self.slacks)
		# print(self.slacks)
		for priority in range(1, number_priorities + 1):
			print("solving for priority level = " + str(priority))
			opti = self.cHQP[priority] #loading the opti instance for the first priority level
			opti.set_value(self.variables0, variable_values)
			opti.set_initial(self.variables_dot, variable_dot_values)

			#set weights for all constraints to zero
			for j in range(1, number_priorities + 1):
				opti.set_value(self.slack_weights[j], [0]*self.slack_weights[j].shape[0])
			#set weights only for the constraints of this particular priority level
			constraints = self.constraint_funs[priority](variable_values, variable_dot_values)
			for j in range(constraints.shape[0]):
				weight = gain/cs.norm_1(constraints[j, :])
				# opti.set_value(self.slack_weights[priority][j], weight)

			#set the values of the slack variables of the previous priority levels
			#with the solution from the previous QP
			for j in range(1, priority):
				#obtain the slack weights from the solution of the previous QP
				sol_previous_qp = sol_cqp[priority - 1].value(self.slacks[j])
				opti.set_value(self.cHQP_slackparams[priority][j], sol_previous_qp)

			#solve the QP for this priority level
			print(opti.p.shape)
			print(self.opti.p.shape)
			blockPrint()
			sol = opti.solve()
			enablePrint()
			sol_cqp[priority] = sol

		return sol_cqp
			


	def solve_HQPl1(self, variable_values, variable_dot_values):

		opti = self.opti
		opti.set_value(self.variables0, variable_values)
		opti.set_initial(self.variables_dot, variable_dot_values)

		#compute slack gains
		gain_least_priority = 1
		gamma = [0.25]*(self.number_priorities-1)
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
					opti.set_value(self.slack_weights[priority][j], weight)
		print("Cumulative weight is")
		print(cumulative_weight)
		sol = opti.solve()

		
		# opti2 = copy.deepcopy(opti)
		# opti2.set_value(self.variables0, variable_values)
		# p_opts = {"expand":True}
		# s_opts = {"max_iter": 100}#, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-6}
		# opti2.solver('ipopt', p_opts, s_opts)
		# opti2.solve()
		#TODO: add something that checks the satisfaction of the hierarchy
		return sol

		print("Not implemented")

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
