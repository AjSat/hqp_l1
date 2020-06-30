# Code containing all the routines for L1-HQP

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

class hqp:
	"""
	Class contains all the functions for implemnting L1-HQP
	"""
	def __init__(self, ts = 0.005):

		self.variables_dot = [] #derivative of the variables
		self.slacks = {} #Slacks for the constraints
		self.slack_weights = {} #the weights for the slack variables
		self.constraints = {}
		self.variables0 = [] #Expression for the current value of the variables

		self.opti = cs.Opti() #creating CasADi's optistack
		self.obj = 0 #the variable to be minimized.

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

	def create_constraint(self, expression, type, priority = 0, options = {}):

		opti = self.opti
		if priority >= 1:
			shape = expression.shape[0]
			slack_var = opti.variable(shape, 1)
			slack_weights = opti.parameter(shape, 1)

			if type == 'equality':
				opti.subject_to(-slack_var <= (expression <= slack_var))

			elif type == 'lub':
				opti.subject_to(-slack_var + options['lb'] <= (expression <= slack_var + options['ub']))
				opti.subject_to(slack_var >= 0)

			elif type == 'ub':
				opti.subject_to(expression <= slack_var + options['ub'])
				opti.subject_to(slack_var >= 0)

			elif type == 'lb':
				opti.subject_to(-slack_var + options['lb'] <= expression)
				opti.subject_to(slack_var >= 0)

			self.obj += cs.mtimes(slack_weights.T, slack_var)

			if priority not in self.slacks:
				self.slacks[priority] = []
				self.slack_weights[priority] = []
				self.constraints[priority] = []

			self.slacks[priority] = cs.vertcat(self.slacks[priority], slack_var)
			self.slack_weights[priority] = cs.vertcat(self.slack_weights[priority], slack_weights)
			self.constraints[priority] = cs.vertcat(self.constraints[priority], expression)

		elif priority == 0:
			if type == 'equality':
				opti.subject_to(expression == 0)

			elif type == 'lub':
				opti.subject_to( options['lb'] <= (expression <= options['ub']))

			elif type == 'ub':
				opti.subject_to(expression <= options['ub'])

			elif type == 'lb':
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
	def solve_cascadedQP(self):

		print("Not implemented")

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
		#TODO: add something that checks the satisfaction of the hierarchy
		return sol

		print("Not implemented")
