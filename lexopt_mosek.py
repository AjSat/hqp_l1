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
from mosek.fusion import *

class lexopt_mosek:
	"""
	Class contains all the functions for implementing HLP
	"""
	def __init__(self, ts = 0.005):

		self.variables_dot = [] #derivative of the variables
		self.lb = [] #variable_dot lower bounds
		self.ub = [] #variable_dot upper bounds
		self.variables0 = [] #Expression for the current value of the variables
		self.constraints = {} #dictionary of constraints at priority levels
		self.opti = cs.Opti() #creating CasADi's optistack
		self.once_solved = False
		self.number_priority_levels = 0

	def create_variable(self, size, lb, ub):

		opti = self.opti
		var0 = opti.parameter(size, 1)
		var_dot = opti.variable(size, 1)
		self.variables0 = cs.vertcat(self.variables0, var0)
		self.variables_dot = cs.vertcat(self.variables_dot, var_dot)
		self.lb = cs.vertcat(self.lb, lb)
		self.ub = cs.vertcat(self.ub, ub)

		return var0, var_dot

	def create_parameter(self, size):

		opti = self.opti
		var0 = opti.parameter(size, 1)
		self.variables0 = cs.vertcat(self.variables0, var0)
		return var0

	def create_constraint(self, expression, ctype, priority = 0, options = {}):

		opti = self.opti
		if priority >= self.number_priority_levels:
			self.number_priority_levels = priority + 1
		#create a dictionary in self.constraints for this priority level if not
		#already present
		if priority not in self.constraints:
			self.constraints[priority] = {}

		constraints = self.constraints[priority]
		A = cs.jacobian(expression, self.variables_dot)

		if ctype == 'lub':

			if 'lub' not in constraints:
				constraints['lub'] = {}
				constraints['lub']['A'] = []
				constraints['lub']['lb'] = []
				constraints['lub']['ub'] = []

			constraints['lub']['A'] = cs.vertcat(constraints['lub']['A'], A)
			constraints['lub']['ub'] = cs.vertcat(constraints['lub']['ub'], options['ub'])
			constraints['lub']['lb'] = cs.vertcat(constraints['lub']['lb'], options['lb'])

		elif ctype == 'ub':

			if 'ub' not in constraints:
				constraints['ub'] = {}
				constraints['ub']['A'] = []
				constraints['ub']['ub'] = []

			constraints['ub']['A'] = cs.vertcat(constraints['ub']['A'], A)
			constraints['ub']['ub'] = cs.vertcat(constraints['ub']['ub'], options['ub'])

		elif ctype == 'eq':

			if 'eq' not in constraints:
				constraints['eq'] = {}
				constraints['eq']['A'] = []
				constraints['eq']['b'] = []

			constraints['eq']['A'] = cs.vertcat(constraints['eq']['A'], A)
			constraints['eq']['b'] = cs.vertcat(constraints['eq']['b'], options['b'])

		else:
			raise Exception("Unknown constraint type added")

	#Creates casadi functions of the constraints and bounds that take
	#variables_dot and variables_0 as inputs
	def configure_constraints(self):

		self.lb = self.lb.full().T[0]
		self.ub = self.ub.full().T[0]
		self.n = self.variables_dot.shape[0]
		for priority in self.constraints:

			constraints = self.constraints[priority]

			if 'lub' in constraints:
				constraints['lub']['A_fun'] = cs.Function('lub_A_'+str(priority), [self.variables_dot, self.variables0], [constraints['lub']['A']])
				constraints['lub']['lb_fun'] = cs.Function('lub_lb_'+str(priority), [self.variables_dot, self.variables0], [constraints['lub']['lb']])
				constraints['lub']['ub_fun'] = cs.Function('lub_ub_'+str(priority), [self.variables_dot, self.variables0], [constraints['lub']['ub']])

			if 'ub' in constraints:
				constraints['ub']['A_fun'] = cs.Function('ub_A_'+str(priority), [self.variables_dot, self.variables0], [constraints['ub']['A']])
				constraints['ub']['ub_fun'] = cs.Function('ub_ub_'+str(priority), [self.variables_dot, self.variables0], [constraints['ub']['ub']])

			if 'eq' in constraints:
				constraints['eq']['A_fun'] = cs.Function('eq_A_'+str(priority), [self.variables_dot, self.variables0], [constraints['eq']['A']])
				constraints['eq']['b_fun'] = cs.Function('eq_b_'+str(priority), [self.variables_dot, self.variables0], [constraints['eq']['b']])

	#Compute the values of the matrices for a given instance
	def compute_matrices(self, variables_dot, variables0):

		for priority in self.constraints:

			constraints = self.constraints[priority]

			if 'lub' in constraints:
				constraints['lub']['A_vals'] = constraints['lub']['A_fun'](variables_dot, variables0).full()
				constraints['lub']['lb_vals'] = constraints['lub']['lb_fun'](variables_dot, variables0).full()
				constraints['lub']['ub_vals'] = constraints['lub']['ub_fun'](variables_dot, variables0).full()

			if 'ub' in constraints:
				constraints['ub']['A_vals'] = constraints['ub']['A_fun'](variables_dot, variables0).full()
				constraints['ub']['ub_vals'] = constraints['ub']['ub_fun'](variables_dot, variables0).full()

			if 'eq' in constraints:
				constraints['eq']['A_vals'] = constraints['eq']['A_fun'](variables_dot,variables0).full()
				constraints['eq']['b_vals'] = constraints['eq']['b_fun'](variables_dot, variables0).full()

	def configure_weighted_problem(self):

		Mw = Model()
		Mw_dict = {}
		x = Mw.variable("x", self.variables_dot.shape[0], Domain.inRange(self.lb, self.ub))
		Mw_dict['x'] = x
		obj = 0
		for priority in self.constraints:

			constraints = self.constraints[priority]
			Mw_dict[priority] = {}

			if 'lub' in constraints:
				m = constraints['lub']['A_vals'].shape[0]
				Mw_dict[priority]['lub_slack'] = Mw.variable(str(priority)+"_w_lub_slack", m, Domain.greaterThan(0.0))
				Mw_dict[priority]['lub_slack_weight'] =  Mw.parameter(str(priority)+"_w_lub_slack_weight", m)
				Mw_dict[priority]['lub_ub'] = Mw.parameter(str(priority)+"_lub_ub", m)
				Mw_dict[priority]['lub_lb'] = Mw.parameter(str(priority)+"_lub_lb", m)
				Mw_dict[priority]['lub_A'] = Mw.parameter(str(priority)+"_lub_A", [m, self.n])
				Mw.constraint(str(priority)+"_lub_con1", Expr.sub(Expr.sub(Expr.mul(Mw_dict[priority]['lub_A'], x), Mw_dict[priority]['lub_ub']), Mw_dict[priority]['lub_slack']), Domain.lessThan(0))
				Mw.constraint(str(priority)+"_lub_con2", Expr.add(Expr.sub(Expr.mul(Mw_dict[priority]['lub_A'], x), Mw_dict[priority]['lub_lb']), Mw_dict[priority]['lub_slack']), Domain.greaterThan(0))
				obj = Expr.add(obj, Expr.dot(Mw_dict[priority]['lub_slack'],Mw_dict[priority]['lub_slack_weight']))

			if 'ub' in constraints:
				m = constraints['ub']['A_vals'].shape[0]
				Mw_dict[priority]['ub_slack'] = Mw.variable(str(priority)+"_w_ub_slack", m, Domain.greaterThan(0.0))
				Mw_dict[priority]['ub_slack_weight'] =  Mw.parameter(str(priority)+"_w_ub_slack_weight", m)
				Mw_dict[priority]['ub_ub'] = Mw.parameter(str(priority)+"_ub_ub", m)
				Mw_dict[priority]['ub_A'] = Mw.parameter(str(priority)+"_ub_A", [m, self.n])
				Mw.constraint(str(priority)+"_ub_con", Expr.sub(Expr.sub(Expr.mul(Mw_dict[priority]['ub_A'], x), Mw_dict[priority]['ub_ub']), Mw_dict[priority]['ub_slack']), Domain.lessThan(0))
				obj = Expr.add(obj, Expr.dot(Mw_dict[priority]['ub_slack'],Mw_dict[priority]['ub_slack_weight']))

			if 'eq' in constraints:
				m = constraints['eq']['A_vals'].shape[0]
				Mw_dict[priority]['eq_slack'] = Mw.variable(str(priority)+"_w_eq_slack", m, Domain.greaterThan(0.0))
				Mw_dict[priority]['eq_slack_weight'] =  Mw.parameter(str(priority)+"_w_eq_slack_weight", m)
				Mw_dict[priority]['eq_b'] = Mw.parameter(str(priority)+"_eq_b", m)
				Mw_dict[priority]['eq_A'] = Mw.parameter(str(priority)+"_eq_A", [m, self.n])
				Mw.constraint(str(priority)+"_eq_con1", Expr.sub(Expr.sub(Expr.mul(Mw_dict[priority]['eq_A'], x), Mw_dict[priority]['eq_b']), Mw_dict[priority]['eq_slack']), Domain.lessThan(0))
				Mw.constraint(str(priority)+"_eq_con2", Expr.add(Expr.sub(Expr.mul(Mw_dict[priority]['eq_A'], x), Mw_dict[priority]['eq_b']), Mw_dict[priority]['eq_slack']), Domain.greaterThan(0))
				obj = Expr.add(obj, Expr.dot(Mw_dict[priority]['eq_slack'],Mw_dict[priority]['eq_slack_weight']))

		Mw.objective(ObjectiveSense.Minimize, obj)
		Mw.setSolverParam('optimizer', 'primalSimplex')
		Mw.setSolverParam("numThreads", 1)
		Mw.setSolverParam('simHotstart', 'statusKeys')
		Mw.setSolverParam("basisRelTolS", 1.0e-4)
		Mw.setSolverParam("simDegen", "none")
		Mw.setSolverParam("simNonSingular", "off")
		Mw.setSolverParam("presolveUse", "off")
		self.Mw = Mw
		self.Mw_dict = Mw_dict


	def solve_weighted_method(self, weights):

		#set values
		for priority in range(1, self.number_priority_levels):

			constraints = self.constraints[priority]
			dict_p = self.Mw_dict[priority]

			if 'lub' in constraints:
				m = constraints['lub']['A_vals'].shape[0]
				dict_p['lub_slack_weight'].setValue(np.ones(m)*weights[priority-1])
				dict_p['lub_ub'].setValue(constraints['lub']['ub_vals'][0])
				dict_p['lub_lb'].setValue(constraints['lub']['lb_vals'][0])
				dict_p['lub_A'].setValue(constraints['lub']['A_vals'])

			if 'ub' in constraints:
				m = constraints['ub']['A_vals'].shape[0]
				dict_p['ub_slack_weight'].setValue(np.ones(m)*weights[priority-1])
				dict_p['ub_ub'].setValue(constraints['ub']['ub_vals'][0])
				dict_p['ub_A'].setValue(constraints['ub']['A_vals'])

			if 'eq' in constraints:
				m = constraints['eq']['A_vals'].shape[0]
				dict_p['eq_slack_weight'].setValue(np.ones(m)*weights[priority-1])
				dict_p['eq_b'].setValue(constraints['eq']['b_vals'][0])
				dict_p['eq_A'].setValue(constraints['eq']['A_vals'])

		self.Mw.solve()

if __name__ == '__main__':

	hlp = lexopt_mosek()
	x0, x = hlp.create_variable(2, cs.vcat([-5, -5]), cs.vcat([5, 5]))
	# z0, z = hlp.create_variable(1, cs.vcat([-5]), cs.vcat([5]))
	hlp.create_constraint(x[0]*0.707 - x[1]*0.707, 'ub', 1, {'ub':0})
	hlp.create_constraint(x[1], 'eq', 2, {'b':0})
	hlp.create_constraint(-x[0], 'lub', 3, {'ub':-1, 'lb':-2})
	hlp.configure_constraints()
	hlp.compute_matrices([0,0],[1,1])
	hlp.configure_weighted_problem()
	hlp.solve_weighted_method([5,2,1])
	hlp.Mw.solve()
	print(hlp.Mw.getSolverDoubleInfo("simTime"))
	print(hlp.Mw_dict['x'].level())
	print(hlp.Mw_dict[2]['eq_slack'].level())
	print(hlp.Mw.primalObjValue())
	print(hlp.constraints)
	# print(hlp.constraints[0]['ub']['A_vals'])
	print("No syntax errors")
