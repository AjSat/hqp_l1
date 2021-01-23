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
		self.time_taken = 0

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
		a0 = expression
		if ctype == 'lub':

			if 'lub' not in constraints:
				constraints['lub'] = {}
				constraints['lub']['A'] = []
				constraints['lub']['lb'] = []
				constraints['lub']['ub'] = []

			constraints['lub']['A'] = cs.vertcat(constraints['lub']['A'], A)
			constraints['lub']['ub'] = cs.vertcat(constraints['lub']['ub'], options['ub'] - a0)
			constraints['lub']['lb'] = cs.vertcat(constraints['lub']['lb'], options['lb']- a0)

		elif ctype == 'ub':

			if 'ub' not in constraints:
				constraints['ub'] = {}
				constraints['ub']['A'] = []
				constraints['ub']['ub'] = []

			constraints['ub']['A'] = cs.vertcat(constraints['ub']['A'], A)
			constraints['ub']['ub'] = cs.vertcat(constraints['ub']['ub'], options['ub']- a0)

		elif ctype == 'eq':

			if 'eq' not in constraints:
				constraints['eq'] = {}
				constraints['eq']['A'] = []
				constraints['eq']['b'] = []

			constraints['eq']['A'] = cs.vertcat(constraints['eq']['A'], A)
			constraints['eq']['b'] = cs.vertcat(constraints['eq']['b'], options['b']- a0)

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
			constraints["m_k"] = 0
			if 'lub' in constraints:
				constraints['lub']['A_fun'] = cs.Function('lub_A_'+str(priority), [self.variables_dot, self.variables0], [constraints['lub']['A']])
				constraints['lub']['lb_fun'] = cs.Function('lub_lb_'+str(priority), [self.variables_dot, self.variables0], [constraints['lub']['lb']])
				constraints['lub']['ub_fun'] = cs.Function('lub_ub_'+str(priority), [self.variables_dot, self.variables0], [constraints['lub']['ub']])
				constraints["m_k"] += constraints['lub']['A'].shape[0]

			if 'ub' in constraints:
				constraints['ub']['A_fun'] = cs.Function('ub_A_'+str(priority), [self.variables_dot, self.variables0], [constraints['ub']['A']])
				constraints['ub']['ub_fun'] = cs.Function('ub_ub_'+str(priority), [self.variables_dot, self.variables0], [constraints['ub']['ub']])
				constraints["m_k"] += constraints['ub']['A'].shape[0]

			if 'eq' in constraints:
				constraints['eq']['A_fun'] = cs.Function('eq_A_'+str(priority), [self.variables_dot, self.variables0], [constraints['eq']['A']])
				constraints['eq']['b_fun'] = cs.Function('eq_b_'+str(priority), [self.variables_dot, self.variables0], [constraints['eq']['b']])
				constraints["m_k"] += constraints['eq']['A'].shape[0]

	#Compute the values of the matrices for a given instance
	def compute_matrices(self, variables_dot, variables0):

		for priority in self.constraints:

			constraints = self.constraints[priority]

			if 'lub' in constraints:
				A_vals = constraints['lub']['A_fun'](variables_dot, variables0)
				lb_vals = constraints['lub']['lb_fun'](variables_dot, variables0)
				ub_vals = constraints['lub']['ub_fun'](variables_dot, variables0)
				for i in range(A_vals.shape[0]):
					row_vec_norm = cs.norm_1(A_vals[i,:])
					lb_vals[i] /= row_vec_norm
					ub_vals[i] /= row_vec_norm
					for j in range(A_vals.shape[1]):
						A_vals[i,j] /= row_vec_norm

				constraints['lub']['A_vals'] = A_vals.full()
				constraints['lub']['lb_vals'] = lb_vals.full().ravel()
				constraints['lub']['ub_vals'] = ub_vals.full().ravel()

			if 'ub' in constraints:

				A_vals = constraints['ub']['A_fun'](variables_dot, variables0)
				ub_vals = constraints['ub']['ub_fun'](variables_dot, variables0)
				for i in range(A_vals.shape[0]):
					row_vec_norm = cs.norm_1(A_vals[i,:])
					ub_vals[i] /= row_vec_norm
					for j in range(A_vals.shape[1]):
						A_vals[i,j] /= row_vec_norm

				constraints['ub']['A_vals'] = A_vals.full()
				constraints['ub']['ub_vals'] = ub_vals.full().ravel()

			if 'eq' in constraints:

				A_vals = constraints['eq']['A_fun'](variables_dot, variables0)
				b_vals = constraints['eq']['b_fun'](variables_dot, variables0)
				for i in range(A_vals.shape[0]):
					row_vec_norm = cs.norm_1(A_vals[i,:])
					b_vals[i] /= row_vec_norm
					for j in range(A_vals.shape[1]):
						A_vals[i,j] /= row_vec_norm

				constraints['eq']['A_vals'] = A_vals.full()
				constraints['eq']['b_vals'] = b_vals.full().ravel()

	def configure_sequential_problem(self):

		M_seq = {}
		Mdict_seq = {}
		for spriority in range(1, self.number_priority_levels):
			M_seq[spriority] = Model()
			Mdict_seq[spriority] = {}
			Mw = M_seq[spriority]
			Mw_dict = Mdict_seq[spriority]
			x = Mw.variable("x", self.variables_dot.shape[0], Domain.inRange(self.lb, self.ub))
			Mw_dict['x'] = x

			for priority in range(spriority+1):

				constraints = self.constraints[priority]
				Mw_dict[priority] = {}
				if priority >= 1:
					obj_level = 0
					if 'lub' in constraints:
						m = constraints['lub']['A_vals'].shape[0]
						Mw_dict[priority]['lub_slack'] = Mw.variable(str(priority)+"_w_lub_slack", m, Domain.greaterThan(0.0))
						Mw_dict[priority]['lub_ub'] = Mw.parameter(str(priority)+"_lub_ub", m)
						Mw_dict[priority]['lub_lb'] = Mw.parameter(str(priority)+"_lub_lb", m)
						Mw_dict[priority]['lub_A'] = Mw.parameter(str(priority)+"_lub_A", [m, self.n])
						Mw.constraint(str(priority)+"_lub_con1", Expr.sub(Expr.sub(Expr.mul(Mw_dict[priority]['lub_A'], x), Mw_dict[priority]['lub_ub']), Mw_dict[priority]['lub_slack']), Domain.lessThan(0))
						Mw.constraint(str(priority)+"_lub_con2", Expr.add(Expr.sub(Expr.mul(Mw_dict[priority]['lub_A'], x), Mw_dict[priority]['lub_lb']), Mw_dict[priority]['lub_slack']), Domain.greaterThan(0))
						obj_level = Expr.add(obj_level, Expr.sum(Mw_dict[priority]['lub_slack']))

					if 'ub' in constraints:
						m = constraints['ub']['A_vals'].shape[0]
						Mw_dict[priority]['ub_slack'] = Mw.variable(str(priority)+"_w_ub_slack", m, Domain.greaterThan(0.0))
						Mw_dict[priority]['ub_ub'] = Mw.parameter(str(priority)+"_ub_ub", m)
						Mw_dict[priority]['ub_A'] = Mw.parameter(str(priority)+"_ub_A", [m, self.n])
						Mw.constraint(str(priority)+"_ub_con", Expr.sub(Expr.sub(Expr.mul(Mw_dict[priority]['ub_A'], x), Mw_dict[priority]['ub_ub']), Mw_dict[priority]['ub_slack']), Domain.lessThan(0))
						obj_level = Expr.add(obj_level, Expr.sum(Mw_dict[priority]['ub_slack']))

					if 'eq' in constraints:
						m = constraints['eq']['A_vals'].shape[0]
						Mw_dict[priority]['eq_slack'] = Mw.variable(str(priority)+"_w_eq_slack", m, Domain.greaterThan(0.0))
						Mw_dict[priority]['eq_b'] = Mw.parameter(str(priority)+"_eq_b", m)
						Mw_dict[priority]['eq_A'] = Mw.parameter(str(priority)+"_eq_A", [m, self.n])
						Mw_dict[priority][str(priority)+"_eq_con1"] = Mw.constraint(str(priority)+"_eq_con1", Expr.sub(Expr.sub(Expr.mul(Mw_dict[priority]['eq_A'], x), Mw_dict[priority]['eq_b']), Mw_dict[priority]['eq_slack']), Domain.lessThan(0))
						Mw_dict[priority][str(priority)+"_eq_con2"] = Mw.constraint(str(priority)+"_eq_con2", Expr.add(Expr.sub(Expr.mul(Mw_dict[priority]['eq_A'], x), Mw_dict[priority]['eq_b']), Mw_dict[priority]['eq_slack']), Domain.greaterThan(0))
						obj_level = Expr.add(obj_level, Expr.sum(Mw_dict[priority]['eq_slack']))

					if priority < spriority:
						Mw_dict[priority]['optimal_slack'] = Mw.parameter(str(priority)+"_opt_slack", 1)
						Mw.constraint(str(priority)+"_slack_limit", Expr.sub(obj_level, Mw_dict[priority]['optimal_slack']), Domain.lessThan(0))

				if priority == 0:
					if 'lub' in constraints:
						m = constraints['lub']['A_vals'].shape[0]
						Mw_dict[priority]['lub_ub'] = Mw.parameter(str(priority)+"_lub_ub", m)
						Mw_dict[priority]['lub_lb'] = Mw.parameter(str(priority)+"_lub_lb", m)
						Mw_dict[priority]['lub_A'] = Mw.parameter(str(priority)+"_lub_A", [m, self.n])
						Mw.constraint(str(priority)+"_lub_con1", Expr.sub(Expr.mul(Mw_dict[priority]['lub_A'], x), Mw_dict[priority]['lub_ub']), Domain.lessThan(0))
						Mw.constraint(str(priority)+"_lub_con2", Expr.sub(Expr.mul(Mw_dict[priority]['lub_A'], x), Mw_dict[priority]['lub_lb']), Domain.greaterThan(0))

					if 'eq' in constraints:
						m = constraints['eq']['A_vals'].shape[0]
						Mw_dict[priority]['eq_b'] = Mw.parameter(str(priority)+"_eq_b", m)
						Mw_dict[priority]['eq_A'] = Mw.parameter(str(priority)+"_eq_A", [m, self.n])
						Mw_dict[priority][str(priority)+"_eq_con1"] = Mw.constraint(str(priority)+"_eq_con1", Expr.sub(Expr.mul(Mw_dict[priority]['eq_A'], x), Mw_dict[priority]['eq_b']), Domain.lessThan(0))
						Mw_dict[priority][str(priority)+"_eq_con2"] = Mw.constraint(str(priority)+"_eq_con2", Expr.sub(Expr.mul(Mw_dict[priority]['eq_A'], x), Mw_dict[priority]['eq_b']), Domain.greaterThan(0))

				if priority == spriority:
					Mw.objective(ObjectiveSense.Minimize, obj_level)
					Mw.setSolverParam('optimizer', 'freeSimplex')
					Mw.setSolverParam("numThreads", 1)
					Mw.setSolverParam('simHotstart', 'statusKeys')
					Mw.setSolverParam("basisRelTolS", 1.0e-4)
					Mw.setSolverParam("simDegen", "none")
					Mw.setSolverParam("simNonSingular", "off")
					Mw.setSolverParam("presolveUse", "off")
					Mw.breakSolver()

		self.M_seq = M_seq
		self.Mdict_seq = Mdict_seq

	def solve_sequential_problem(self):

		self.time_taken = 0
		optimal_slacks = {}
		for spriority in range(1, self.number_priority_levels):

			for priority in range(spriority + 1):
				constraints = self.constraints[priority]
				dict_p = self.Mdict_seq[spriority][priority]

				if 'lub' in constraints:
					m = constraints['lub']['A_vals'].shape[0]
					dict_p['lub_ub'].setValue(constraints['lub']['ub_vals'])
					dict_p['lub_lb'].setValue(constraints['lub']['lb_vals'])
					dict_p['lub_A'].setValue(constraints['lub']['A_vals'])

				if 'ub' in constraints:
					m = constraints['ub']['A_vals'].shape[0]
					dict_p['ub_ub'].setValue(constraints['ub']['ub_vals'])
					dict_p['ub_A'].setValue(constraints['ub']['A_vals'])

				if 'eq' in constraints:
					m = constraints['eq']['A_vals'].shape[0]
					dict_p['eq_b'].setValue(constraints['eq']['b_vals'])
					dict_p['eq_A'].setValue(constraints['eq']['A_vals'])

				if priority > 0 and priority < spriority:
					dict_p["optimal_slack"].setValue(optimal_slacks[priority])

			self.M_seq[spriority].solve()
			self.time_taken += self.M_seq[spriority].getSolverDoubleInfo("simPrimalTime") + self.M_seq[spriority].getSolverDoubleInfo("simDualTime")
			optimal_slacks[spriority] = self.M_seq[spriority].primalObjValue()

		self.optimal_slacks = optimal_slacks

	def configure_weighted_problem(self):

		Mw = Model()
		Mw_dict = {}
		x = Mw.variable("x", self.variables_dot.shape[0], Domain.inRange(self.lb, self.ub))
		Mw_dict['x'] = x
		obj = 0
		for priority in self.constraints:

			constraints = self.constraints[priority]
			Mw_dict[priority] = {}
			if priority >= 1:
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
					Mw_dict[priority][str(priority)+"_eq_con1"] = Mw.constraint(str(priority)+"_eq_con1", Expr.sub(Expr.sub(Expr.mul(Mw_dict[priority]['eq_A'], x), Mw_dict[priority]['eq_b']), Mw_dict[priority]['eq_slack']), Domain.lessThan(0))
					Mw_dict[priority][str(priority)+"_eq_con2"] = Mw.constraint(str(priority)+"_eq_con2", Expr.add(Expr.sub(Expr.mul(Mw_dict[priority]['eq_A'], x), Mw_dict[priority]['eq_b']), Mw_dict[priority]['eq_slack']), Domain.greaterThan(0))
					obj = Expr.add(obj, Expr.dot(Mw_dict[priority]['eq_slack'],Mw_dict[priority]['eq_slack_weight']))

			if priority == 0:
				if 'lub' in constraints:
					m = constraints['lub']['A_vals'].shape[0]
					Mw_dict[priority]['lub_ub'] = Mw.parameter(str(priority)+"_lub_ub", m)
					Mw_dict[priority]['lub_lb'] = Mw.parameter(str(priority)+"_lub_lb", m)
					Mw_dict[priority]['lub_A'] = Mw.parameter(str(priority)+"_lub_A", [m, self.n])
					Mw.constraint(str(priority)+"_lub_con1", Expr.sub(Expr.mul(Mw_dict[priority]['lub_A'], x), Mw_dict[priority]['lub_ub']), Domain.lessThan(0))
					Mw.constraint(str(priority)+"_lub_con2", Expr.sub(Expr.mul(Mw_dict[priority]['lub_A'], x), Mw_dict[priority]['lub_lb']), Domain.greaterThan(0))

				if 'eq' in constraints:
					m = constraints['eq']['A_vals'].shape[0]
					Mw_dict[priority]['eq_b'] = Mw.parameter(str(priority)+"_eq_b", m)
					Mw_dict[priority]['eq_A'] = Mw.parameter(str(priority)+"_eq_A", [m, self.n])
					Mw_dict[priority][str(priority)+"_eq_con1"] = Mw.constraint(str(priority)+"_eq_con1", Expr.sub(Expr.mul(Mw_dict[priority]['eq_A'], x), Mw_dict[priority]['eq_b']), Domain.equalsTo(0))

		Mw.objective(ObjectiveSense.Minimize, obj)
		Mw.setSolverParam('optimizer', 'freeSimplex')
		Mw.setSolverParam("numThreads", 4)
		Mw.setSolverParam('simHotstart', 'statusKeys')
		Mw.setSolverParam("basisRelTolS", 1.0e-4)
		Mw.setSolverParam("simDegen", "none")
		Mw.setSolverParam("simNonSingular", "off")
		Mw.setSolverParam("presolveUse", "off")
		Mw.breakSolver()
		self.Mw = Mw
		self.Mw_dict = Mw_dict


	def solve_weighted_method(self, weights):

		#solve for all the epsilons (absolute weights)
		number_priorities = self.number_priority_levels
		epsilons = [1]
		summation_counter = self.constraints[number_priorities - 1]["m_k"]
		for i in range(1, number_priorities-1):
			epsilons.insert(0, summation_counter*weights[number_priorities - 2 - i])
			summation_counter += epsilons[0]*self.constraints[number_priorities - i - 1]["m_k"]
		#set values
		for priority in range(self.number_priority_levels):
			constraints = self.constraints[priority]
			dict_p = self.Mw_dict[priority]

			if 'lub' in constraints:
				m = constraints['lub']['A_vals'].shape[0]
				if priority >= 1:
					dict_p['lub_slack_weight'].setValue(np.ones(m)*epsilons[priority-1])
				dict_p['lub_ub'].setValue(constraints['lub']['ub_vals'])
				dict_p['lub_lb'].setValue(constraints['lub']['lb_vals'])
				dict_p['lub_A'].setValue(constraints['lub']['A_vals'])

			if 'ub' in constraints:
				m = constraints['ub']['A_vals'].shape[0]
				if priority >= 1:
					dict_p['ub_slack_weight'].setValue(np.ones(m)*epsilons[priority-1])
				dict_p['ub_ub'].setValue(constraints['ub']['ub_vals'])
				dict_p['ub_A'].setValue(constraints['ub']['A_vals'])

			if 'eq' in constraints:
				m = constraints['eq']['A_vals'].shape[0]
				if priority >= 1:
					dict_p['eq_slack_weight'].setValue(np.ones(m)*epsilons[priority-1])
				dict_p['eq_b'].setValue(constraints['eq']['b_vals'])
				dict_p['eq_A'].setValue(constraints['eq']['A_vals'])

		self.epsilons = epsilons
		print(epsilons)
		self.Mw.solve()
		self.time_taken += self.Mw.getSolverDoubleInfo("simPrimalTime") + self.Mw.getSolverDoubleInfo("simDualTime")

if __name__ == '__main__':

	hlp = lexopt_mosek()
	x0, x = hlp.create_variable(2, cs.vcat([-5, -5]), cs.vcat([5, 5]))
	# z0, z = hlp.create_variable(1, cs.vcat([-5]), cs.vcat([5]))
	hlp.create_constraint(x[0]*0.707 - x[1]*0.707, 'lub', 0, {'ub':0, 'lb':0})
	hlp.create_constraint(x[1], 'eq', 1, {'b':0})
	hlp.create_constraint(-x[0], 'lub', 2, {'ub':-1, 'lb':-2})
	hlp.configure_constraints()
	hlp.compute_matrices([0,0],[1,1])
	hlp.configure_weighted_problem()
	hlp.configure_sequential_problem()
	hlp.solve_sequential_problem()
	hlp.solve_weighted_method([5,2,1])
	hlp.solve_weighted_method([5,2,1])
	print(hlp.Mw.getSolverDoubleInfo("optimizerTime"))
	print(hlp.Mdict_seq[2]['x'].level())
	print(hlp.Mw_dict['x'].level())
	print(hlp.Mw_dict[2]['lub_slack'].level())
	print(hlp.Mw.primalObjValue())
	print(hlp.constraints)
	# print(hlp.constraints[0]['ub']['A_vals'])
	print("No syntax errors")
