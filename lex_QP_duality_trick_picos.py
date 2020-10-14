# Code containing all the routines for L1-HQP

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import matplotlib.pyplot as plt
# import tkinter
from pylab import *
import copy
import hqp as hqp_p
import casadi as cs
import  unittest
from picos import *

 	
if __name__ == '__main__':

	print("Main function")
	#Now comparing LexLP with the sequential method for a significantly larger problem
	failures = 0
	tot_time_seq = 0
	tot_time_dt = 0
	for rand_seed in range(0,100):
		n = 30
		total_ineq_constraints = 15
		ineq_con_rank = 5
		pl = 5
		con_per_level = int(total_ineq_constraints/pl)
		n_ineq_per_level = int(total_ineq_constraints / pl)
		ineq_last_level = n_ineq_per_level + (total_ineq_constraints - n_ineq_per_level*pl)
		
		print("Using random seed " + str(rand_seed))
		np.random.seed(rand_seed) 
	
		
		
		A_ineq_all = cs.DM(np.random.randn(ineq_con_rank, n))
		A_ineq_extra = (A_ineq_all.T@cs.DM(np.random.randn(ineq_con_rank, total_ineq_constraints - ineq_con_rank))).T
		A_ineq_all = cs.vertcat(A_ineq_all, A_ineq_extra)
		A_ineq_all = A_ineq_all.full()
		np.random.shuffle(A_ineq_all)
		b_ineq_all = np.random.randn(total_ineq_constraints)

		row_ineqvec_norm = []
		for i in range(A_ineq_all.shape[0]):
			row_ineqvec_norm.append(cs.norm_1(A_ineq_all[i,:])) 
			b_ineq_all[i] /= row_ineqvec_norm[i]
			for j in range(A_ineq_all.shape[1]):
				A_ineq_all[i,j] = A_ineq_all[i,j]/row_ineqvec_norm[i]
		
		A_ineq = {}
		b_upper = {}
	
		#create these tasks
		counter_ineq = 0
	
		for i in range(pl):
	
			if i != pl-1:
	
				A_ineq[i] = A_ineq_all[counter_ineq:counter_ineq + n_ineq_per_level, :]
				b_upper[i] = b_ineq_all[counter_ineq:counter_ineq + n_ineq_per_level]
				counter_ineq += n_ineq_per_level
	
			else:	
	
				A_ineq[i] = A_ineq_all[counter_ineq:counter_ineq + ineq_last_level, :]
				b_upper[i] = b_ineq_all[counter_ineq:counter_ineq + ineq_last_level]
				counter_ineq += ineq_last_level

		#Creating the lexHQP problem
		solvername = 'gurobi'
		P_dt = Problem() #The problem that uses duality trick
		P_dt.options.solver = solvername
		P_seq = {} #Dictonary of problem instances that use the duality trick
		w = {}
		sol_seq = {}
		seq_comp_time = 0
		for i in range(1, pl):
			P_seq[i] = Problem()
			P_seq[i].options.solver = solvername
			w[i] = RealVariable('w'+str(i), b_upper[i].shape[0])

			
		x= RealVariable('x', n)

		#Solve the problem using the sequential method
		try:
			for i in range(1, pl):
				P_seq[i].add_constraint(A_ineq[0]*x <= b_upper[0])

				for j in range(1, i):
					#Compute the optimal value of w from previous levels
					wj = sol_seq[j].primals[w[j]]
					P_seq[i].add_constraint(A_ineq[j]*x <= b_upper[j] + wj)

				P_seq[i].add_constraint(A_ineq[i]*x <= b_upper[i] + w[i])
				obj = abs(w[i])**2
				P_seq[i].set_objective("min", obj)
				sol_seq[i] = P_seq[i].solve()
				seq_comp_time += sol_seq[i].searchTime
		except:
			continue


		#Construct and solve the same problem using the duality trick

		#create the lagrange multipliers 
		lams = {}
		for i in range(1,pl):
			for j in range(0, i + 1):
				if i!= j:
					lams[(i, j)] = RealVariable('lam('+str(i) +','+str(j) +')', b_upper[j].shape[0])
					P_dt.add_constraint(lams[(i, j)] >= 0)
				else:
					lams[(i,i)] = w[i]

		#Add all the primal feasibility constraints
		P_dt.add_constraint(A_ineq[0]*x <= b_upper[0])
		for i in range(1, pl):
			P_dt.add_constraint(A_ineq[i]*x <= b_upper[i] + w[i])

		#Add the dual feasibility constraints
		
		for i in range(1, pl -1):

			# P_dt.add_constraint(lams[(i,i)] == w[i])

			constraint_acc = 0
			for j in range(0, i + 1):
				constraint_acc += A_ineq[j].T * lams[(i, j)]
			P_dt.add_constraint(constraint_acc == 0)

			#Adding the duality trick constraints
			constraint_acc = 0
			# for j in range(1, i):
				# constraint_acc -= lams[(i, j)].T*w[j]
			for j in range(0, i+1):
				constraint_acc -= lams[(i, j)].T*b_upper[j]
			constraint_acc -= 0.5*abs(lams[(i,i)])**2
			P_dt.add_constraint(0.5*abs(w[i])**2 <= constraint_acc)


		obj = 0.5*abs(w[pl-1])**2
		P_dt.set_objective("min", obj)

		try:
			sol_dt = P_dt.solve()
		except:
			continue


		
		
		# print(P_seq[2])
		# print(P_dt)
		print("Sequential total search time = " + str(seq_comp_time))
		tot_time_seq += seq_comp_time
		print("Duality trick total search time = " + str(sol_dt.searchTime))
		tot_time_dt += sol_dt.searchTime
	

	
		# print("Error norm seq and dt = " + str(cs.norm_1(sol_chqp[pl-1].value(hqp.cHQP_xdot[pl-1]) - sol.value(x_dot_dt))))
		# print("Error norm seq and ahqp2 = " + str(cs.norm_1(sol_chqp[pl-1].value(hqp.cHQP_xdot[pl-1]) - sol_ahqp2.value(x_dot))))
		
		for i in range(1, pl):
			con_per_level = len(sol_seq[i].primals[w[i]])
			print("violation at level = " + str(i))
			print("By sequential = " + str(cs.DM.ones(1, con_per_level)@sol_seq[i].primals[w[i]]))
			print("By duality trick = " + str(cs.DM.ones(1, con_per_level)@sol_dt.primals[w[i]]))
			# if cs.sumsqr(sol.value(hqp_dt.slacks[i])) <= cs.sumsqr(sol_chqp[i].value(hqp.cHQP_slacks[i])) -1e-3:
			# # 	print("DT gives a better lex optimal solution!!!!!!!!!")
			# 	break

			if cs.sumsqr(sol_seq[i].primals[w[i]]) <= cs.sumsqr(sol_dt.primals[w[i]]) - 1e-3:
				failures += 1
				break
			if cs.sumsqr(sol_seq[i].primals[w[i]]) >=cs.sumsqr(sol_dt.primals[w[i]]) + 1e-3:
				break

	print("Number of failures = " + str(failures))
	print("Total time by sequential method = " + str(tot_time_seq))
	print("Total time by duality trick method = " + str(tot_time_dt))
	
		


