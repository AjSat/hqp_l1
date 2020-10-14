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
import gurobipy as gp
from gurobipy import GRB

 	
if __name__ == '__main__':

	print("Main function")
	#Now comparing LexLP with the sequential method for a significantly larger problem
	failures = 0
	tot_time_seq = 0
	tot_time_dt = 0
	infeasibilities_sequential = 0
	infeasibilities_dt = 0
	for rand_seed in range(61,62):
		n = 5
		total_ineq_constraints = 20
		ineq_con_rank = 5
		pl = 10
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

		P_seq = {} #Dictonary of problem instances that use the duality trick
		w = {}
		P_seq_vars = {}
		sol_seq = {}
		seq_comp_time = 0
		for i in range(1, pl):
			P_seq[i] = gp.Model()
			w[i] = P_seq[i].addMVar(b_upper[i].shape[0], lb = 0.0, name ='w'+str(i))
			P_seq_vars[i] = {}
			P_seq_vars[i]['x'] = P_seq[i].addMVar(n, name = 'x')
			
		

		#Solve the problem using the sequential method
		try:
			for i in range(1, pl):
				P_seq[i].addConstr( A_ineq[0]@P_seq_vars[i]['x']<= b_upper[0])
				for j in range(1, i):
					#Compute the optimal value of w from previous levels
					wj = w[j].X
					P_seq[i].addConstr(A_ineq[j]@P_seq_vars[i]['x'] <= b_upper[j] + wj)
				P_seq[i].addConstr(A_ineq[i]@P_seq_vars[i]['x'] <= b_upper[i] + w[i])
				obj = np.ones(w[i].shape)@w[i]
				P_seq[i].setObjective(obj, GRB.MINIMIZE)
				P_seq[i].optimize()
				seq_comp_time += P_seq[i].Runtime
				w[i].X
		except:
			infeasibilities_sequential += 1
			continue

		

		#Construct and solve the same problem using the duality trick
		#create the lagrange multipliers 
		P_dt = gp.Model() #The problem that uses duality trick
		x= P_dt.addMVar(n, name = 'x')
		w_dt = {}
		for i in range(1, pl):
			w_dt[i] = P_dt.addMVar(b_upper[i].shape[0], lb = 0)
		lams = {}
		for i in range(1,pl):
			for j in range(0, i + 1):
				lams[(i, j)] = P_dt.addMVar(b_upper[j].shape[0], lb = 0, name = 'lam('+str(i) +','+str(j) +')')

		#Add all the primal feasibility constraints
		P_dt.addConstr(A_ineq[0]@x <= b_upper[0])
		for i in range(1, pl):
			P_dt.addConstr(A_ineq[i]@x <= b_upper[i] + w_dt[i])

		#Add the dual feasibility constraints
		
		for i in range(1, pl -1):

			P_dt.addConstr(lams[(i,i)] <= 1)

			constraint_acc = 0
			for j in range(0, i + 1):
				constraint_acc += A_ineq[j].T @ lams[(i, j)]
			P_dt.addConstr(constraint_acc == 0)

			#Adding the duality trick constraints
			constraint_acc = 0
			# for j in range(	1, i):
			# 	constraint_acc -= lams[(i, j)]@w_dt[j]
			for j in range(0, i+1):
				constraint_acc -= lams[(i, j)]@b_upper[j]
			temp = np.ones(w_dt[i].shape)@w_dt[i]
			P_dt.addConstr(temp <= constraint_acc)


		obj = w_dt[pl-1].sum()
		P_dt.setObjective(obj, GRB.MINIMIZE)

		# P_dt.params.NonConvex = 2
		P_dt.params.Method = 5
		# P_dt.params.PSDTol = 100
		try:
			P_dt.optimize()
			w_dt[i].X
		except:
			print("DT INFEASILBE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
			infeasibilities_dt += 1
			continue


		
		
		# print(P_seq[2])
		# print(P_dt)
		print("\n\n\n\n")
		print("Sequential total search time = " + str(seq_comp_time))
		tot_time_seq += seq_comp_time
		print("Duality trick total search time = " + str(P_dt.Runtime))
		tot_time_dt += P_dt.Runtime
	

	
		# print("Error norm seq and dt = " + str(cs.norm_1(sol_chqp[pl-1].value(hqp.cHQP_xdot[pl-1]) - sol.value(x_dot_dt))))
		# print("Error norm seq and ahqp2 = " + str(cs.norm_1(sol_chqp[pl-1].value(hqp.cHQP_xdot[pl-1]) - sol_ahqp2.value(x_dot))))
		
		for i in range(1, pl):
			con_per_level = b_upper[i].shape[0]
			print("violation at level = " + str(i))
			print("By sequential = " + str(cs.DM.ones(1, con_per_level)@w[i].X))
			print("By duality trick = " + str(cs.DM.ones(1, con_per_level)@w_dt[i].X))
			# if cs.sumsqr(sol.value(hqp_dt.slacks[i])) <= cs.sumsqr(sol_chqp[i].value(hqp.cHQP_slacks[i])) -1e-3:
			# # 	print("DT gives a better lex optimal solution!!!!!!!!!")
			# 	break

			if cs.DM.ones(1, con_per_level)@w[i].X <= cs.DM.ones(1, con_per_level)@w_dt[i].X - 1e-3:
				failures += 1
				break
			if cs.DM.ones(1, con_per_level)@w[i].X >= cs.DM.ones(1, con_per_level)@w_dt[i].X + 1e-3:
				break

	print("Number of failures = " + str(failures))
	print("Number of infeasibilities seq = +" + str(infeasibilities_sequential))
	print("Number of infeasibilities dt = +" + str(infeasibilities_dt))
	print("Total time by sequential method = " + str(tot_time_seq))
	print("Total time by duality trick method = " + str(tot_time_dt))
	
		


