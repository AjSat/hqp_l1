#Code to analyze the data of the benchmarking test that is stored in the json form
import json
import matplotlib.pyplot as plt
# import tkinter
from pylab import *

#Load the json data
with open('/home/ajay/Desktop/hqp_l1/hqp_vs_chqp_results3(0.2).txt', 'r') as fp:
	results = json.load(fp)

#begin averaging the data
# pl_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10]
# gamma_vals = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 7.0, 10.0, 14.0, 15.0, 20.0, 40.0, 60, 100, 150, 200, 300, 400, 500]

gamma_vals = [0.2,  1.0,  5.0]
# gamma_vals = [0.1]
# gamma_vals = [10.0, 20.0, 50.0]
pl_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# pl_vals = [3, 4, 5, 6, 7]
#Creating an empty array for storing all parameters for different horizon sizes
hierarchy_failed_average = {}
identical_solution_average = {}
geq_constraints_average = {}
same_constraints_satisfied_average = {}

chqp_rate = {}
hqp_rate = {}
success_rate = {}
figure()
for p in pl_vals:
	for gamma in gamma_vals:
		res = results[str(p)+','+str(gamma)]
		failure_rate = 	res[0]/(res[4] + 1e-12)
		identical_soln_rate = res[2]/(res[4] + 1e-12)
		
		# if identical_soln_rate >= 0.5:
		if identical_soln_rate >=0.95:
			semilogy(p, gamma, 'gx')
		else:
			semilogy(p, gamma, 'rx')
		hqp_rate[(p, gamma)] = len(res[9])/300
		success_rate[(p, gamma)] = (res[5]/(res[4] + 1e-12)) * (len(res[9]) >= 25)
	chqp_rate[p] = len(res[8])/300

print(chqp_rate)
print(hqp_rate)
print(success_rate)


suc_rate = []
hqp_r = []
chqp_r = []

for p in pl_vals:
	gamma = 0.1
	suc_rate.append(success_rate[(p, gamma)])
	hqp_r.append(hqp_rate[(p, gamma)])
	chqp_r.append(chqp_rate[p])

figure()
# for p in pl_vals:
	# gamma = 0.1

plot(pl_vals, suc_rate, 'gx', label = 'success rate')
plot(pl_vals, hqp_r, 'kx', label = 'hqp solution rate' )
plot(pl_vals, chqp_r, 'bx', label='seq method solution rate')
		
legend()
show(block=True)