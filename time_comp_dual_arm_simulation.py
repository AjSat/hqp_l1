import json
import matplotlib.pyplot as plt
# import tkinter
from pylab import *

#Load the json data
with open('comp_time_l1_weighted_warm_start2.txt', 'r') as fp:
	weighted_warm_start = json.load(fp)

with open('comp_time_weighted_mosek.txt', 'r') as fp:
	weighted_warm_start_mosek = json.load(fp)

with open('comp_time_l1_weighted_nowarm_start.txt', 'r') as fp:
	weighted_nowarm_start = json.load(fp)

with open('comp_time_seq_no_warm_start.txt', 'r') as fp:
	seq_no_warm_start = json.load(fp)

with open('comp_time_weighted_adaptive.txt', 'r') as fp:
	weighted_adaptive_warm_start = json.load(fp)

time_step = 0.005

t_weighted_nw = weighted_nowarm_start['comp_time']
t_axis1 = np.array(range(len(t_weighted_nw)))*0.005

figure()
plot(t_axis1, t_weighted_nw, label = 'weighted-qpoases')

t_seq = seq_no_warm_start['comp_time']
t_axis3 = np.array(range(len(t_seq)))*0.005
plot(t_axis3, t_seq, label = 'sequential-qpoases')

t_weighted_w = weighted_warm_start['comp_time']
t_axis2 = np.array(range(len(t_weighted_w)))*0.005
plot(t_axis2, t_weighted_w, label = 'weighted-warmstart-qpoases')

t_weighted_adaptive = weighted_adaptive_warm_start['comp_time']
t_axis2 = np.array(range(len(t_weighted_adaptive)))*0.005
plot(t_axis2, t_weighted_adaptive, label = 'weighted-adaptive-warmstart-qpoases')

t_weighted_mosek = weighted_warm_start_mosek['comp_time']
t_axis2 = np.array(range(len(t_weighted_mosek)))*0.005
plot(t_axis2, t_weighted_mosek, label = 'weighted-warmstart-mosek')

legend()
title("Computation time comparison between sequential and weighted method for robot control")
xlabel("Time(s)")
ylabel('Computation time(s)')
plt.yscale('log')
plt.grid()
# ax.set_xlabel("Number of priority levels")
# ax.set_ylabel("Computation time (s)")
# ax.set_title("Timing comparison between different formulations")

# plt.legend([vp1['bodies'][0],vp2['bodies'][0], vp3['bodies'][0]], ['Sequential', 'Duality trick', 'Weighted (fixed)'], loc=2)




plt.show()
