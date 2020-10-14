import json
import matplotlib.pyplot as plt
# import tkinter
from pylab import *

#Load the json data
with open('/home/ajay/Desktop/hqp_l1/dt_vs_chqp_benchmark_ipopt4.txt', 'r') as fp:
	data_run = json.load(fp)


pl_all = [2, 3, 4, 5, 6, 7, 8, 9, 10]

seq_time_array_mean = []
dt_time_array_mean = []
hqp_l1_ta_mean = []

seq_time_array_std = []
dt_time_array_std = []
hqp_l1_ta_std = []

seq_time_array_list = []
dt_time_array_list = []
hqp_l1_ta_list = []

xtickLabels = []
fig, ax = plt.subplots()
for pl in pl_all:

	xtickLabels.append(str(pl))
	data = data_run[str(pl)]
	counter = data['counter']
	seq_time_array = np.array(data['seq_time_array'])
	dt_time_array = np.array(data['dt_time_array'])
	hqp_l1_ta = np.array(data['hqp-l1_ta']) 

	seq_time_array_list.append(seq_time_array)
	dt_time_array_list.append(dt_time_array)
	hqp_l1_ta_list.append(hqp_l1_ta)

	seq_time_array_mean.append(np.mean(seq_time_array))
	dt_time_array_mean.append(np.mean(dt_time_array))
	hqp_l1_ta_mean.append(np.mean(hqp_l1_ta))

	seq_time_array_std.append(np.std(seq_time_array))
	dt_time_array_std.append(np.std(dt_time_array))
	hqp_l1_ta_std.append(np.std(hqp_l1_ta))

#Using a different dataset for the 3 priority levels because the previous one was affected by 
#background cpu processes
# with open('/home/ajay/Desktop/hqp_l1/dt_vs_chqp_benchmark_ipopt2.txt', 'r') as fp:
# 	data_run = json.load(fp)

# data = data_run['3']
# seq_time_array = np.array(data['seq_time_array'])
# dt_time_array = np.array(data['dt_time_array'])
# hqp_l1_ta = np.array(data['hqp-l1_ta'])

# seq_time_array_mean[2] = np.mean(seq_time_array)
# dt_time_array_mean[2] = np.mean(dt_time_array)
# hqp_l1_ta_mean[2] = np.mean(hqp_l1_ta)
# seq_time_array_std[2] = np.std(seq_time_array)
# dt_time_array_std[2] = np.std(dt_time_array)
# hqp_l1_ta_std[2] = np.std(hqp_l1_ta)


#Error bar
# ax.errorbar(pl_all,seq_time_array_mean, yerr = seq_time_array_std, fmt = '-o')
# ax.errorbar(np.array(pl_all)+0.05,dt_time_array_mean, yerr = dt_time_array_std, fmt = '-o')
# ax.errorbar(np.array(pl_all)-0.05,hqp_l1_ta_mean, yerr = hqp_l1_ta_std, fmt = '-o')

#Plotting a violin plot
vp1 = ax.violinplot(seq_time_array_list, showmeans=False, showmedians=True)
# ax.boxplot(seq_time_array_list)
ax.set_xticks(np.array(pl_all)-1)
ax.set_xticklabels(xtickLabels)

vp2 = ax.violinplot(dt_time_array_list, showmeans=False, showmedians=True)
# ax.boxplot(dt_time_array_list)
ax.set_xticks(np.array(pl_all)-1)
ax.set_xticklabels(xtickLabels)


vp3 = ax.violinplot(hqp_l1_ta_list, showmeans=False, showmedians=True)
# ax.set_xticks(np.array(pl_all)-0.2)
ax.set_xticklabels(xtickLabels)


plt.yscale('log')
ax.set_xlabel("Number of priority levels")
ax.set_ylabel("Computation time (s)")
ax.set_title("Timing comparison between different formulations")
# ax.yaxis(grid = 'true')

plt.legend([vp1['bodies'][0],vp2['bodies'][0], vp3['bodies'][0]], ['Sequential', 'Duality trick', 'Weighted (fixed)'], loc=2)




plt.show()