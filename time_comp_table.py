import json
import matplotlib.pyplot as plt
import statistics
# import tkinter
from pylab import *

#Load the json data
with open('wlp-a_vs_chqp_results.txt', 'r') as fp:
	comp_time = json.load(fp)

results_dict = {}
for i in range(2,11):
	slp_time = comp_time[str(i)+",0.5"][8]
	slp_mean = statistics.mean(slp_time)
	slp_std = statistics.stdev(slp_time)
	wlp_time = comp_time[str(i)+",0.5"][9]
	wlp_mean = statistics.mean(wlp_time)
	wlp_std = statistics.stdev(wlp_time)
	results_dict[i] = {}
	# results_dict[i]['mean_slp'] = slp_mean
	# results_dict[i]['std_slp'] = slp_std
	results_dict[i]['mean_wlp'] = wlp_mean
	# results_dict[i]['std_wlp'] = wlp_std

print(results_dict)
