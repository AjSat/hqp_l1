import numpy as np
from scipy import linalg as la 
import time

import casadi as cs

a = np.random.rand(15, 100)
b = np.random.rand(15, 1)


tic = time.time(); 
for i in range(1000):
	# c = np.linalg.solve(a, b); 
	la.lstsq(a, b)

print(time.time() - tic)

acas = cs.SX.sym('as', 150, 150)

