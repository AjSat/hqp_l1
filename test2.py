from picos import *
import random
import numpy as np
from scipy import sparse

P = Problem()

n = 25
m = 100

x = RealVariable("x", n)

w = RealVariable("w", m)

# b = np.random.randn(m, n)
b = (sparse.random(m, n, density=0.05))
b = b.toarray()
sparsity_pattern = (b != 0.0)
# print(sparsity_pattern)
b =  4*(b-0.5) * sparsity_pattern
print(b)
# obj = abs(x)**2*1e-6
obj = 0
for wi in w:
	obj += wi + wi**2

# obj += abs(w)**2*1e-6

P.add_constraint(b*x + 10 <= w)
P.add_constraint(50 >= w >= 0)

P.set_objective("min", obj)
# P.set_objective("min", (t - 5)**2 + 2)
solvername = 'mosek'
print(P)

P.options.solver = solvername

P.options.hotstart = True
solution = P.solve()	
print(solution.primals)
print(solution.searchTime)



solution = P.solve()
print(solution.searchTime)

solution = P.solve()
print(solution.searchTime)
# print(P.all_solvers)

P2 = Problem()
# b2 = Constant("b2", np.random.randn(m*n), [m,n])
b2 = np.random.randn(m, n)
x2 = RealVariable("x", n)
w2 = RealVariable("w", m)
con = P2.add_constraint(b2*x2 + 10 <= w2)
P2.add_constraint(w2 >= 0)
obj = 0
for wi in w2:
	obj += wi+ wi**2
P2.set_objective("min", obj)
P2.options.solver = solvername
P2.options.hotstart = True
solution = P2.solve()	
print("With P2")
print(solution.searchTime)

solution = P2.solve()	
print("With P2")
print(solution.searchTime)

solution = P.solve()	
print("With P")
print(solution.searchTime)

solution = P2.solve()	
print("With P2")
print(solution.searchTime)

P3 = Problem()
b3 = np.random.randn(m, n)
x3 = RealVariable("x", n)
w3 = RealVariable("w", m)
P3.add_constraint(b3*x3 + 10 <= w3)
P3.add_constraint(w3 >= 0)
obj = 0
for wi in w3:
	obj += wi+ wi**2
P3.options.solver = solvername
P3.options.hotstart = True
P3.set_objective("min", obj)
solution = P3.solve()	
print("With P3")
print(solution.searchTime)

solution = P3.solve()	
print("With P3")
print(solution.searchTime)

P2.options.verbosity = True
tot_time = 0
for i in range(100):
	del P2
	del w2
	del b2
	P2 = Problem()
	b2 = np.random.randn(m, n)
	b2 = (sparse.random(m, n, density=1.0))
	b2 = b2.toarray()
	sparsity_pattern = (b2 != 0.0)
	# print(sparsity_pattern)
	b2 =  4*(b2-0.5) * sparsity_pattern
	x2 = RealVariable("x", n)
	w2 = RealVariable("w", m)
	obj = 0
	for wi in w2:
		obj += wi**2 + wi
	P2.add_constraint(50 >= w2 >= 0)
	P2.set_objective("min", obj)
	con = P2.add_constraint(b2*x2 + 10 <= w2)
	P2.options.solver = solvername
	solution = P2.solve()	
	tot_time += solution.searchTime

print("Average computation time")
print(tot_time/100)