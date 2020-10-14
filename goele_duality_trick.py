import casadi as cs

opti = cs.Opti()

w3 = opti.variable()
w2 = opti.variable()
w1 = opti.variable()

x = opti.variable()

lam1 = opti.variable()

lam21 = opti.variable()
lam22 = opti.variable()

opti.subject_to(x <= 0.5 + w3)
opti.subject_to(x <= 1 + w1)
opti.subject_to(-x <= -3 + w2)

opti.subject_to(w1 >= 0)
opti.subject_to(w2 >= 0)
opti.subject_to(w3 >= 0)
opti.subject_to(lam1 >= 0)
opti.subject_to(lam1 <= 1)
opti.subject_to(lam21 >= 0)
opti.subject_to(lam22 >= 0)
opti.subject_to(lam22 <= 1)

opti.subject_to(w1 <= lam1)
opti.subject_to(w2 <= -lam21 + 3*lam22)

opti.subject_to(lam1 == 0)
opti.subject_to(lam21 - lam22 == 0 )
opti.minimize(w3)

opti.solver("ipopt",  {'ipopt':{"max_iter": 2000, 'hessian_approximation':'limited-memory', 'limited_memory_max_history' : 5, 'tol':1e-3, 'print_level':5}})

sol = opti.solve()

print(sol.value(x))