import casadi as cs
import numpy as np

opti = cs.Opti()


b = opti.variable(50,1)
H = opti.parameter(50,50)

obj = b.T@H@b
opti.subject_to(b <= np.random.randn(50,1))
opti.minimize(obj)

opti.solver("sqpmethod", {"expand":True, "qpsol": 'qpoases',   'print_iteration': False, 'print_header': False, 'print_status': False, "print_time":False, 'max_iter': 1000})

opti2 = cs.Opti()
b2 = opti2.variable(50,1)
H2 = opti2.parameter(50,50)
obj = b2.T@H2@b2

opti2.subject_to(b2 <= np.random.randn(50,1))

opti2.minimize(obj)
opti2.solver("sqpmethod", {"expand":True, "qpsol": 'qpoases',  'print_iteration': False, 'print_header': False, 'print_status': False, "print_time":False, 'max_iter': 1000})
H_val = np.random.randn(50,50)
H_val = H_val.T@H_val

H_val2 = np.random.randn(50,50)
H_val2 = H_val2.T@H_val2

for i in range(5):

	opti.set_value(H, H_val)
	opti.solve()

	opti2.set_value(H2, H_val2)
	opti2.solve()

	r = np.random.randn(50,50)
	H_add = r.T@r*1e-6
	H_val = H_val + H_add

	r2 = np.random.randn(50,50)
	H_add2 = r2.T@r2*1e-6
	H_val2 = H_val2 + H_add2
