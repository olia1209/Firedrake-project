import firedrake as fd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from firedrake import adjoint as adj
import numpy as np

# Omega is unit square
mesh = fd.UnitSquareMesh(10, 10)

# H_1 space
V = fd.FunctionSpace(mesh, "CG", 1)

# Define forward solving function
f = fd.Function(V)
x, y = fd.SpatialCoordinate(mesh)
f.interpolate(fd.sin(fd.pi * x) * fd.sin(fd.pi * y))

# Dirichlet
bc1 = fd.DirichletBC(V, 0, "on_boundary")


u_trial = fd.TrialFunction(V)
v = fd.TestFunction(V)
f_ = fd.Function(V)
a = fd.inner(fd.grad(u_trial), fd.grad(v)) * fd.dx
L = fd.inner(f_, v) * fd.dx
u = fd.Function(V)
problem = fd.LinearVariationalProblem(a, L, u, bcs=bc1)
solver = fd.LinearVariationalSolver(problem)


def forward(f_new):
    f_.assign(f_new)
    solver.solve()
    return u

# Solve forward equation
u_adj = forward(f)

# Solve adjoint equation
lambda_trial = fd.TrialFunction(V) # Define lambda in V
v = fd.TestFunction(V)
a = fd.inner(fd.grad(v), fd.grad(lambda_trial)) * fd.dx
L = fd.inner(u_adj, v) * fd.dx
lambda_ = fd.Function(V)
fd.solve(a == L, lambda_, bcs=[bc1]) #! Forget to add bc here



#---------------- Taylor test--------------------

# Define delta m
delta_m = fd.Function(V)
delta_m.interpolate(1 / 10 * fd.sin(x) * fd.cos(y))

# Define h
h_value = [1e-1, 1e-2, 1e-3, 1e-4]

# Get dJ
alpha = fd.Constant(10)
dJ = fd.assemble((lambda_ + alpha * f)* delta_m *fd.dx)

# Since J_hat(m) = J(u(m),m), when changing m, u should be also changed
def J(f_update):
    u_new = forward(f_update)
    return fd.assemble(0.5 * fd.inner(u_new, u_new)*fd.dx + 0.5 * alpha*fd.inner(f_update, f_update)*fd.dx)


# Calculate diff
diff = []
J2 = J(f)

for h in h_value:
    f_new = fd.Function(V)
    f_new.assign(f + h * delta_m)
    J1 = J(f_new)
    error = np.abs(J1 - J2 - h * dJ)
    diff.append(error)

for i in range(len(h_value) - 1):
    p = np.log(diff[i] / diff[i+1]) / np.log(h_value[i]/h_value[i+1])
    print(p)


