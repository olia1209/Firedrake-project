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

u_trial = fd.TrialFunction(V)
v = fd.TestFunction(V)
f = fd.Function(V)
x, y = fd.SpatialCoordinate(mesh)
f.interpolate(fd.sin(fd.pi * x) * fd.sin(fd.pi * y))

# Dirichlet
bc1 = fd.DirichletBC(V, 0, "on_boundary")

# Solve forward equation
a = fd.inner(fd.grad(u_trial), fd.grad(v)) * fd.dx
L = fd.inner(f, v) * fd.dx
u = fd.Function(V)
fd.solve(a == L, u, bcs=[bc1])

# Solve adjoint equation
lambda_trial = fd.TrialFunction(V) # Define lambda in V
a = fd.inner(fd.grad(v), fd.grad(lambda_trial)) * fd.dx
L = fd.inner(u, v) * fd.dx
lambda_ = fd.Function(V)
fd.solve(a == L, lambda_, bcs=[bc1])





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
def J(f_new):
    u_new = fd.TrialFunction(V)
    a = fd.inner(fd.grad(u_new), fd.grad(v)) * fd.dx
    L = fd.inner(f, v) * fd.dx
    u_new = fd.Function(V)
    fd.solve(a == L, u_new, bcs=[bc1])
    return fd.assemble(0.5 * fd.inner(u_new, u_new)*fd.dx + 0.5 * alpha*fd.inner(f_new, f_new)*fd.dx)


# Calculate diff
diff = []
J2 = J(f)

for h in h_value:
    f_new = fd.Function(V)
    f_new.assign(f + h * delta_m)
    J1 = J(f_new)
    error = np.abs(J1 - J2 - h * dJ)
    diff.append(error)
    print("h = {:.1e}, error = {:.3e}".format(h, error))

for i in range(len(h_value) - 1):
    p = np.log(diff[i] / diff[i+1]) / np.log(h_value[i]/h_value[i+1])
    print("h = {:.1e} -> h = {:.1e}, 收敛率 ≈ {:.2f}".format(h_value[i], h_value[i+1], p))


