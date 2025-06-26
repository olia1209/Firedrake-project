import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import firedrake as fd
from firedrake import *
from firedrake.adjoint import *

# Able to use adjoint
continue_annotation()

# Omega is unit square
mesh = fd.UnitSquareMesh(10, 10)

# H_1 space
V = fd.FunctionSpace(mesh, "CG", 1)

u = fd.TrialFunction(V)
v = fd.TestFunction(V)

#  Use f
f = fd.Function(V)
x, y = fd.SpatialCoordinate(mesh)
f.interpolate(fd.sin(fd.pi * x) * fd.sin(fd.pi * y))

# Dirichlet
bc1 = fd.DirichletBC(V, 0, "on_boundary")

# Solve forward equation
a = fd.inner(fd.grad(u), fd.grad(v)) * fd.dx
L = fd.inner(f, v) * fd.dx
u = fd.Function(V)
fd.solve(a == L, u, bcs=[bc1])


# Set optimisation problem
alpha = fd.Constant(10)
J = fd.assemble(1/2*fd.inner(u, u)*fd.dx + alpha/2*fd.inner(f, f)*fd.dx)
m = Control(f)
Jhat = ReducedFunctional(J, m)
pause_annotation()
get_working_tape().progress_bar = ProgressBar
tape = get_working_tape()
#tape.visualise('tape.pdf')

dJ = Jhat.derivative(options = {'riesz_representation':None})
print(dJ)

#-----------------Taylor test-----------------
delta_m = fd.Function(V)
delta_m.interpolate(1 / 10 * fd.sin(x) * fd.cos(y))

taylor_test(Jhat, f, delta_m)

f_opt = minimize(Jhat)

# Plot f_opt
fig, axes = plt.subplots()
contours = fd.tricontour(f_opt, axes=axes)
fig.colorbar(contours)

plt.savefig("output.png")

print("Jhat(f) = %.8g\nJhat(f_opt) = %.8g" % (Jhat(f), Jhat(f_opt)))

