import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.adjoint import *

# -----------------optimal control poisson----------------------
continue_annotation()

mesh = UnitSquareMesh(10, 10)

# Define two different function spaces
V = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "CG", 1)

f = Function(W)
f.assign(0)
u = TrialFunction(V)
v = TestFunction(V)

# Solve the Poisson equation
bc1 = DirichletBC(V, 0, "on_boundary")
a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx
u = Function(V)
solve(a == L, u, bcs=[bc1])

# Define d(x, y)
d = Function(W)
x, y = SpatialCoordinate(mesh)
d.interpolate(1 / (2 * pi **2) * sin(pi * x) * sin(pi * y))

# Define regularisation parameter
alpha = Constant(1e-6)


# Define control inner product
ctrl_inner_product = "L2"

# Define functional of interest and the reduced functional
if ctrl_inner_product == "L2":
    J = assemble((0.5*inner(u-d, u-d))*dx + alpha/2*f**2*dx)
elif ctrl_inner_product == "H1":
    J = assemble((0.5*inner(u-d, u-d))*dx + alpha/2*((grad(f)**2)*dx + f**2*dx))
else:
    raise ValueError("Choose control inner product either as L2 or H1")


# Define reduced functional
control = Control(f)
Jhat = ReducedFunctional(J, control)
pause_annotation()
get_working_tape().progress_bar = ProgressBar
tape = get_working_tape()
#tape.visualise('tape2.pdf')

iter = [0]
# Solve optimsation problem
def callback(*args):
    iter[0] += 1

f_opt = minimize(Jhat, method="L-BFGS-B", options={"gtol": 1e-9, "maxiter": 100, "disp":True}, callback=callback)
print(iter)


# Plot solution
fig, axes = plt.subplots()
contours = tricontour(f_opt, axes=axes)
fig.colorbar(contours)
plt.title("Optimized Control f_opt")
plt.savefig("output.png")