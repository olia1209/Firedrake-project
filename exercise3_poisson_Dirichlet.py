import firedrake as fd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Omega is unit square
mesh = fd.UnitSquareMesh(10, 10)

# Require H_1 space
V = fd.FunctionSpace(mesh, "CG", 1)

u = fd.TrialFunction(V)
v = fd.TestFunction(V)

# Define f
f = fd.Function(V)
x, y = fd.SpatialCoordinate(mesh)
f.interpolate(fd.sin(fd.pi * x) * fd.sin(fd.pi * y))

# Dirichlet
bc1 = fd.DirichletBC(V, 0, "on_boundary")

a = fd.inner(fd.grad(u), fd.grad(v)) * fd.dx
L = fd.inner(f, v) * fd.dx

# Let u be a function holding the solution
u = fd.Function(V)

fd.solve(a == L, u, bcs=[bc1], solver_parameters={'ksp_type':'cg', 'pc_type':'none'})


# Make plot
fd.VTKFile("poisson.pvd").write(u)

fig, axes = plt.subplots()
contours = fd.tricontour(u, axes=axes)
fig.colorbar(contours)
#plt.savefig('poisson.png')
plt.show()
