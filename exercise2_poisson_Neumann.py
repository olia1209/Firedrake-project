import firedrake as fd
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
#f.interpolate(fd.sin(fd.pi * x) * fd.sin(fd.pi * y)) # maybe f is not correct?
# try Helmholtz eqn's f
f.interpolate((1+8*fd.pi*fd.pi)*fd.cos(x*fd.pi*2)*fd.cos(y*fd.pi*2))

a = fd.inner(fd.grad(u), fd.grad(v)) * fd.dx
L = fd.inner(f, v) * fd.dx 

# Let u be a function holding the solution
u = fd.Function(V)

# the operator is symmetric positive definie? cg seems not work, also gmres
fd.solve(a == L, u)


# Make plot
fd.VTKFile("poisson.pvd").write(u)

fig, axes = plt.subplots()
contours = fd.tricontour(u, axes=axes)
fig.colorbar(contours)
plt.show()
