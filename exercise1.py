import firedrake as fd
import matplotlib.pyplot as plt

mesh = fd.UnitSquareMesh(10, 10)
V = fd.FunctionSpace(mesh, "CG", 1)

u = fd.TrialFunction(V)
v = fd.TestFunction(V)

f = fd.Function(V)
x, y = fd.SpatialCoordinate(mesh)
f.interpolate((1+8*fd.pi*fd.pi)*fd.cos(x*fd.pi*2)*fd.cos(y*fd.pi*2))

a = (fd.inner(fd.grad(u), fd.grad(v)) + fd.inner(u, v)) * fd.dx
L = fd.inner(f, v) * fd.dx

u = fd.Function(V)
fd.solve(a == L, u, solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'})

fd.VTKFile("helmholtz.pvd").write(u)

fig, axes = plt.subplots()
contours = fd.tricontour(u, axes=axes)
fig.colorbar(contours)

plt.show()

f.interpolate(fd.cos(x*fd.pi*2)*fd.cos(y*fd.pi*2))
print(fd.sqrt(fd.assemble(fd.dot(u - f, u - f) * fd.dx)))

