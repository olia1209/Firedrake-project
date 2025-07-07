import numpy as np
import time
from firedrake import *
from firedrake.adjoint import *
from pyadjoint import Block, MinimizationProblem, TAOSolver, get_working_tape
import netgen


# Create mesh
n = 16
mesh = UnitSquareMesh(n, n)

# Refine in the center
def mark_center(mesh):
    W = FunctionSpace(mesh, "DG", 0)
    mark = Function(W)
    x, y = SpatialCoordinate(mesh)
    center = conditional( And((abs(x - 0.5) < 0.1) , (abs(y - 0.5) < 0.1)), 1.0, 0.0 )
    mark.interpolate(center)
    return mark

# Refinement function
def refine_mesh_center(mesh, max_iterations=5):
    for i in range(max_iterations):
        print(f"Refinement level: {i}")
        mark = mark_center(mesh)

        mesh = mesh.refine_marked_elements(mark)
        VTKFile(f"output/refined_mesh_{i}.pvd").write(mesh)
    return mesh

# Uniform refine
hierarchy = MeshHierarchy(mesh, 4)
mesh = hierarchy[-1]

#mesh = refine_mesh_center(mesh, 4)



# Define discrete function spaces and funcions
V = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "CG", 1)

f = Function(W)
f.assign(0)
u = TrialFunction(V)
v = TestFunction(V)

# Define and solve the Poisson equation to generate the dolfin-adjoint annotation
F = (inner(grad(u), grad(v)) - inner(f, v))*dx
bc = DirichletBC(V, 0.0, "on_boundary")
u = Function(V)
solve(F == 0, u, bc)

# Define regularisation parameter
alpha = Constant(1e-6)

# Define the expressions of the analytical solution
d = Function(W)
x, y = SpatialCoordinate(mesh)
d.interpolate(1 / (2 * pi **2) * sin(pi * x) * sin(pi * y))
# f_analytic = assemble(sin(pi*x)*sin(pi*y))
# u_analytic = assemble(1/(2*pi*pi)*sin(pi*x)*sin(pi*y))
# j_analytic = assemble((1./2*(u_analytic-d)**2 + alpha*f_analytic**2)*dx(mesh))

# Define functional of interest and the reduced functional
ctrl_inner_product = "L2"
regularisation_norm = "L2"

if regularisation_norm == "L2":
    J = assemble((0.5*inner(u-d, u-d))*dx + alpha/2*f**2*dx)
elif regularisation_norm == "H1":
    J = assemble((0.5*inner(u-d, u-d))*dx + alpha/2*((grad(f)**2)*dx + f**2*dx))
elif regularisation_norm == "H0_1":
    J = assemble((0.5*inner(u-d, u-d))*dx + alpha/2*(grad(f)**2)*dx)
elif regularisation_norm == "H2":
    J = assemble((0.5*inner(u-d, u-d))*dx + alpha/2*((grad(grad(f))**2)*dx + (grad(f)**2)*dx + f**2*dx))

control = Control(f)
rf = ReducedFunctional(J, control)

opt_package = "tao_lmvm"
convert_options = "l2"

if opt_package == "tao_lmvm":
    solver = TAOSolver(rf, {"tao_type": "lmvm",
                                 "tao_gatol": 1.0e-7,
                                 "tao_grtol": 0.0,
                                 "tao_gttol": 0.0},
                       convert_options=convert_options)
    start_time = time.time()
    f_opt = solver.solve()

elif opt_package == "tao_nlm":
    solver = TAOSolver(rf, {"tao_type": "nlm",
                                 "tao_gatol": 1.0e-7,
                                 "tao_grtol": 0.0,
                                 "tao_gttol": 0.0},
                       convert_options=convert_options)
    start_time = time.time()
    f_opt = solver.solve()


plot(f_opt, title="f_opt", interactive=True)

j = rf(f_opt)
dj = rf.derivative(forget=False, project=True)[0]
print ("Final:   \tJ = %s\t |dJ|_L2 = %s" % (j, norm(dj)))
print ("=================================")
print ("h(min):              %e." % mesh.hmin())
print ("h(max):              %e." % mesh.hmax())
print ("=================================")