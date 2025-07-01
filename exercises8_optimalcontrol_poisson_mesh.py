import numpy.random
import numpy as np
import time
from firedrake import *
from firedrake.adjoint import *
from pyadjoint import Block, MinimizationProblem, TAOSolver, get_working_tape

set_log_level(ERROR)
parameters['std_out_all_processes'] = False
tao_args = """
            --petsc.tao_view
            --petsc.tao_monitor
            --petsc.tao_nls_ksp_type cg
            --petsc.tao_nls_pc_type none
            --petsc.tao_ntr_pc_type none
           """.split()

print ("Tao arguments:", tao_args)
parameters.parse(tao_args)

# Create mesh, refined in the center
n = 16  # Use n = 4 for random refine
        # Use n = 4, 8, 16, 32 for uniform refinement
        # Use n = 8 for center refine
mesh = UnitSquareMesh(n, n)

# def randomly_refine(initial_mesh, ratio_to_refine= .3):
#     numpy.random.seed(0)
#     cf = CellFunction('bool', initial_mesh)
#     for k in xrange(len(cf)):
#         if numpy.random.rand() < ratio_to_refine:
#             cf[k] = True
#     return refine(initial_mesh, cell_markers = cf)

def refine_center(mesh, L=0.2):
    cf = CellFunction("bool", mesh)
    subdomain = CompiledSubDomain('std::abs(x[0]-0.5)<'+str(L)+' && std::abs(x[1]-0.5)<'+str(L))
    subdomain.mark(cf, True)
    return refine(mesh, cf)

mesh = refine_center(mesh, L=0.4)

# Define discrete function spaces and funcions
V = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "CG", 1)

f = Function(W)
x, y = SpatialCoordinate(mesh)
f.interpolate(x + y, name='Control')
#f = interpolate(Expression("x[0]+x[1]"), W, name='Control')
#f = interpolate(Expression("sin(pi*x[0])*sin(pi*x[1])"), W, name='Control')
u = Function(V, name='State')
v = TestFunction(V)

# Define and solve the Poisson equation to generate the dolfin-adjoint annotation
F = (inner(grad(u), grad(v)) - f*v)*dx
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, u, bc)

# Define regularisation parameter
alpha = Constant(1e-6)

# Define the expressions of the analytical solution
x = SpatialCoordinate(mesh)
d = (1/(2*pi**2) + 2*alpha*pi**2)*sin(pi*x[0])*sin(pi*x[1]) # the desired temperature profile
f_analytic = assemble(sin(pi*x)*sin(pi*y))
u_analytic = assemble(1/(2*pi*pi)*sin(pi*x)*sin(pi*y))
j_analytic = assemble((1./2*(u_analytic-d)**2 + alpha*f_analytic**2)*dx(mesh))

# Define functional of interest and the reduced functional
ctrl_inner_product = "H1"
regularisation_norm = "H1"

if regularisation_norm == "L2":
    J = assemble((0.5*inner(u-d, u-d))*dx + alpha/2*f**2*dx)
elif regularisation_norm == "H1":
    J = assemble((0.5*inner(u-d, u-d))*dx + alpha/2*((grad(f)**2)*dx + f**2*dx))
elif regularisation_norm == "H0_1":
    J = assemble((0.5*inner(u-d, u-d))*dx + alpha/2*(grad(f)**2)*dx)
elif regularisation_norm == "H2":
    J = assemble((0.5*inner(u-d, u-d))*dx + alpha/2*((grad(grad(f))**2)*dx + (grad(f)**2)*dx + f**2*dx))

control = Control(f)

dj_tol = 1e-7
f_func = project(f_analytic, W)

# How to define cd here?
def cb(j, dj, x):

    # Compute gradient norm in H 
    dj = moola.DolfinDualVector(dj, inner_product=ctrl_inner_product).primal().data
    dj_norm = norm(dj)
    print ("Callback:\tJ = %s\t |dJ|_H = %s" % (j, dj_norm))

    if dj_norm < dj_tol:
        print("="*10 + " % s seconds " % (time.time() - start_time) + "="*10 )
        print ("="*10 + " gradient norm below tolerance " + "="*10)
        plot(dj, title="dj", interactive=True)
        import ipdb; ipdb.set_trace()

rf = ReducedFunctional(J, control, derivative_cb_post=cb)

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