import numpy as np
import time
from pyadjoint import Block, MinimizationProblem, TAOSolver, get_working_tape
import netgen
from netgen.geom2d import SplineGeometry
from firedrake import *
from firedrake.adjoint import *
import argparse

"""
Library versions:
    
    numpy - 2.3.0
    scipy - 1.15.3
    firedrake - 2025.4.1
    pyadjoint-ad - 2025.4.0
    petsc4py - 3.23.3
"""
# Create mesh using netgen
geo = SplineGeometry()
geo.AddRectangle(p1=(-1, -1),
                 p2=(1, 1),
                 bc="rectangle",
                 leftdomain=1,
                 rightdomain=0)

ngmsh = geo.GenerateMesh(maxh=0.1)


def mark_center(mesh):
    # Used to refine the center of mesh
        W = FunctionSpace(mesh, "DG", 0)
        mark = Function(W)
        x, y = SpatialCoordinate(mesh)
        center = conditional( And((abs(x) < 0.4) , (abs(y) < 0.4)), 1.0, 0.0 )
        mark.interpolate(center)
        return mark


# Uniform refine
# n = 16
# mesh = UnitSquareMesh(n, n)
# hierarchy = MeshHierarchy(mesh, 4)
# mesh = hierarchy[-1]


def run(opt_package="tao_lmvm",
        riesz_rep="L2",
        max_level=4):
    
    """
    Parameters
    ----------
    solver_name : str
        Select optimisation package: 'tao_lmvm', 'tao_nls', 'scipy'
    riesz_rep : str
        Inner product: 'L2', 'l2'
    max_level : int
        Maximum refine level
    """

    for level in range(max_level + 1):
        tape = get_working_tape()
        tape.clear_tape() 
        continue_annotation()
        if level == 0:
            # Origin mesh
            print(f"Refinement level: {level}")
            mesh = Mesh(ngmsh)      
            VTKFile("output/Mesh_Origin_uniform.pvd").write(mesh)
        else:
            # Perform refinement based on former mesh
            print(f"Refinement level: {level}")
            mark = mark_center(mesh)
            mesh = mesh.refine_marked_elements(mark)
            VTKFile(f"output/refined_central_{level}.pvd").write(mesh)


        # Define discrete function spaces and funcions
        V = FunctionSpace(mesh, "CG", 1)
        W = FunctionSpace(mesh, "CG", 1)

        f = Function(W)
        f.assign(0.0)
        u = TrialFunction(V)
        v = TestFunction(V)

        # Define and solve the Poisson equation to generate the dolfin-adjoint annotation
        bc = DirichletBC(V, 0.0, "on_boundary")
        a = inner(grad(u), grad(v)) * dx
        L = inner(f, v) * dx
        u = Function(V)
        solve(a == L, u, bc)

        # Define regularisation parameter
        alpha = Constant(1e-6)

        # Define the expressions of the analytical solution
        x, y = SpatialCoordinate(mesh)
        d = Function(W)
        d.interpolate(1 / (2 * pi **2) * sin(pi * x) * sin(pi * y))
        # f_analytic = assemble(sin(pi*x)*sin(pi*y))
        # u_analytic = assemble(1/(2*pi*pi)*sin(pi*x)*sin(pi*y))
        # j_analytic = assemble((1./2*(u_analytic-d)**2 + alpha*f_analytic**2)*dx(mesh))

        # Define functional of interest and the reduced functional
        #ctrl_inner_product = "L2"
        regularisation_norm = "L2"

        if regularisation_norm == "L2":
            J = assemble((0.5*inner(u-d, u-d))*dx + alpha/2*f**2*dx)
        elif regularisation_norm == "H1":
            J = assemble((0.5*inner(u-d, u-d))*dx + alpha/2*((grad(f)**2)*dx + f**2*dx))
        elif regularisation_norm == "H0_1":
            J = assemble((0.5*inner(u-d, u-d))*dx + alpha/2*(grad(f)**2)*dx)
        elif regularisation_norm == "H2":
            J = assemble((0.5*inner(u-d, u-d))*dx + alpha/2*((grad(grad(f))**2)*dx + (grad(f)**2)*dx + f**2*dx))
        else:
            raise ValueError("Choose control inner product as required")

        control = Control(f)
        rf = ReducedFunctional(J, control)
        pause_annotation()


        convert_options = {"riesz_representation": riesz_rep}

        if opt_package == "tao_lmvm":
            problem = MinimizationProblem(rf)
            solver = TAOSolver(problem, {"tao_type": "lmvm",
                                        "tao_gatol": 1.0e-6,
                                        "tao_grtol": 0.0,
                                        "tao_gttol": 0.0,
                                        #"tao_monitor": None,
                                        #"tao_view": None
                                        },
                            convert_options=convert_options)
            start_time = time.time()
            f_opt = solver.solve()
            # solver.tao.view()
            runtime = time.time() - start_time
            iter = solver.tao.getIterationNumber()

        elif opt_package == "tao_nls":
            problem = MinimizationProblem(rf)
            solver = TAOSolver(problem, {"tao_type": "nls",
                                        "tao_gatol": 1.0e-6,
                                        "tao_grtol": 0.0,
                                        "tao_gttol": 0.0,
                                        #"tao_monitor": None,
                                        #"tao_view": None
                                        },
                            convert_options=convert_options)
            start_time = time.time()
            f_opt = solver.solve()
            # solver.tao.view()
            runtime = time.time() - start_time
            iter = solver.tao.getIterationNumber()

        elif opt_package == "tao_bncg":
            problem = MinimizationProblem(rf)
            solver = TAOSolver(problem, {"tao_type": "bncg",
                                        "tao_gatol": 1.0e-6,
                                        "tao_grtol": 0.0,
                                        "tao_gttol": 0.0,
                                        "tao_monitor": None,
                                        "tao_view": None},
                            convert_options=convert_options)
            start_time = time.time()
            f_opt = solver.solve()
            # solver.tao.view()
            runtime = time.time() - start_time
            iter = solver.tao.getIterationNumber()

        elif opt_package == "tao_ntr":
            problem = MinimizationProblem(rf)
            solver = TAOSolver(problem, {"tao_type": "ntr",
                                        "tao_gatol": 1.0e-6,
                                        "tao_grtol": 0.0,
                                        "tao_gttol": 0.0,
                                        "tao_monitor": None,
                                        "tao_view": None},
                            convert_options=convert_options)
            start_time = time.time()
            f_opt = solver.solve()
            # solver.tao.view()
            runtime = time.time() - start_time
            iter = solver.tao.getIterationNumber()

        elif opt_package == "scipy":
            counter = {"nit": 0}
            def cb(x):
                counter["nit"] += 1
            start_time = time.time()
            f_opt = minimize(rf, tol=1.0e-6, method="L-BFGS-B", options={"maxiter": 15000, 'maxcor': 10}, callback = cb)
            runtime = time.time() - start_time
            iter = counter["nit"]

        #plot(f_opt, title="f_opt", interactive=True)

        j = rf(f_opt)
        dj = rf.derivative()
        h  = mesh.cell_sizes               
        hmin = h.dat.data_ro.min()    
        hmax = h.dat.data_ro.max()

        print(f"Operating {opt_package}")
        print ("Final:   \tJ = %s\t |dJ|_L2 = %s" % (j, norm(dj)))
        print ("=================================")
        print(f"h(min):              {hmin:.6e}")
        print(f"h(max):              {hmax:.6e}")
        print(f"h(max)/h(min:        {hmax/hmin:.6e}")
        print("=================================")
        print(f"Iteration count:           {iter}")
        print(f"Run time:                  {runtime}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PDE optimal control demo with selectable optimisers"
    )
    parser.add_argument("--opt", "-o",
                        default="tao_lmvm",
                        choices=["tao_lmvm", "tao_nls", "tao_bncg", "tao_ntr", "scipy"],
                        help="Select optimisation package (Default tao_lmvm)")
    parser.add_argument("--riesz", "-r",
                        default="L2",
                        help="Select inner product (Default: L2)")
    parser.add_argument("--max_level", "-m",
                        type=int,
                        default=4,
                        help="Maximum refinement level")
    args = parser.parse_args()

    run(opt_package=args.opt,
        riesz_rep=args.riesz,
        max_level=args.max_level)