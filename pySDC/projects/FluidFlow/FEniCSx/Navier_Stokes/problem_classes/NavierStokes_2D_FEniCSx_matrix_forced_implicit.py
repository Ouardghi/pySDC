import logging

from mpi4py import MPI
from petsc4py import PETSc
import dolfinx as dfx
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import ufl
import basix
import numpy as np
import pyvista

from pySDC.core.problem import Problem
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


# noinspection PyUnusedLocal
class fenicsx_NSE_mass(Problem):
    r"""
    Example solving the incompressible Navier-Stokes equations for the DFG 2D-3 benchmark
    (flow around a cylinder) using FEniCSx and SDC from pySDC.

    .. math::
        \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} - \nu \Delta \mathbf{u} + \nabla p = 0,
        \quad \nabla \cdot \mathbf{u} = 0

    for a domain with a cylinder obstruction and inflow-outflow boundary conditions.

    Weak Formulation:
    .. math::
        \int_\Omega \frac{\partial \mathbf{u}}{\partial t} \cdot \mathbf{v}\,dx +
        \int_\Omega (\mathbf{u} \cdot \nabla) \mathbf{u} \cdot \mathbf{v}\,dx +
        \nu \int_\Omega \nabla \mathbf{u} : \nabla \mathbf{v}\,dx - \int_\Omega p \nabla \cdot \mathbf{v}\,dx = 0
        #
        \int_\Omega q \nabla \cdot \mathbf{u}\,dx = 0

    The spatial domain is discretized using continuous Lagrange elements, while the time-stepping scheme uses
    spectral deferred corrections (SDC) for high-order accuracy.

    Parameters
    ----------
    c_nvars : int, optional
        Spatial resolution, i.e., numbers of degrees of freedom in space.
    t0 : float, optional
        Starting time.
    family : str, optional
        Indicates the family of elements used to create the function space
        for the trail and test functions. The default is ``'CG'``, which are the class
        of Continuous Galerkin, a *synonym* for the Lagrange family of elements, see [2]_.
    order : int, optional
        Defines the order of the elements in the function space.
    refinements : int, optional
        Denotes the refinement of the mesh. ``refinements=2`` refines the mesh by factor :math:`2`.
    nu : float, optional
        Diffusion coefficient :math:`\nu`.
    rho : float, optional
        Fluid density.
    T : float, optional
        End time for the simulation.
    dt : float, optional
        Time step size.

    Attributes
    ----------
    V : FunctionSpace
        Defines the function space of the trial and test functions for velocity.
    Q : FunctionSpace
        Defines the function space of the trial and test functions for pressure
    M : scalar, vector, matrix or higher rank tensor
        Denotes the expression :math:`\int_\Omega u_t v\,dx`.
    K : scalar, vector, matrix or higher rank tensor
        Denotes the expression :math:`- \nu \int_\Omega \nabla u \nabla v\,dx`.
    g : Expression
        The forcing term :math:`f` in the Navier-Stokes equations.
    bcup : list
        Boundary conditions for the coupled velocity and pressure w = (u,p) in the monolithic scheme.
    bcu : list
        Boundary conditions for velocity.
    bcp : list
        Boundary conditions for pressure.
    u : Function
        Velocity field.
    p : Function
        Pressure field.

    Boundary Conditions:
    - Inlet: Parabolic velocity profile.
    - Walls/Cylinder: No-slip condition.
    - Outlet: Pressure outflow condition.

    References
    ----------
    .. [1] FEniCSx Project Documentation.
    .. [2] DFG 2D-3 Benchmark for Incompressible Flow Around a Cylinder
    .. [3] The FEniCS Project Version 1.5. M. S. Alnaes, J. Blechta, J. Hake, A. Johansson, B. Kehlet, A. Logg,
        C. Richardson, J. Ring, M. E. Rognes, G. N. Wells. Archive of Numerical Software (2015).
    .. [4] Automated Solution of Differential Equations by the Finite Element Method. A. Logg, K.-A. Mardal, G. N.
        Wells and others. Springer (2012).
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(
        self, nelems=128, t0=0.0, family='CG', order=4, refinements=1, nu=0.1, ofiles=None, comm=MPI.COMM_WORLD
    ):
        """Initialization routine"""

        # from dolfinx.cpp.log import set_log_level, LogLevel
        # set_log_level(LogLevel.INFO)

        domain, ct, ft = dfx.io.gmshio.read_from_msh("cylinder.msh", comm, 0, gdim=2)
        self.domain = domain

        Ve = basix.ufl.element(family, domain.topology.cell_name(), order, shape=(domain.geometry.dim,))
        Pe = basix.ufl.element(family, domain.topology.cell_name(), order - 1, shape=())
        We = basix.ufl.mixed_element([Ve, Pe])

        self.W = dfx.fem.functionspace(domain, We)
        self.V, _ = self.W.sub(0).collapse()
        self.Q, _ = self.W.sub(1).collapse()

        tmp = dfx.fem.Function(self.W)
        nx = len(tmp.x.array)
        print('DoFs on this level:', nx)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(init=(nx, comm, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'nelems', 't0', 'family', 'order', 'refinements', 'nu', 'ofiles', 'comm', localVars=locals(), readOnly=True
        )

        """
          Create boundary condition
        """
        # Define the boundary condition
        self.u_in = dfx.fem.Function(self.V)
        u_noslip = dfx.fem.Function(self.V)
        p_out = dfx.fem.Function(self.Q)

        #
        self.u_in.interpolate(self.InletVelocity(t0))
        p_out.interpolate(lambda x: np.full(x.shape[1], 0.0, dtype=np.float64))
        u_noslip.interpolate(
            lambda x: np.vstack(
                (np.full(x.shape[1], 0.0, dtype=np.float64), np.full(x.shape[1], 0.0, dtype=np.float64))
            )
        )
        # Get the facets dimension
        fdim = domain.topology.dim - 1
        #
        inlet = dfx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
        outlet = dfx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 2.2))
        walls = dfx.mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], 0.41)
        )
        cylin = dfx.mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.isclose((x[0] - 0.2) ** 2 + (x[1] - 0.2) ** 2, (0.05) ** 2)
        )
        #
        dofs_inlet = dfx.fem.locate_dofs_topological((self.W.sub(0), self.V), fdim, inlet)
        dofs_outlet = dfx.fem.locate_dofs_topological((self.W.sub(1), self.Q), fdim, outlet)
        dofs_walls = dfx.fem.locate_dofs_topological((self.W.sub(0), self.V), fdim, walls)
        dofs_cylin = dfx.fem.locate_dofs_topological((self.W.sub(0), self.V), fdim, cylin)
        #
        inlet_bc = dfx.fem.dirichletbc(self.u_in, dofs_inlet, self.W.sub(0))
        outlet_bc = dfx.fem.dirichletbc(p_out, dofs_outlet, self.W.sub(1))
        walls_bc = dfx.fem.dirichletbc(u_noslip, dofs_walls, self.W.sub(0))
        cylin_bc = dfx.fem.dirichletbc(u_noslip, dofs_cylin, self.W.sub(0))
        #
        self.bcup = [inlet_bc, walls_bc, outlet_bc, cylin_bc]

        # Define boundary condition for the residual
        bndry_facets = dfx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
        #
        dofs_bndry_p = dfx.fem.locate_dofs_topological((self.W.sub(1), self.Q), fdim, bndry_facets)
        dofs_bndry_u = dfx.fem.locate_dofs_topological((self.W.sub(0), self.V), fdim, bndry_facets)
        #
        bc_hom_p = dfx.fem.dirichletbc(p_out, dofs_bndry_p, self.W.sub(1))
        bc_hom_u = dfx.fem.dirichletbc(u_noslip, dofs_bndry_u, self.W.sub(0))
        #
        self.bc_hom = [bc_hom_u, bc_hom_p]
        self.fix_bc_for_residual = True

        #
        Obs = dfx.mesh.meshtags(domain, domain.topology.dim - 1, cylin, np.full_like(cylin, 5, dtype=np.intc))
        self.dObs = ufl.Measure("ds", domain=domain, subdomain_data=Obs, subdomain_id=5)

        # Define trial and test functions
        self.u, self.p = ufl.TrialFunctions(self.W)
        self.v, self.q = ufl.TestFunctions(self.W)

        # Assemble mass matrices
        '''
                | M_v   0 |                  | M_v    0  |
            M = |         |      and    M2 = |           |
                |  0    0 |                  |  0    M_p |

        where M_v and M_p are the mass matrix in the velocity and pressure spaces, respectively.
        '''
        a_M = ufl.dot(self.u, self.v) * ufl.dx
        self.M = dolfinx.fem.petsc.assemble_matrix(dfx.fem.form(a_M), bcs=self.bcup)
        self.M.assemble()

        a_M2 = ufl.dot(self.u, self.v) * ufl.dx + ufl.dot(self.p, self.q) * ufl.dx
        self.M2 = dolfinx.fem.petsc.assemble_matrix(dfx.fem.form(a_M2), bcs=self.bcup)
        self.M2.assemble()

        self.tmp_u = dfx.fem.Function(self.W)
        self.tmp_f = dfx.fem.Function(self.W)

        self.g = dfx.fem.Function(self.V)
        self.rhs = dfx.fem.Function(self.W)

        # self.Lin_solver = PETSc.KSP().create(domain.comm)
        # self.Lin_solver.setType(PETSc.KSP.Type.PREONLY)
        # self.Lin_solver.getPC().setType(PETSc.PC.Type.LU)

        self.Lin_solver = PETSc.KSP().create(domain.comm)
        self.Lin_solver.setType(PETSc.KSP.Type.BCGS)
        self.Lin_solver.setTolerances(rtol=1e-15, atol=1e-15)
        pc = self.Lin_solver.getPC()
        pc.setType(PETSc.PC.Type.JACOBI)

        # ------------------------------------

        self.factor = dfx.fem.Constant(domain, PETSc.ScalarType(0.0))

        self.w = dfx.fem.Function(self.W)
        u, p = ufl.split(self.w)

        rhs_u, rhs_p = ufl.split(self.rhs)

        F = ufl.dot(u, self.v) * ufl.dx
        F += self.factor * ufl.dot(ufl.dot(u, ufl.nabla_grad(u)), self.v) * ufl.dx
        F += self.factor * ufl.inner(self.nu * ufl.nabla_grad(u), ufl.nabla_grad(self.v)) * ufl.dx
        F -= self.factor * ufl.dot(p, ufl.div(self.v)) * ufl.dx
        F -= self.factor * ufl.dot(self.g, self.v) * ufl.dx
        F -= self.factor * ufl.dot(ufl.div(u), self.q) * ufl.dx
        F -= ufl.dot(rhs_u, self.v) * ufl.dx
        F -= ufl.dot(rhs_p, self.q) * ufl.dx

        problem = dolfinx.fem.petsc.NonlinearProblem(F, self.w, self.bcup)
        self.solver = dolfinx.nls.petsc.NewtonSolver(domain.comm, problem)
        self.solver.convergence_criterion = "residual"  # "residual" #"incremental"
        self.solver.rtol = 1e-12
        self.solver.atol = 1e-12
        self.solver.report = True
        self.solver.max_it = 10
        self.solver.error_on_nonconvergence = True

        ksp = self.solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "preonly"  # "gmres" #"preonly"
        opts[f"{option_prefix}pc_type"] = "lu"  # "gamg" # "lu"
        opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
        ksp.setFromOptions()

        # ------------------------------------

        # self.plotter = pyvista.Plotter(shape=(2, 1))
        # self.plotter.show(interactive_update=True)

    @staticmethod
    def convert_to_fenicsx_vector(input, output):
        output.x.array[:] = input[:]

    @staticmethod
    def convert_from_fenicsx_vector(input, output):
        output[:] = input.x.array[:]

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Dolfin's linear solver for :math:`(M - factor \cdot A) \vec{u} = \vec{rhs}`.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver (not used here so far).
        t : float
            Current time.

        Returns
        -------
        u : dtype_u
            Solution.
        """

        u = self.dtype_u(self.init)
        tmp_f = dfx.fem.Function(self.W)
        self.factor.value = factor
        #
        self.u_in.interpolate(self.InletVelocity(t))
        self.g.interpolate(self.source_term(t))
        #
        self.convert_to_fenicsx_vector(input=rhs, output=tmp_f)
        dolfinx.fem.petsc.set_bc(tmp_f.x.petsc_vec, self.bcup)
        tmp_f = self.__invert_mass_matrix(tmp_f)
        #
        with tmp_f.x.petsc_vec.localForm() as loc_f, self.rhs.x.petsc_vec.localForm() as loc_rhs:
            loc_f.copy(loc_rhs)
        #
        self.solver.solve(self.w)
        self.w.x.scatter_forward()
        #
        self.convert_from_fenicsx_vector(input=self.w, output=u)
        #
        return u

    def eval_f(self, w, t):
        """
        Routine to evaluate both parts of the right-hand side of the problem.

        Parameters
        ----------
        w : dtype_u
            Current values of the numerical solution.
        t : float
            Current time at which the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side divided into two parts.
        """

        f = self.dtype_f(self.init)
        tmp_u = dfx.fem.Function(self.W)
        tmp_f = dfx.fem.Function(self.W)

        self.convert_to_fenicsx_vector(input=w, output=tmp_u)

        u, p = ufl.split(tmp_u)

        # Get the forcing term
        g = self.source_term(t)

        F = -ufl.dot(ufl.dot(u, ufl.nabla_grad(u)), self.v) * ufl.dx
        F -= ufl.inner(self.nu * ufl.nabla_grad(u), ufl.nabla_grad(self.v)) * ufl.dx
        F += ufl.dot(p, ufl.div(self.v)) * ufl.dx
        F += ufl.dot(g, self.v) * ufl.dx
        F += ufl.dot(ufl.div(u), self.q) * ufl.dx

        b = dolfinx.fem.petsc.assemble_vector(dfx.fem.form(F))
        b.assemble()
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

        #
        with b.localForm() as loc_b, tmp_f.x.petsc_vec.localForm() as loc_f:
            loc_b.copy(loc_f)

        # dolfinx.fem.petsc.set_bc(tmp_f.x.petsc_vec, self.bcup)
        self.convert_from_fenicsx_vector(input=tmp_f, output=f)

        return f

    def apply_mass_matrix(self, u):
        r"""
        Routine to apply mass matrix.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.

        Returns
        -------
        uM : dtype_u
            The product :math:`M \vec{u}`.
        """

        self.convert_to_fenicsx_vector(input=u, output=self.tmp_u)
        self.M.mult(self.tmp_u.x.petsc_vec, self.tmp_f.x.petsc_vec)
        uM = self.dtype_u(self.init)
        self.convert_from_fenicsx_vector(input=self.tmp_f, output=uM)
        return uM

    def source_term(self, t):
        r"""
        Routine to compute the source term at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        g : dtype_u
            Source term.
        """

        g = dfx.fem.Function(self.V)
        g.interpolate(
            lambda x: np.vstack(
                (np.full(x.shape[1], 0.0, dtype=np.float64), np.full(x.shape[1], 0.0, dtype=np.float64))
            )
        )

        return g

    def InletVelocity(self, t):
        r"""
        Routine to compute the inlet velocity profile at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        u_in : dtype_u
               inlet velocity profile.
        """
        u_in = dfx.fem.Function(self.V)
        u_in.interpolate(
            lambda x: np.vstack(
                (
                    4 * 1.5 * np.sin(t * np.pi / 8) * x[1] * (0.41 - x[1]) / (0.41**2),
                    np.full(x.shape[1], 0.0, dtype=np.float64),
                )
            )
        )

        return u_in

    def u_exact(self, t):
        r"""
        Routine to compute the exact solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            Exact solution.
        """

        w0 = dfx.fem.Function(self.W)
        w0.sub(0).interpolate(
            lambda x: np.vstack(
                (np.full(x.shape[1], 0.0, dtype=np.float64), np.full(x.shape[1], 0.0, dtype=np.float64))
            )
        )
        w0.sub(1).interpolate(lambda x: np.full(x.shape[1], 0.0, dtype=np.float64))

        me = self.dtype_u(self.init)
        self.convert_from_fenicsx_vector(input=w0, output=me)

        return me

    def fix_residual(self, res):
        """
        Applies homogeneous Dirichlet boundary conditions to the residual

        Parameters
        ----------
        res : dtype_u
              Residual
        """
        self.convert_to_fenicsx_vector(input=res, output=self.tmp_u)
        dolfinx.fem.petsc.set_bc(self.tmp_u.x.petsc_vec, self.bc_hom)
        self.tmp_u.x.scatter_forward()
        self.convert_from_fenicsx_vector(input=self.tmp_u, output=res)
        return None

    def __invert_mass_matrix(self, u):
        r"""
        Helper routine to invert mass matrix.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.

        Returns
        -------
        me : dtype_u
            The product :math:`M^{-1} \vec{u}`.
        """
        solver = PETSc.KSP().create(self.domain.comm)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)

        solver.setOperators(self.M2)
        solver.solve(u.x.petsc_vec, self.tmp_f.x.petsc_vec)
        self.tmp_f.x.scatter_forward()

        return self.tmp_f

    def plot(self, w):
        r"""
        Plot solutions.

        Note
        ----
        This method plots the solutions using pyvista.

        Parameters
        ----------
        w : dtype_u
            Current values of the numerical solution.
        """

        self.convert_to_fenicsx_vector(input=w, output=self.tmp_u)

        u, p = self.tmp_u.split()
        u_ = u.collapse()
        p_ = p.collapse()

        import time

        # Plotting
        # ------------------------- Velocity ---------------------------------
        cells, types, x = dfx.plot.vtk_mesh(self.V)
        gridu = pyvista.UnstructuredGrid(cells, types, x)
        #
        values = np.zeros((x.shape[0], 3), dtype=np.float64)
        values[:, : len(u_)] = u_.x.array.real.reshape((x.shape[0], len(u_)))
        gridu["u"] = values
        glyphs = gridu.glyph(orient="u", factor=0.08)
        #
        # ------------------------- pressure ---------------------------------
        cells, types, x = dfx.plot.vtk_mesh(self.Q)
        gridp = pyvista.UnstructuredGrid(cells, types, x)
        #
        gridp.point_data["p"] = p_.x.array.real
        gridp.set_active_scalars("p")
        #

        plotter = self.plotter

        plotter.subplot(0, 0)
        # plotter.add_mesh(gridu, style="wireframe", color="k")
        plotter.add_mesh(glyphs, cmap='jet')
        plotter.view_xy()
        plotter.camera.zoom(2.7)
        #
        plotter.subplot(1, 0)
        plotter.add_mesh(gridp, show_edges=True, cmap='jet')
        plotter.view_xy()
        plotter.camera.zoom(2.7)

        # plotter.show()

    def LiftDrag(self, w, t):
        r"""
        Compute the drag and lift coefficients over the obstacle.

        Parameters
        ----------
        w : dtype_u
            Current values of the numerical solution.

        Returns
        -------
        d_cf : scalar
               drag coefficient.

        l_cf : scalar
               lift coefficient.
        """

        # Convert the mesh vector w to a FEniCSx function
        self.convert_to_fenicsx_vector(input=w, output=self.tmp_u)

        # split the vector into velocity and pressure
        u_, p_ = ufl.split(self.tmp_u)

        rho = 1

        # Normal pointing out of obstacle
        n = -ufl.FacetNormal(self.domain)

        # Tangential velocity component at the interface of the obstacle
        u_t = ufl.inner(ufl.as_vector((n[1], -n[0])), u_)

        # compute the drag and lift coefficients
        drag = dfx.fem.form(2 / 0.1 * (self.nu / rho * ufl.inner(ufl.grad(u_t), n) * n[1] - p_ * n[0]) * self.dObs)
        lift = dfx.fem.form(-2 / 0.1 * (self.nu / rho * ufl.inner(ufl.grad(u_t), n) * n[0] + p_ * n[1]) * self.dObs)

        # Assemble the scalar values
        d_cf = dfx.fem.assemble_scalar(drag)
        l_cf = dfx.fem.assemble_scalar(lift)

        # Open the drag and lift file and add the currentvalues
        f = open(self.ofiles[0], 'a')
        out = f'{t} {l_cf} {d_cf}'
        f.write(out + '\n')

        return None
