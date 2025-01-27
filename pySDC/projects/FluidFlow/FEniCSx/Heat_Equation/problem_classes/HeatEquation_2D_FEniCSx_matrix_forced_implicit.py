import logging

from mpi4py import MPI
from petsc4py import PETSc
import dolfinx as dfx
import dolfinx.fem.petsc
import ufl
import numpy as np
import pyvista

from pySDC.core.problem import Problem
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


# noinspection PyUnusedLocal
class fenicsx_heat_mass(Problem):
    r"""
    Example implementing the forced one-dimensional heat equation with Dirichlet boundary conditions

    .. math::
        \frac{d u}{d t} = \nu \frac{d^2 u}{d x^2} + f

    for :math:`x \in \Omega:=[0,1]`, where the forcing term :math:`f` is defined by

    .. math::
        f(x, t) = -\sin(\pi x) (\sin(t) - \nu \pi^2 \cos(t)).

    For initial conditions with constant c and

    .. math::
        u(x, 0) = \sin(\pi x) + lc

    the exact solution of the problem is given by

    .. math::
        u(x, t) = \sin(\pi x)\cos(t) + c.

    In this class the problem is implemented in the way that the spatial part is solved using ``FEniCS`` [1]_.
    Hence, the problem is reformulated to the *weak formulation*

    .. math:
        \int_\Omega u_t v\,dx = - \nu \int_\Omega \nabla u \nabla v\,dx + \int_\Omega f v\,dx.

    The part containing the forcing term is treated explicitly, where it is interpolated in the function space.
    The other part will be treated in an implicit way.

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
    c: float, optional
        Constant for the Dirichlet boundary condition :math: `c`

    Attributes
    ----------
    V : FunctionSpace
        Defines the function space of the trial and test functions.
    M : scalar, vector, matrix or higher rank tensor
        Denotes the expression :math:`\int_\Omega u_t v\,dx`.
    K : scalar, vector, matrix or higher rank tensor
        Denotes the expression :math:`- \nu \int_\Omega \nabla u \nabla v\,dx`.
    g : Expression
        The forcing term :math:`f` in the heat equation.
    bc : DirichletBC
        Denotes the Dirichlet boundary conditions.

    References
    ----------
    .. [1] The FEniCS Project Version 1.5. M. S. Alnaes, J. Blechta, J. Hake, A. Johansson, B. Kehlet, A. Logg,
        C. Richardson, J. Ring, M. E. Rognes, G. N. Wells. Archive of Numerical Software (2015).
    .. [2] Automated Solution of Differential Equations by the Finite Element Method. A. Logg, K.-A. Mardal, G. N.
        Wells and others. Springer (2012).
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars=128, t0=0.0, family='CG', order=4, refinements=0, nu=0.1, c=0.0, comm=MPI.COMM_WORLD):
        """Initialization routine"""

        self.comm = comm

        domain = dfx.mesh.create_rectangle(
            comm,
            points=((0.0, 0.0), (1.0, 1.0)),
            n=(nvars, nvars),
            cell_type=dfx.mesh.CellType.triangle,
        )

        domain.topology.create_entities(1)

        # for _ in range(refinements):
        #    domain = dfx.mesh.refine(domain)[0]

        self.V = dfx.fem.functionspace(domain, (family, order))
        self.x = ufl.SpatialCoordinate(domain)

        tmp = dfx.fem.Function(self.V)
        nx = len(tmp.x.array)
        print('DoFs on this level:', nx)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(init=(nx, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'nvars', 't0', 'family', 'order', 'refinements', 'nu', 'c', localVars=locals(), readOnly=True
        )

        # Create boundary condition
        fdim = domain.topology.dim - 1
        boundary_facets = dfx.mesh.locate_entities_boundary(
            domain,
            fdim,
            lambda x: np.full(x.shape[1], True, dtype=bool),
        )
        self.bc = dfx.fem.dirichletbc(
            PETSc.ScalarType(self.c),
            dfx.fem.locate_dofs_topological(self.V, fdim, boundary_facets),
            self.V,
        )
        self.bc_hom = dfx.fem.dirichletbc(
            PETSc.ScalarType(0), dfx.fem.locate_dofs_topological(self.V, fdim, boundary_facets), self.V
        )
        self.fix_bc_for_residual = True

        # Stiffness term (Laplace) and mass term
        self.u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)

        a_K = -1.0 * ufl.dot(ufl.grad(self.u), self.nu * ufl.grad(self.v)) * ufl.dx
        a_M = self.u * self.v * ufl.dx

        self.K = dolfinx.fem.petsc.assemble_matrix(dfx.fem.form(a_K), bcs=[self.bc])
        self.K.assemble()

        self.M = dolfinx.fem.petsc.assemble_matrix(dfx.fem.form(a_M), bcs=[self.bc])
        self.M.assemble()

        # set forcing term
        self.g = dfx.fem.Function(self.V)
        t = self.t0
        # self.g.interpolate(
        #    lambda x: -np.sin(np.pi * x[0]) * (np.sin(t) -  self.nu * np.pi * np.pi * np.cos(t)))

        self.tmp_u = dfx.fem.Function(self.V)
        self.tmp_f = dfx.fem.Function(self.V)
        self.tmp_g = dfx.fem.Function(self.V)
        self.tmp_rhs = dfx.fem.Function(self.V)

        # self.solver = PETSc.KSP().create(domain.comm)
        # self.solver.setType(PETSc.KSP.Type.PREONLY)
        # self.solver.getPC().setType(PETSc.PC.Type.LU)

        """
        self.solver = PETSc.KSP().create(mesh.comm)
        self.solver.setType(PETSc.KSP.Type.MINRES)
        self.solver.setTolerances(rtol=1e-10, atol=1e-10)
        pc = self.solver.getPC()
        pc.setType(PETSc.PC.Type.HYPRE)
        pc.setHYPREType("boomeramg")
        """

        # Solver
        self.solver = PETSc.KSP().create(mesh.comm)
        self.solver.setType(PETSc.KSP.Type.BCGS)
        self.solver.setTolerances(rtol=1e-10, atol=1e-10)
        pc = self.solver.getPC()
        pc.setType(PETSc.PC.Type.JACOBI)

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

        # f = self.dtype_f(self.init)
        g = self.source_term(t)
        tmp_f = dfx.fem.Function(self.V)
        # rhs1 = dfx.fem.Function(self.W)

        self.convert_to_fenicsx_vector(input=u0, output=self.tmp_u)
        #

        self.convert_to_fenicsx_vector(input=rhs, output=tmp_f)
        dolfinx.fem.petsc.set_bc(tmp_f.x.petsc_vec, [self.bc])
        tmp_f = self.__invert_mass_matrix(tmp_f)

        #
        F = self.u * self.v * ufl.dx
        F += factor * ufl.dot(ufl.grad(self.u), self.nu * ufl.grad(self.v)) * ufl.dx
        F -= factor * g * self.v * ufl.dx
        F -= tmp_f * self.v * ufl.dx
        #
        A = dolfinx.fem.petsc.assemble_matrix(dfx.fem.form(ufl.lhs(F)), bcs=[self.bc])
        b = dolfinx.fem.petsc.assemble_vector(dfx.fem.form(ufl.rhs(F)))
        #
        A.assemble()
        b.assemble()
        #
        dolfinx.fem.petsc.set_bc(b, [self.bc])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

        self.solver.setOperators(A)
        self.solver.solve(b, self.tmp_u.x.petsc_vec)

        self.tmp_u.x.scatter_forward()

        u = self.dtype_u(self.init)
        self.convert_from_fenicsx_vector(input=self.tmp_u, output=u)

        return u

    def eval_f(self, u, t):
        """
        Routine to evaluate both parts of the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time at which the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side divided into two parts.
        """

        f1 = self.dtype_f(self.init)
        f2 = self.dtype_f(self.init)
        f = self.dtype_f(self.init)

        g = self.source_term(t)
        #
        self.convert_to_fenicsx_vector(input=u, output=self.tmp_u)
        #
        self.K.mult(self.tmp_u.x.petsc_vec, self.tmp_f.x.petsc_vec)
        self.M.mult(g.x.petsc_vec, self.tmp_g.x.petsc_vec)
        #
        self.convert_from_fenicsx_vector(input=self.tmp_f, output=f1)
        self.convert_from_fenicsx_vector(input=self.tmp_g, output=f2)

        f = f1 + f2

        """
        f  = self.dtype_f(self.init)
        g = self.source_term(t)
        #
        self.convert_to_fenicsx_vector(input=u, output=self.tmp_u)
        #
        F  = -1.0 * ufl.dot(ufl.grad(self.tmp_u), self.nu * ufl.grad(self.v)) * ufl.dx
        F += g * self.v * ufl.dx
        #
        rhs = dolfinx.fem.petsc.assemble_vector(dfx.fem.form(F))
        rhs.assemble()

        dolfinx.fem.petsc.set_bc(rhs, [self.bc])
        #
        rhs.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        print(type(rhs))
        f[:] = rhs[:]
        #f[:] = rhs.getArrayRead()
        """
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
        me : dtype_u
            Exact solution.
        """

        g = dfx.fem.Function(self.V)
        g.interpolate(
            lambda x: -np.sin(np.pi * x[0])
            * np.sin(np.pi * x[1])
            * (np.sin(t) - 2 * self.nu * np.pi * np.pi * np.cos(t))
        )

        return g

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

        u0 = dfx.fem.Function(self.V)
        u0.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.cos(t) + self.c)

        me = self.dtype_u(self.init)
        self.convert_from_fenicsx_vector(input=u0, output=me)

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
        dolfinx.fem.petsc.set_bc(self.tmp_u.x.petsc_vec, [self.bc_hom])
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

        # self.convert_to_fenicsx_vector(input=u, output=self.tmp_u)
        # dolfinx.fem.petsc.set_bc(self.tmp_f.x.petsc_vec, [self.bc])
        self.solver.setOperators(self.M)
        self.solver.solve(u.x.petsc_vec, self.tmp_f.x.petsc_vec)

        self.tmp_f.x.scatter_forward()

        # me = self.dtype_u(self.init)
        # self.convert_from_fenicsx_vector(input=self.tmp_f, output=me)

        return self.tmp_f

    def plot(self, w, t):
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
        uex = self.u_exact(t)

        self.convert_to_fenicsx_vector(input=w, output=self.tmp_u)
        self.convert_to_fenicsx_vector(input=uex, output=self.tmp_f)

        u = self.tmp_u
        ue = self.tmp_f
        # Plotting

        cells, types, x = dfx.plot.vtk_mesh(self.V)
        grid = pyvista.UnstructuredGrid(cells, types, x)
        #
        grid.point_data["un"] = u.x.array.real
        grid.set_active_scalars("un")
        warped = grid.warp_by_scalar("un")

        grid.point_data["ue"] = ue.x.array.real
        grid.set_active_scalars("ue")
        warped = grid.warp_by_scalar("ue")
        #

        """
        plotter = pyvista.Plotter()
        plotter.show(interactive_update=True)
        #plotter = self.plotter
        #plotter.subplot(0, 0)
        plotter.add_mesh(warped, scalars="un", show_edges=True,cmap='jet')
        #plotter.view_xy()
        plotter.camera.zoom(2.7)
        plotter.show()
        print("Plotted")
        """

        sargs = dict(
            height=0.8,
            width=0.1,
            vertical=True,
            position_x=0.05,
            position_y=0.05,
            fmt="%1.2e",
            title_font_size=40,
            color="black",
            label_font_size=25,
        )

        subplotter = pyvista.Plotter(shape=(1, 2))
        subplotter.subplot(0, 0)
        subplotter.add_text("Numerical solution", font_size=14, color="black", position="upper_edge")
        # subplotter.add_mesh(grid, scalars="un", show_edges=False, show_scalar_bar=True, cmap='jet')
        # subplotter.view_xy()
        # subplotter.set_position([-3, 2.6, 0.3])
        # subplotter.set_focus([3, -1, 0.2])
        # subplotter.set_viewup([0, 0, 1])
        subplotter.add_mesh(warped, scalars="un", show_edges=False, scalar_bar_args=sargs, cmap='jet')
        subplotter.view_xy()

        subplotter.subplot(0, 1)
        subplotter.add_text("Exact solution", position="upper_edge", font_size=14, color="black")
        # subplotter.set_position([-3, 2.6, 0.3])
        # subplotter.set_focus([3, -1, 0.2])
        # subplotter.set_viewup([0, 0, 1])
        subplotter.add_mesh(warped, scalars="ue", show_edges=False, scalar_bar_args=sargs, cmap='jet')
        subplotter.view_xy()

        subplotter.show()
