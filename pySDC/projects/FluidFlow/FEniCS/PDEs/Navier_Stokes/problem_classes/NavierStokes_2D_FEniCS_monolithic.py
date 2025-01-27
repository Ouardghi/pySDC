import logging

import dolfin as df
import numpy as np

import matplotlib.pyplot as plt

from pySDC.core.problem import Problem
from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh, rhs_fenics_mesh


# noinspection PyUnusedLocal
class fenics_NSE_2D_Monolithic(Problem):
    r"""
    Example implementing the forced two-dimensional incompressible Navier-Stokes equations with time-dependent Dirichlet boundary
    conditions

    .. math::
        \frac{d u}{d t} = - u \cdot \nabla u  + \nu \nabla u - \nabla p + f
                      0 = \nabla \cdot u

    for :math:`x \in \Omega`, where the forcing term :math:`f` is defined by

    .. math::
        f(x, t) = (0, 0).

    and the boundary conditions are given by

    .. math::
        u(x, t) =

    In this class the problem is implemented in the way that the spatial part is solved using ``FEniCS`` [1]_. Hence, the problem
    is reformulated to the *weak formulation*

    .. math:
        \int_\Omega u_t v\,dx = -

    The forcing term is treated explicitly, and is expressed via the mass matrix resulting from the left-hand side term
    :math:`\int_\Omega u_t v\,dx`, and the other part will be treated in an implicit way.

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
        Denotes the time-dependent Dirichlet boundary conditions.
    bc_hom : DirichletBC
        Denotes the homogeneous Dirichlet boundary conditions, potentially required for fixing the residual
    fix_bc_for_residual: boolean
        flag to indicate that the residual requires special treatment due to boundary conditions

    References
    ----------
    .. [1] The FEniCS Project Version 1.5. M. S. Alnaes, J. Blechta, J. Hake, A. Johansson, B. Kehlet, A. Logg,
        C. Richardson, J. Ring, M. E. Rognes, G. N. Wells. Archive of Numerical Software (2015).
    .. [2] Automated Solution of Differential Equations by the Finite Element Method. A. Logg, K.-A. Mardal, G. N.
        Wells and others. Springer (2012).
    """

    dtype_u = fenics_mesh
    dtype_f = fenics_mesh
    # dtype_f = rhs_fenics_mesh

    def __init__(self, c_nvars=64, t0=0.0, family='CG', order=2, refinements=1, nu=0.001, c=0.0, sigma=0.05):
        """Initialization routine"""
        self.fix_bc_for_residual = True

        # define the Dirichlet boundary
        def Boundary(x, on_boundary):
            return on_boundary

        # set logger level for FFC and dolfin
        logging.getLogger('FFC').setLevel(logging.WARNING)
        logging.getLogger('UFL').setLevel(logging.WARNING)

        # set solver and form parameters
        df.parameters["form_compiler"]["optimize"] = True
        df.parameters["form_compiler"]["cpp_optimize"] = True
        df.parameters['allow_extrapolation'] = True

        # set mesh and refinement (for multilevel)
        # domain = Rectangle(df.Point(0.0, 0.0), df.Point(1.0, 1.0))
        # mesh = generate_mesh(domain, c_nvars)

        mesh = df.Mesh('cylinder.xml')

        # for _ in range(refinements):
        #    mesh = df.refine(mesh)

        n = df.FacetNormal(mesh)

        # define function spaces for future reference (Taylor-Hood)
        P2 = df.VectorElement("P", mesh.ufl_cell(), order)
        P1 = df.FiniteElement("P", mesh.ufl_cell(), order - 1)

        TH = df.MixedElement([P2, P1])

        self.W = df.FunctionSpace(mesh, TH)
        self.V = df.FunctionSpace(mesh, P2)
        self.Q = df.FunctionSpace(mesh, P1)

        tmp = df.Function(self.W)
        print('DoFs on this level:', len(tmp.vector()[:]))

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(fenics_NSE_2D_Monolithic, self).__init__(self.W)
        self._makeAttributeAndRegister(
            'c_nvars', 't0', 'family', 'order', 'refinements', 'nu', 'c', 'sigma', localVars=locals(), readOnly=True
        )

        # Trial and test function for the Mixed FE space
        self.u, self.p = df.TrialFunctions(self.W)
        self.v, self.q = df.TestFunctions(self.W)

        """
        # Trial and test functions for the velocity space
        u = df.TrialFunction(self.V)
        v = df.TestFunction(self.V)
        """
        # Mass term
        a_M = df.inner(self.u, self.v) * df.dx + df.inner(df.Constant(0) * self.p, self.q) * df.dx
        self.M = df.assemble(a_M)

        # set boundary values
        inflow = 'near(x[0], 0)'
        outflow = 'near(x[0], 2.2)'
        walls = 'near(x[1], 0) || near(x[1], 0.41)'
        cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'

        # Prepare Dirichlet boundary conditions
        Uin = '4.0*1.5*sin(pi*t/8)*x[1]*(0.41 - x[1]) / pow(0.41, 2)'
        self.u_in = df.Expression((Uin, '0'), pi=np.pi, t=t0, degree=self.order)

        bc_in = df.DirichletBC(self.W.sub(0), self.u_in, inflow)
        bc_out = df.DirichletBC(self.W.sub(1), 0, outflow)
        bc_walls = df.DirichletBC(self.W.sub(0), (0, 0), walls)
        bc_cylinder = df.DirichletBC(self.W.sub(0), (0, 0), cylinder)
        #
        self.bcu = [bc_cylinder, bc_walls, bc_out, bc_in]
        #

        # Homogen boundary conditions for the residual
        self.bc_hom = df.DirichletBC(self.V, df.Constant((0, 0)), Boundary)

        bc_hom_u = df.DirichletBC(self.W.sub(0), df.Constant((0, 0)), Boundary)
        bc_hom_p = df.DirichletBC(self.W.sub(1), df.Constant(0), Boundary)

        self.bc_hom2 = [bc_hom_u, bc_hom_p]

        # set forcing term as expression
        self.g = df.Expression(('0', '0'), a=np.pi, b=self.nu, t=self.t0, degree=self.order)

        path = 'data/dataLin/data_N4_dt_0.01_LU/'
        self.xdmffile_p = df.XDMFFile(path + 'Cylinder_pressure.xdmf')
        self.xdmffile_u = df.XDMFFile(path + 'Cylinder_velocity.xdmf')

        self.wold = df.Function(self.W)

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

        w = self.dtype_u(u0)
        R = self.dtype_u(u0)

        uold, pold = df.split(u0.values)

        # update time in Boundary conditions
        self.u_in.t = t

        # Get the forcing term
        self.g.t = t
        G0 = self.dtype_f(df.interpolate(self.g, self.V), val=self.V)

        F = df.dot(self.u, self.v) * df.dx
        F += factor * df.dot(df.dot(self.u, df.nabla_grad(uold)), self.v) * df.dx
        F += factor * df.dot(df.dot(uold, df.nabla_grad(self.u)), self.v) * df.dx
        F -= factor * df.dot(df.dot(uold, df.nabla_grad(uold)), self.v) * df.dx
        F += factor * self.nu * df.inner(df.nabla_grad(self.u), df.nabla_grad(self.v)) * df.dx
        F -= factor * df.dot(self.p, df.div(self.v)) * df.dx
        F -= factor * df.dot(G0.values, self.v) * df.dx
        F -= factor * df.dot(df.div(self.u), self.q) * df.dx

        A = df.assemble(df.lhs(F))
        R0 = df.assemble(df.rhs(F))
        R.values.vector()[:] = R0[:]

        b = self.dtype_u(df.project(rhs.values + R.values, self.W), val=self.W)

        [bc.apply(A, b.values.vector()) for bc in self.bcu]

        df.solve(A, w.values.vector(), b.values.vector())

        return w

    def eval_f(self, w, t):
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

        f = self.dtype_f(self.W)

        u, p = df.split(w.values)

        # Get the forcing term
        self.g.t = t
        G0 = self.dtype_f(df.interpolate(self.g, self.V), val=self.V)

        F = -1.0 * df.dot(df.dot(u, df.nabla_grad(u)), self.v) * df.dx
        F -= self.nu * df.inner(df.nabla_grad(u), df.nabla_grad(self.v)) * df.dx
        F += df.dot(p, df.div(self.v)) * df.dx
        F += df.dot(G0.values, self.v) * df.dx
        F += df.dot(df.div(u), self.q) * df.dx

        # f = self.dtype_f(df.assemble(F))

        F0 = df.assemble(F)
        f.values.vector()[:] = F0[:]

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
        me : dtype_u
            The product :math:`M \vec{u}`.
        """

        me = self.dtype_u(self.W)
        self.M.mult(u.values.vector(), me.values.vector())

        return me

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

        me = self.dtype_u(self.V)

        b = self.dtype_u(u)
        M = self.M
        self.bc_hom.apply(M, b.values.vector())

        df.solve(M, me.values.vector(), b.values.vector())
        return me

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

        u0 = df.Function(self.V)
        p0 = df.Function(self.Q)
        up = df.Function(self.W)

        un, pn = up.split()

        u0 = df.Expression(('0.0', '0.0'), degree=self.order)
        p0 = df.Expression('0.0', degree=self.order - 1)

        un.assign(df.interpolate(u0, self.V))
        pn.assign(df.interpolate(p0, self.Q))

        df.assign(up, [un, pn])

        me = self.dtype_u(up, val=self.W)

        return me

    def fix_residual(self, res):
        """
        Applies homogeneous Dirichlet boundary conditions to the residual

        Parameters
        ----------
        res : dtype_u
              Residual
        """
        [bc.apply(res.values.vector()) for bc in self.bc_hom2]

        return None

    def WriteFiles(self, w, t):

        u, p = df.split(w.values)

        self.xdmffile_p.write_checkpoint(df.project(p, self.Q), "pn", t, df.XDMFFile.Encoding.HDF5, True)
        self.xdmffile_u.write_checkpoint(df.project(u, self.V), "un", t, df.XDMFFile.Encoding.HDF5, True)

        return None

    def CloseXDMFfile(self):

        self.xdmffile_p.close()
        self.xdmffile_u.close()

        return None
