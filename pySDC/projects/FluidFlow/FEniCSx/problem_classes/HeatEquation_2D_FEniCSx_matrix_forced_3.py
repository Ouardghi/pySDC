import logging

from mpi4py import MPI
from petsc4py import PETSc
import dolfinx as dfx
import dolfinx.fem.petsc 
import ufl
import numpy as np

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

    In this class the problem is implemented in the way that the spatial part is solved using ``FEniCS`` [1]_. Hence, the problem
    is reformulated to the *weak formulation*

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
    dtype_f = imex_mesh

    def __init__(self, nelems=128, t0=0.0, family='CG', order=4, refinements=1, nu=0.1, c=0.0, comm=MPI.COMM_SELF):
        """Initialization routine"""

        # Define mesh
        #domain = dfx.mesh.create_interval(comm, nx=nelems, points=np.array([0, 1]))
        
        domain = dfx.mesh.create_rectangle( 
                            comm, 
                            points=((0.0, 0.0), (1.0, 1.0)), 
                            n=(nelems, nelems), 
                            cell_type=dfx.mesh.CellType.triangle,
                                )

        self.V = dfx.fem.functionspace(domain, (family, order))
        self.x = ufl.SpatialCoordinate(domain)
        
        tmp = dfx.fem.Function(self.V)
        nx = len(tmp.x.array)
        print('DoFs on this level:', nx)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(init=(nx, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'nelems', 't0', 'family', 'order', 'refinements', 'nu', 'c', localVars=locals(), readOnly=True
        )

        # Create boundary condition
        fdim = domain.topology.dim - 1
        boundary_facets = dfx.mesh.locate_entities_boundary( domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
        self.bc     = dfx.fem.dirichletbc(PETSc.ScalarType(self.c), dfx.fem.locate_dofs_topological(self.V, fdim, boundary_facets), self.V)
        self.bc_hom = dfx.fem.dirichletbc(PETSc.ScalarType(0),      dfx.fem.locate_dofs_topological(self.V, fdim, boundary_facets), self.V)
        self.fix_bc_for_residual = True

        # Stiffness term (Laplace) and mass term
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        a_K = -1.0 * ufl.dot(ufl.grad(u), self.nu * ufl.grad(v)) * ufl.dx
        a_M = u * v * ufl.dx

        self.K = dolfinx.fem.petsc.assemble_matrix(dfx.fem.form(a_K), bcs=[self.bc])
        self.K.assemble()

        self.M = dolfinx.fem.petsc.assemble_matrix(dfx.fem.form(a_M), bcs=[self.bc])
        self.M.assemble()

        # set forcing term
        self.g = dfx.fem.Function(self.V)
        t = self.t0
        #self.g.interpolate(
        #    lambda x: -np.sin(np.pi * x[0]) * (np.sin(t) -  self.nu * np.pi * np.pi * np.cos(t)))

        self.tmp_u = dfx.fem.Function(self.V)
        self.tmp_f = dfx.fem.Function(self.V)

        self.solver = PETSc.KSP().create(domain.comm)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)

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

        self.convert_to_fenicsx_vector(input=u0, output=self.tmp_u)
        self.convert_to_fenicsx_vector(input=rhs, output=self.tmp_f)
        # T = self.M - factor * self.K
        # dofs, _ = self.bc.dof_indices()
        # T.zeroRowsLocal(dofs, diag=1)
        dolfinx.fem.petsc.set_bc(self.tmp_f.vector, [self.bc])
        self.solver.setOperators(self.M - factor * self.K)
        self.solver.solve(self.tmp_f.vector, self.tmp_u.vector)
        # dolfinx.fem.petsc.set_bc(self.tmp_u.vector, [self.bc])

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

        f = self.dtype_f(self.init)

        self.convert_to_fenicsx_vector(input=u, output=self.tmp_u)
        self.K.mult(self.tmp_u.vector, self.tmp_f.vector)
        self.convert_from_fenicsx_vector(input=self.tmp_f, output=f.impl)

        self.g.interpolate(
            lambda x: -np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * (np.sin(t) - 2 * self.nu * np.pi * np.pi * np.cos(t)))
        self.M.mult(self.g.vector, self.tmp_f.vector)
        self.convert_from_fenicsx_vector(input=self.tmp_f, output=f.expl)

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
        self.M.mult(self.tmp_u.vector, self.tmp_f.vector)
        uM = self.dtype_u(self.init)
        self.convert_from_fenicsx_vector(input=self.tmp_f, output=uM)
        return uM
    
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
        dolfinx.fem.petsc.set_bc(self.tmp_u.vector, [self.bc_hom])
        self.tmp_u.x.scatter_forward()
        self.convert_from_fenicsx_vector(input=self.tmp_u, output=res)
        return None

# noinspection PyUnusedLocal
class fenicsx_heat(fenicsx_heat_mass):
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

    In this class the problem is implemented in the way that the spatial part is solved using ``FEniCS`` [1]_. Hence, the problem
    is reformulated to the *weak formulation*

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
    def __init__(self, nelems=128, t0=0.0, family='CG', order=4, refinements=1, nu=0.1, c=0.0, comm=MPI.COMM_SELF):
        """Initialization routine"""

        super().__init__(nelems, t0, family, order, refinements, nu, c, comm)

        self.fix_bc_for_residual = True


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
        Mrhs = self.apply_mass_matrix(rhs)     
        self.convert_to_fenicsx_vector(input=Mrhs, output=self.tmp_f)
        
        dolfinx.fem.petsc.set_bc(self.tmp_f.vector, [self.bc])
        self.solver.setOperators(self.M - factor * self.K)
        self.solver.solve(self.tmp_f.vector, self.tmp_u.vector)


        u = self.dtype_u(self.init)
        self.convert_from_fenicsx_vector(input=self.tmp_u, output=u)
        
        #print('called')

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

        f = self.dtype_f(self.init)

        self.convert_to_fenicsx_vector(input=u, output=self.tmp_u)
        self.K.mult(self.tmp_u.vector, self.tmp_f.vector)
        
        self.tmp_f=self.__invert_mass_matrix(self.tmp_f)
        
        self.convert_from_fenicsx_vector(input=self.tmp_f, output=f.impl)

        self.g.interpolate(
            lambda x: -np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * (np.sin(t) - 2 * self.nu * np.pi * np.pi * np.cos(t)))
        #self.M.mult(self.g.vector, self.tmp_f.vector)
        self.convert_from_fenicsx_vector(input=self.g, output=f.expl)

        return f
    
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
        tmp_f = dfx.fem.Function(self.V)

        self.solver.setOperators(self.M)
        self.solver.solve(u.vector, tmp_f.vector)
        
        return tmp_f