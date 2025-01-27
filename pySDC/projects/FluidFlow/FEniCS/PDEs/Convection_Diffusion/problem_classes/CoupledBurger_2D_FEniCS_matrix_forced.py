import logging
import dolfin as df
import numpy as np
import mshr
from pySDC.core.problem import Problem
from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh, rhs_fenics_mesh


# noinspection PyUnusedLocal
class fenics_Burger2D(Problem):
    r"""
    This example demonstrates the implementation of a forced two-dimensional convection-diffusion equation using
    Dirichlet boundary conditions. The problem considered is the coupled viscous Burger's equations.

    The equations we are solving are the two-dimensional nonlinear convection-diffusion equation:

    .. math::
        \frac{d u}{d t} = - u \cdot \nabla u +  \frac{1}{Re} \Delta u + f

    where:
        .. math:`u(x, y, t)` is the vector field we are solving for (e.g., concentration or temperature).
        .. math:`\frac{\partial u}{\partial t}`  is the time derivative of  .. math:`u`.
        .. math:`\nabla u` is the gradient of .. math:`u`.
        .. math:`\Delta u` is the Laplacian of .. math:`u`, representing diffusion.
        .. math:`Re` is the Reinlods number (scalar)
        .. math:`f(x, y, t)` is the source term, representing external forcing or generation of .. math:`u`.

    The computational domain for this problem is:

    .. math::
         x \in \Omega := [0, 1] \times [0, 1]

    Dirichlet boundary conditions are applied, meaning that the value of .. math:`u` is specified on the boundary of the domain.
    In this benchmark example, the forcing term .. math:`f` is:

    .. math::
        f(x,y,t) = 0

    This implies there are no additional sources or sinks affecting the field .. math:`u`, simplifying the problem to just the
    effects of convection and diffusion. The analytical solution for the vector field .. math:`u is given by:

    .. math::
        u_1(,y,t) = \frac{3}{4}-\frac{1}{g(x,y,t)}
        u_2(,y,t) = \frac{3}{4}+\frac{1}{g(x,y,t)}

    where
    .. math::
        g(x,y,t) = 4*(1+\exp(- \frac{(4*x - 4*y + t)*Re}{32}))

    This solution describes the velocity components .. math:`u_1` and .. math:`u_2` in the domain over time, showing how
    the initial conditions evolve due to convection and diffusion.

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

    dtype_u = fenics_mesh
    dtype_f = rhs_fenics_mesh

    def __init__(self, c_nvars=64, t0=0.0, family='CG', order=2, refinements=1, nu=0.002, c=0.0, sigma=0.05):
        """Initialization routine"""

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
        domain = mshr.Rectangle(df.Point(0.0, 0.0), df.Point(1.0, 1.0))
        mesh = mshr.generate_mesh(domain, c_nvars)

        # for _ in range(refinements):
        #    mesh = df.refine(mesh)

        # define function space for future reference
        self.V = df.VectorFunctionSpace(mesh, family, order)
        tmp = df.Function(self.V)
        print('DoFs on this level:', len(tmp.vector()[:]))

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(fenics_Burger2D, self).__init__(self.V)
        self._makeAttributeAndRegister(
            'c_nvars', 't0', 'family', 'order', 'refinements', 'nu', 'c', 'sigma', localVars=locals(), readOnly=True
        )

        # Stiffness term (Laplace)
        u = df.TrialFunction(self.V)
        v = df.TestFunction(self.V)

        self.u = df.TrialFunction(self.V)
        self.v = df.TestFunction(self.V)

        a_K = -1.0 * df.inner(df.nabla_grad(u), self.nu * df.nabla_grad(v)) * df.dx

        # Mass term
        a_M = df.inner(u, v) * df.dx

        self.M = df.assemble(a_M)
        self.K = df.assemble(a_K)

        # set boundary values
        self.bc = df.DirichletBC(self.V, df.Constant((0.0, 0.0)), Boundary)
        self.bc_hom = df.DirichletBC(self.V, df.Constant((0.0, 0.0)), Boundary)

        # set forcing term as expression
        self.g = df.Expression(('0.0', '0.0'), t=self.t0, degree=self.order)

        self.un = self.u_exact(0.0)

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

        b = self.apply_mass_matrix(rhs)

        u = self.dtype_u(u0)
        T = self.M - factor * self.K
        self.bc.apply(T, b.values.vector())
        df.solve(T, u.values.vector(), b.values.vector())
        self.un = u
        return u

    def __eval_fexpl(self, u, t):
        """
        Helper routine to evaluate the explicit part of the right-hand side.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution (not used here).
        t : float
            Current time at which the numerical solution is computed.

        Returns
        -------
        fexpl : dtype_u
            Explicit part of the right-hand side.
        """

        self.g.t = t
        fexpl1 = self.dtype_u(df.interpolate(self.g, self.V))

        a_C = -1.0 * df.inner(df.dot(self.un.values, df.nabla_grad(self.un.values)), self.v) * df.dx
        self.C = df.assemble(a_C)

        tmp = self.dtype_u(self.V)
        tmp.values.vector()[:] = self.C[:]
        fexpl2 = self.__invert_mass_matrix(tmp)

        fexpl = fexpl1 + fexpl2

        return fexpl

    def __eval_fimpl(self, u, t):
        """
        Helper routine to evaluate the implicit part of the right-hand side.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time at which the numerical solution is computed.

        Returns
        -------
        fimpl : dtype_u
            Explicit part of the right-hand side.
        """

        tmp = self.dtype_u(self.V)
        self.K.mult(u.values.vector(), tmp.values.vector())
        fimpl = self.__invert_mass_matrix(tmp)

        return fimpl

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

        f = self.dtype_f(self.V)
        f.impl = self.__eval_fimpl(u, t)
        f.expl = self.__eval_fexpl(u, t)
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

        me = self.dtype_u(self.V)
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

        u0 = df.Expression(
            (
                '0.75-1.0/(4.0*(1+exp(-((4*x[0]-4*x[1]+t)*Re)/32)))',
                '0.75+1.0/(4.0*(1+exp(-((4*x[0]-4*x[1]+t)*Re)/32)))',
            ),
            Re=500,
            t=t,
            degree=self.order,
        )

        # u0 = df.Expression('sin(a*x[0]) * sin(a*x[1]) * cos(t) + c', c=self.c, a=np.pi, t=t, degree=self.order)
        me = self.dtype_u(df.interpolate(u0, self.V), val=self.V)

        return me


# noinspection PyUnusedLocal
class fenics_Burger2D_mass(fenics_Burger2D):
    r"""
    This example demonstrates the implementation of a forced two-dimensional convection-diffusion equation using
    Dirichlet boundary conditions. The problem considered is the coupled viscous Burger's equations.

    The equations we are solving are the two-dimensional nonlinear convection-diffusion equation:

    .. math::
        \frac{d u}{d t} = - u \cdot \nabla u +  \frac{1}{Re} \Delta u + f

    where:
        .. math:`u(x, y, t)` is the vector field we are solving for (e.g., concentration or temperature).
        .. math:`\frac{\partial u}{\partial t}`  is the time derivative of  .. math:`u`.
        .. math:`\nabla u` is the gradient of .. math:`u`.
        .. math:`\Delta u` is the Laplacian of .. math:`u`, representing diffusion.
        .. math:`Re` is the Reinlods number (scalar)
        .. math:`f(x, y, t)` is the source term, representing external forcing or generation of .. math:`u`.

    The computational domain for this problem is:

    .. math::
         x \in \Omega := [0, 1] \times [0, 1]

    Dirichlet boundary conditions are applied, meaning that the value of .. math:`u` is specified on the boundary of the domain.
    In this benchmark example, the forcing term .. math:`f` is:

    .. math::
        f(x,y,t) = 0

    This implies there are no additional sources or sinks affecting the field .. math:`u`, simplifying the problem to just the
    effects of convection and diffusion. The analytical solution for the vector field .. math:`u is given by:

    .. math::
        u_1(,y,t) = \frac{3}{4}-\frac{1}{g(x,y,t)}
        u_2(,y,t) = \frac{3}{4}+\frac{1}{g(x,y,t)}

    where
    .. math::
        g(x,y,t) = 4*(1+\exp(- \frac{(4*x - 4*y + t)*Re}{32}))

    This solution describes the velocity components .. math:`u_1` and .. math:`u_2` in the domain over time, showing how
    the initial conditions evolve due to convection and diffusion.

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
        Denotes the Dirichlet boundary conditions.
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

    def __init__(self, c_nvars=64, t0=0.0, family='CG', order=2, refinements=1, nu=0.002, c=0.0, sigma=0.05):
        """Initialization routine"""

        super().__init__(c_nvars, t0, family, order, refinements, nu, c)

        self.fix_bc_for_residual = True

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Dolfin's linear solver for :math:`(M - factor A) \vec{u} = \vec{rhs}`.

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

        u = self.dtype_u(u0)
        T = self.M - factor * self.K
        b = self.dtype_u(rhs)

        self.bc.apply(T, b.values.vector())

        df.solve(T, u.values.vector(), b.values.vector())

        self.un = u
        return u

    def eval_f(self, u, t):
        """
        Routine to evaluate both parts of the right-hand side.

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

        f = self.dtype_f(self.V)

        self.K.mult(u.values.vector(), f.impl.values.vector())

        self.g.t = t
        fexpl1 = self.dtype_u(df.interpolate(self.g, self.V))
        fexpl1 = self.apply_mass_matrix(fexpl1)

        a_C = -1.0 * df.inner(df.dot(self.un.values, df.nabla_grad(self.un.values)), self.v) * df.dx
        self.C = df.assemble(a_C)

        fexpl2 = self.dtype_u(self.V)
        fexpl2.values.vector()[:] = self.C[:]

        f.expl = fexpl1 + fexpl2

        return f

    def fix_residual(self, res):
        """
        Applies homogeneous Dirichlet boundary conditions to the residual

        Parameters
        ----------
        res : dtype_u
              Residual
        """
        self.bc_hom.apply(res.values.vector())
        return None


# noinspection PyUnusedLocal
class fenics_Burger2D_mass_timebc(fenics_Burger2D_mass):
    r"""
    This example demonstrates the implementation of a forced two-dimensional convection-diffusion equation using
    Dirichlet boundary conditions. The problem considered is the coupled viscous Burger's equations.

    The equations we are solving are the two-dimensional nonlinear convection-diffusion equation:

    .. math::
        \frac{d u}{d t} = - u \cdot \nabla u +  \frac{1}{Re} \Delta u + f

    where:
        .. math:`u(x, y, t)` is the vector field we are solving for (e.g., concentration or temperature).
        .. math:`\frac{\partial u}{\partial t}`  is the time derivative of  .. math:`u`.
        .. math:`\nabla u` is the gradient of .. math:`u`.
        .. math:`\Delta u` is the Laplacian of .. math:`u`, representing diffusion.
        .. math:`Re` is the Reinlods number (scalar)
        .. math:`f(x, y, t)` is the source term, representing external forcing or generation of .. math:`u`.

    The computational domain for this problem is:

    .. math::
         x \in \Omega := [0, 1] \times [0, 1]

    Dirichlet boundary conditions are applied, meaning that the value of .. math:`u` is specified on the boundary of the domain.
    In this benchmark example, the forcing term .. math:`f` is:

    .. math::
        f(x,y,t) = 0

    This implies there are no additional sources or sinks affecting the field .. math:`u`, simplifying the problem to just the
    effects of convection and diffusion. The analytical solution for the vector field .. math:`u is given by:

    .. math::
        u_1(,y,t) = \frac{3}{4}-\frac{1}{g(x,y,t)}
        u_2(,y,t) = \frac{3}{4}+\frac{1}{g(x,y,t)}

    where
    .. math::
        g(x,y,t) = 4*(1+\exp(- \frac{(4*x - 4*y + t)*Re}{32}))

    This solution describes the velocity components .. math:`u_1` and .. math:`u_2` in the domain over time, showing how
    the initial conditions evolve due to convection and diffusion.

    In this class the problem is implemented in the way that the spatial part is solved using ``FEniCS`` [1]_. Hence, the problem
    is reformulated to the *weak formulation*

    .. math:
        \int_\Omega u_t v\,dx = - \int_\Omega u \cdot\nabla u v\,dx - \frac{1}{Re}\int_\Omega \nabla u \nabla v\,dx + \int_\Omega f v\,dx.

    The forcing term  and the convectif part are treated explicitly, and are expressed via the mass matrix resulting from the left-hand
    side term :math:`\int_\Omega u_t v\,dx`, and the other part will be treated in an implicit way.

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

    def __init__(self, c_nvars=64, t0=0.0, family='CG', order=2, refinements=1, nu=0.002, c=0.0, sigma=0.05):
        """Initialization routine"""

        # define the Dirichlet boundary
        def Boundary(x, on_boundary):
            return on_boundary

        super().__init__(c_nvars, t0, family, order, refinements, nu, c)

        self.u_D = df.Expression(
            (
                '0.75-1.0/(4.0*(1+exp(-((4*x[0]-4*x[1]+t)*Re)/32)))',
                '0.75+1.0/(4.0*(1+exp(-((4*x[0]-4*x[1]+t)*Re)/32)))',
            ),
            Re=500,
            t=t0,
            degree=self.order,
        )

        self.bc = df.DirichletBC(self.V, self.u_D, Boundary)
        self.bc_hom = df.DirichletBC(self.V, df.Constant((0.0, 0.0)), Boundary)

        # set forcing term as expression
        self.g = df.Expression(('0', '0'), t=self.t0, degree=self.order)

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Dolfin's linear solver for :math:`(M - factor A) \vec{u} = \vec{rhs}`.

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

        u = self.dtype_u(u0)
        T = self.M - factor * self.K
        b = self.dtype_u(rhs)

        self.u_D.t = t

        self.bc.apply(T, b.values.vector())
        # self.bc.apply(b.values.vector())

        df.solve(T, u.values.vector(), b.values.vector())

        self.un = u
        return u

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

        u0 = df.Expression(
            (
                '0.75-1.0/(4.0*(1+exp(-((4*x[0]-4*x[1]+t)*Re)/32)))',
                '0.75+1.0/(4.0*(1+exp(-((4*x[0]-4*x[1]+t)*Re)/32)))',
            ),
            Re=500,
            t=t,
            degree=self.order,
        )

        me = self.dtype_u(df.interpolate(u0, self.V), val=self.V)

        return me
