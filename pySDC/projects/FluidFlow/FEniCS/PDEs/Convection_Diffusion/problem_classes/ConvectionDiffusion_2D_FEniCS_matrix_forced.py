import logging
import dolfin as df
import numpy as np
import mshr
from pySDC.core.problem import Problem
from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh, rhs_fenics_mesh


# noinspection PyUnusedLocal
class fenics_ConvDiff2D(Problem):
    r"""
    This example demonstrates the implementation of a forced two-dimensional convection-diffusion equation using
    Dirichlet boundary conditions. The problem considered is a benchmark test with a rotating Gaussian profile.

    The equation we are solving is the two-dimensional convection-diffusion equation:

    .. math::
        \frac{d u}{d t} = - U \cdot \nabla u +  \nu \Delta u + f

    where:
        .. math:`u(x, y, t)` is the scalar field we are solving for (e.g., concentration or temperature).
        .. math:`\frac{\partial u}{\partial t}`  is the time derivative of  .. math:`u`.
        .. math:`U = (u, v)` is the velocity field of the flow.
        .. math:`\nabla u` is the gradient of .. math:`u`.
        .. math:`\Delta u` is the Laplacian of .. math:`u`, representing diffusion.
        .. math:`\nu` is the kinematic viscosity, which quantifies the diffusion rate.
        .. math:`f(x, y, t)` is the source term, representing external forcing or generation of .. math:`u`.

    The computational domain for this problem is:

    .. math::
         x \in \Omega := [-0.5, 0.5] \times [-0.5, 0.5]

    Dirichlet boundary conditions are applied, meaning that the value of .. math:`u` is specified on the boundary of the domain.
    In this benchmark example, the forcing term .. math:`f` is:

    .. math::
        f(x,y,t) = 0

    This implies there are no additional sources or sinks affecting the field .. math:`u`, simplifying the problem to just the
    effects of convection and diffusion. The analytical solution for the scalar field .. math:`u is given by:

    .. math::
        u(x,y,t) = \frac{\sigma^2}{\sigma^2 + \omega\nu t}*\exp(- \frac{(\hat{x}-x_0)^2 + (\hat{y}-y_0)^2}{\sigma^2 + \omega\nu t}


    where:
      .. math:`\sigma` is the initial standard deviation of the Gaussian.
      .. math:`\omega` is the angular velocity of the rotation.
      .. math:`t` is the time.
      .. math:`(\hat{x}, \hat{y})` are the rotated coordinates defined by:
      .. math::
          \hat{x} = \cos(\omega t) x + \sin(\omega t) y
          \hat{y} = -\sin(\omega t) x + \cos(\omega t) y
      .. math:`(x_0, y_0)` is the center of the Gaussian, given as .. math:`(-0.25, 0.0)`.


    The velocity field .. math:`U` for the rotating Gaussian is defined as:

    .. math::
        U = (u,v) = (-\omega*y, \omega*x)

    This represents a counter-clockwise rotation around the origin with angular velocity .. math:`\omega`.
    The flow causes the scalar field .. math:`u` to be advected in a circular pattern, simulating the rotation of the Gaussian.


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
        The forcing term :math:`f` in the convection-diffusion equations.
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

    def __init__(self, c_nvars=64, t0=0.0, family='CG', order=2, refinements=1, nu=0.01, c=0.0, sigma=0.05):
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
        domain = mshr.Rectangle(df.Point(-0.5, -0.5), df.Point(0.5, 0.5))
        mesh = mshr.generate_mesh(domain, c_nvars)

        for _ in range(refinements):
            mesh = df.refine(mesh)

        # define function space for future reference
        self.V = df.FunctionSpace(mesh, family, order)
        tmp = df.Function(self.V)
        print('DoFs on this level:', len(tmp.vector()[:]))

        # define velocity
        self.VC = df.VectorFunctionSpace(mesh, family, order)
        expr = df.Expression(('-4*x[1]', '4*x[0]'), degree=order)
        self.U = df.interpolate(expr, self.VC)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(fenics_ConvDiff2D, self).__init__(self.V)
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
        a_M = u * v * df.dx

        self.M = df.assemble(a_M)
        self.K = df.assemble(a_K)

        # set boundary values
        self.bc = df.DirichletBC(self.V, df.Constant(0.0), Boundary)
        self.bc_hom = df.DirichletBC(self.V, df.Constant(0.0), Boundary)

        # set forcing term as expression
        self.g = df.Expression('0.0', t=self.t0, degree=self.order)

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

        a_C = -1.0 * df.inner(df.inner(self.U, df.nabla_grad(self.un.values)), self.v) * df.dx
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
            'pow(s,2)/(pow(s,2)+4*nu*t)*exp(-(pow(((cos(4*t)*x[0]+sin(4*t)*x[1])-x0),2)\
            +pow(((-sin(4*t)*x[0]+cos(4*t)*x[1])-y0),2))/(pow(s,2)+4*nu*t))',
            s=self.sigma,
            nu=self.nu,
            x0=-0.25,
            y0=0.0,
            t=t,
            degree=self.order,
        )

        # u0 = df.Expression('sin(a*x[0]) * sin(a*x[1]) * cos(t) + c', c=self.c, a=np.pi, t=t, degree=self.order)
        me = self.dtype_u(df.interpolate(u0, self.V), val=self.V)

        return me


# noinspection PyUnusedLocal
class fenics_ConvDiff2D_mass(fenics_ConvDiff2D):
    r"""
    This example demonstrates the implementation of a forced two-dimensional convection-diffusion equation using
    Dirichlet boundary conditions. The problem considered is a benchmark test with a rotating Gaussian profile.

    The equation we are solving is the two-dimensional convection-diffusion equation:

    .. math::
        \frac{d u}{d t} = - U \cdot \nabla u +  \nu \Delta u + f

    where:
        .. math:`u(x, y, t)` is the scalar field we are solving for (e.g., concentration or temperature).
        .. math:`\frac{\partial u}{\partial t}`  is the time derivative of  .. math:`u`.
        .. math:`U = (u, v)` is the velocity field of the flow.
        .. math:`\nabla u` is the gradient of .. math:`u`.
        .. math:`\Delta u` is the Laplacian of .. math:`u`, representing diffusion.
        .. math:`\nu` is the kinematic viscosity, which quantifies the diffusion rate.
        .. math:`f(x, y, t)` is the source term, representing external forcing or generation of .. math:`u`.

    The computational domain for this problem is:

    .. math::
         x \in \Omega := [-0.5, 0.5] \times [-0.5, 0.5]

    Dirichlet boundary conditions are applied, meaning that the value of .. math:`u` is specified on the boundary of the domain.
    In this benchmark example, the forcing term .. math:`f` is:

    .. math::
        f(x,y,t) = 0

    This implies there are no additional sources or sinks affecting the field .. math:`u`, simplifying the problem to just the
    effects of convection and diffusion. The analytical solution for the scalar field .. math:`u is given by:

    .. math::
        u(x,y,t) = \frac{\sigma^2}{\sigma^2 + \omega\nu t}*\exp(- \frac{(\hat{x}-x_0)^2 + (\hat{y}-y_0)^2}{\sigma^2 + \omega\nu t}


    where:
      .. math:`\sigma` is the initial standard deviation of the Gaussian.
      .. math:`\omega` is the angular velocity of the rotation.
      .. math:`t` is the time.
      .. math:`(\hat{x}, \hat{y})` are the rotated coordinates defined by:
      .. math::
          \hat{x} = \cos(\omega t) x + \sin(\omega t) y
          \hat{y} = -\sin(\omega t) x + \cos(\omega t) y
      .. math:`(x_0, y_0)` is the center of the Gaussian, given as .. math:`(-0.25, 0.0)`.


    The velocity field .. math:`U` for the rotating Gaussian is defined as:

    .. math::
        U = (u,v) = (-\omega*y, \omega*x)

    This represents a counter-clockwise rotation around the origin with angular velocity .. math:`\omega`.
    The flow causes the scalar field .. math:`u` to be advected in a circular pattern, simulating the rotation of the Gaussian.


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
        The forcing term :math:`f` in the convection-diffusion equations.
    bc : DirichletBC
        Denotes the Dirichlet boundary conditions.

    References
    ----------
    .. [1] The FEniCS Project Version 1.5. M. S. Alnaes, J. Blechta, J. Hake, A. Johansson, B. Kehlet, A. Logg,
        C. Richardson, J. Ring, M. E. Rognes, G. N. Wells. Archive of Numerical Software (2015).
    .. [2] Automated Solution of Differential Equations by the Finite Element Method. A. Logg, K.-A. Mardal, G. N.
        Wells and others. Springer (2012).
    """

    def __init__(self, c_nvars=64, t0=0.0, family='CG', order=2, refinements=1, nu=0.1, c=0.0, sigma=0.05):
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

        a_C = -1.0 * df.inner(df.dot(self.U, df.nabla_grad(self.un.values)), self.v) * df.dx
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
class fenics_ConvDiff2D_mass_timebc(fenics_ConvDiff2D_mass):
    r"""
    This example demonstrates the implementation of a forced two-dimensional convection-diffusion equation using
    Dirichlet boundary conditions. The problem considered is a benchmark test with a rotating Gaussian profile.

    The equation we are solving is the two-dimensional convection-diffusion equation:

    .. math::
        \frac{d u}{d t} = - U \cdot \nabla u +  \nu \Delta u + f

    where:
        .. math:`u(x, y, t)` is the scalar field we are solving for (e.g., concentration or temperature).
        .. math:`\frac{\partial u}{\partial t}`  is the time derivative of  .. math:`u`.
        .. math:`U = (u, v)` is the velocity field of the flow.
        .. math:`\nabla u` is the gradient of .. math:`u`.
        .. math:`\Delta u` is the Laplacian of .. math:`u`, representing diffusion.
        .. math:`\nu` is the kinematic viscosity, which quantifies the diffusion rate.
        .. math:`f(x, y, t)` is the source term, representing external forcing or generation of .. math:`u`.

    The computational domain for this problem is:

    .. math::
         x \in \Omega := [-0.5, 0.5] \times [-0.5, 0.5]

    Dirichlet boundary conditions are applied, meaning that the value of .. math:`u` is specified on the boundary of the domain.
    In this benchmark example, the forcing term .. math:`f` is:

    .. math::
        f(x,y,t) = 0

    This implies there are no additional sources or sinks affecting the field .. math:`u`, simplifying the problem to just the
    effects of convection and diffusion. The analytical solution for the scalar field .. math:`u is given by:

    .. math::
        u(x,y,t) = \frac{\sigma^2}{\sigma^2 + \omega\nu t}*\exp(- \frac{(\hat{x}-x_0)^2 + (\hat{y}-y_0)^2}{\sigma^2 + \omega\nu t}


    where:
      .. math:`\sigma` is the initial standard deviation of the Gaussian.
      .. math:`\omega` is the angular velocity of the rotation.
      .. math:`t` is the time.
      .. math:`(\hat{x}, \hat{y})` are the rotated coordinates defined by:
      .. math::
          \hat{x} = \cos(\omega t) x + \sin(\omega t) y
          \hat{y} = -\sin(\omega t) x + \cos(\omega t) y
      .. math:`(x_0, y_0)` is the center of the Gaussian, given as .. math:`(-0.25, 0.0)`.


    The velocity field .. math:`U` for the rotating Gaussian is defined as:

    .. math::
        U = (u,v) = (-\omega*y, \omega*x)

    This represents a counter-clockwise rotation around the origin with angular velocity .. math:`\omega`.
    The flow causes the scalar field .. math:`u` to be advected in a circular pattern, simulating the rotation of the Gaussian.


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
        The forcing term :math:`f` in the convection-diffusion equations.
    bc : DirichletBC
        Denotes the Dirichlet boundary conditions.

    References
    ----------
    .. [1] The FEniCS Project Version 1.5. M. S. Alnaes, J. Blechta, J. Hake, A. Johansson, B. Kehlet, A. Logg,
        C. Richardson, J. Ring, M. E. Rognes, G. N. Wells. Archive of Numerical Software (2015).
    .. [2] Automated Solution of Differential Equations by the Finite Element Method. A. Logg, K.-A. Mardal, G. N.
        Wells and others. Springer (2012).
    """

    def __init__(self, c_nvars=64, t0=0.0, family='CG', order=2, refinements=1, nu=0.1, c=0.0, sigma=0.05):
        """Initialization routine"""

        # define the Dirichlet boundary
        def Boundary(x, on_boundary):
            return on_boundary

        super().__init__(c_nvars, t0, family, order, refinements, nu, c)

        self.u_D = df.Expression(
            'pow(s,2)/(pow(s,2)+4*nu*t)*exp(-(pow(((cos(4*t)*x[0]+sin(4*t)*x[1])-x0),2)\
            +pow(((-sin(4*t)*x[0]+cos(4*t)*x[1])-y0),2))/(pow(s,2)+4*nu*t))',
            s=self.sigma,
            nu=self.nu,
            x0=-0.25,
            y0=0.0,
            t=t0,
            degree=self.order,
        )

        self.bc = df.DirichletBC(self.V, self.u_D, Boundary)
        self.bc_hom = df.DirichletBC(self.V, df.Constant(0), Boundary)

        # set forcing term as expression
        self.g = df.Expression('0', t=self.t0, degree=self.order)

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
        self.bc.apply(b.values.vector())

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
            'pow(s,2)/(pow(s,2)+4*nu*t)*exp(-(pow(((cos(4*t)*x[0]+sin(4*t)*x[1])-x0),2)\
            +pow(((-sin(4*t)*x[0]+cos(4*t)*x[1])-y0),2))/(pow(s,2)+4*nu*t))',
            s=self.sigma,
            nu=self.nu,
            x0=-0.25,
            y0=0.0,
            t=t,
            degree=self.order,
        )

        me = self.dtype_u(df.interpolate(u0, self.V), val=self.V)

        return me
