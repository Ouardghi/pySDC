import logging
from math import degrees

import dolfin as df
import numpy as np

from pySDC.core.problem import Problem
from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh, rhs_fenics_mesh


# noinspection PyUnusedLocal
class fenics_heat(Problem):
    r"""
    Example implementing the forced two-dimensional Coupled viscous Burger equations with Dirichlet boundary conditions

    .. math::
        \frac{\partial u}{\partial t} = - u \cdot \nabla u  +  \nu \nabla^2 u  +  f

    for :math:`x \in \Omega:=[0,1]\times[0,1]`, where the forcing term :math:`f` is defined by

    .. math::
        f(x, t) = (0, 0)
    and

    .. math::
        u(x, 0) = (1/4 - 1/(4*exp(-(4x-4y)/(32 \nu))), 1/4 + 1/(4*exp(-(4x-4y)/(32 \nu))))

    the exact solution of the problem is given by

    .. math::
        u(x, t) = (1/4 - 1/(4*exp(-(4x-4y + t)/(32 \nu))), 1/4 + 1/(4*exp(-(4x-4y + t)/(32 \nu))))

    In this class the problem is implemented in the way that the spatial part is solved using ``FEniCS`` [1]_. Hence, the problem
    is reformulated to the *weak formulation*

    .. math:
        \int_\Omega u_t v\,dx = -\int_\Omega u \cdot \nabla u v\,dx
                                - \nu \int_\Omega \nabla u \nabla v\,dx + \int_\Omega f v\,dx.

    The convective part is non-linear, the newton solver is used to solve this equation.

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
    dtype_f = fenics_mesh
    #dtype_f = rhs_fenics_mesh
    df.set_log_active(False)
    def __init__(self, c_nvars=128, t0=0.0, family='CG', order=4, refinements=1, nu=0.1, c=0.0):
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
        mesh = df.UnitSquareMesh(c_nvars,c_nvars)

        #for _ in range(refinements):
        #    mesh = df.refine(mesh)

        # define function space for future reference
        self.V = df.VectorFunctionSpace(mesh, family, order)
        tmp = df.Function(self.V)
        print('DoFs on this level:', len(tmp.vector()[:]))

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(fenics_heat, self).__init__(self.V)
        self._makeAttributeAndRegister(
            'c_nvars', 't0', 'family', 'order', 'refinements', 'nu', 'c', localVars=locals(), readOnly=True
        )

        # Stiffness term (Laplace)
        u = df.TrialFunction(self.V)
        v = df.TestFunction(self.V)
        
        self.u = u
        self.v = v

        a_K = -1.0 * df.inner(df.nabla_grad(u), self.nu * df.nabla_grad(v)) * df.dx

        # Mass term
        a_M = df.inner(u , v) * df.dx

        self.M = df.assemble(a_M)
        self.K = df.assemble(a_K)

        # set boundary values
        ue = '0.75-1.0/(4.0*(1+exp(-(4*x[0]-4*x[1]+t)/(32*nu))))'
        ve = '0.75+1.0/(4.0*(1+exp(-(4*x[0]-4*x[1]+t)/(32*nu))))'
        self.uex = df.Expression((ue, ve), nu = self.nu, t=self.t0, degree = self.order)

        dudt = '-1/(512*nu*pow(cosh((t + 4*x[0] - 4*x[1])/(64*nu)),2))'
        dvdt = ' 1/(512*nu*pow(cosh((t + 4*x[0] - 4*x[1])/(64*nu)),2))'
        self.dUdt = df.Expression((dudt, dvdt), nu = self.nu, t=self.t0, degree = self.order)

        self.bc = df.DirichletBC(self.V, self.dUdt, Boundary)
        self.bcu = df.DirichletBC(self.V, self.uex, Boundary)
        self.bc_hom = df.DirichletBC(self.V, df.Constant((0,0)), Boundary)

        # set forcing term as expression
        
        self.g = df.Expression(
            ('0','0'),
            a=np.pi,
            b=self.nu,
            t=self.t0,
            degree=self.order,
        )

        self.uold = df.Function(self.V)
    
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

        u = self.dtype_u(u0)
        uapp = rhs.values 
        
        self.dUdt.t=t
        self.g.t=t
        G0 = self.dtype_f(df.interpolate(self.g, self.V), val=self.V) 

        
        F  = df.inner(u.values , self.v )* df.dx  
        F += factor**2*df.inner(df.dot(u.values, df.nabla_grad(u.values)), self.v) * df.dx 
        F += factor*df.inner(df.dot(u.values, df.nabla_grad(uapp)), self.v) * df.dx 
        F += factor*df.inner(df.dot(uapp, df.nabla_grad(u.values)), self.v) * df.dx 
        F += df.inner(df.dot(uapp, df.nabla_grad(uapp)), self.v) * df.dx 
        F += factor*df.inner(df.nabla_grad(u.values), self.nu * df.nabla_grad(self.v)) * df.dx
        F += df.inner(df.nabla_grad(uapp), self.nu * df.nabla_grad(self.v)) * df.dx
        F += df.inner(G0.values, self.v) * df.dx    

        df.solve(F==0,u.values,self.bc, solver_parameters={"newton_solver":{"absolute_tolerance": 1e-15 }})

                                           
        """
        
        F  = df.inner(u.values, self.v )* df.dx 
        F += factor*df.inner(df.dot(u.values, df.nabla_grad(self.uold)), self.v) * df.dx 
        F += df.inner(df.dot(uapp, df.nabla_grad(self.uold)), self.v) * df.dx
        F += factor*df.inner(df.dot(self.uold, df.nabla_grad(u.values)), self.v) * df.dx 
        F += df.inner(df.dot(self.uold, df.nabla_grad(uapp)), self.v) * df.dx
        F += -1.0*df.inner(df.dot(self.uold, df.nabla_grad(self.uold)), self.v) * df.dx
        F += self.nu*factor*df.inner(df.nabla_grad(u.values), df.nabla_grad(self.v)) * df.dx
        F += self.nu*df.inner(df.nabla_grad(uapp), df.nabla_grad(self.v)) * df.dx
        F += df.inner(G0.values, self.v) * df.dx                                       
    
        df.solve(F==0,u.values,self.bc)
        """
        return u

    def eval_f(self, u, du, t):
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
        self.g.t=t
        f = self.dtype_f(self.V)
        
        # Diffusive part      
        diff = -df.inner(self.nu*df.nabla_grad(u.values), df.nabla_grad(self.v))*df.dx
        
        # Convective part  
        conv = df.inner(df.dot(u.values, df.nabla_grad(u.values)), self.v)*df.dx
         
        # External forces 
        frc = df.inner(df.interpolate(self.g, self.V),self.v)*df.dx

        # Time derivative 
        dudt = df.inner(du.values, self.v)*df.dx
        
        g = dudt - diff + conv  - frc

        f.values.vector()[:] = df.assemble(g)[:]
        #f =  self.__invert_mass_matrix(f)
        

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

        #u0 = df.Expression('sin(a*x[0]) * sin(a*x[1]) * cos(t) + c', c=self.c, a=np.pi, t=t, degree=self.order)
        
        u = '0.75-1.0/(4.0*(1+exp(-(4*x[0]-4*x[1]+t)/(32*nu))))'
        v = '0.75+1.0/(4.0*(1+exp(-(4*x[0]-4*x[1]+t)/(32*nu))))'

        u0 = df.Expression((u,v), nu=self.nu, t=t, degree=self.order)    
        me = self.dtype_u(df.interpolate(u0, self.V), val=self.V)

        return me


    def apply_bc(self, u, t):
        """
        Applies boundary conditions to the solution

        Parameters
        ----------
        u   : dtype_u
              Current values of the numerical solution.
        t : float
            Current time at which the numerical solution is computed.  
        """
        self.uex.t=t
        self.bcu.apply(u.values.vector()) 
        return None
    

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



    def OldSolution(self, uold):
        
        self.wold = uold.values.copy()
         
        return None





