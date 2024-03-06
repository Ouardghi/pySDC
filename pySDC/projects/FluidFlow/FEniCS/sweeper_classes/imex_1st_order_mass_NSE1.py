import numpy as np

from pySDC.core.Sweeper import sweeper
from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass

import dolfin as df

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


class imex_1st_order_mass_NSE(imex_1st_order_mass):
    """
    Custom sweeper class, implements Sweeper.py

    First-order IMEX sweeper using implicit/explicit Euler as base integrator

    Attributes:
        QI: implicit Euler integration matrix
        QE: explicit Euler integration matrix
    """
    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes

        Returns:
            None
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked

        # get number of collocation nodes for easier access
        M = self.coll.num_nodes

        # gather all terms which are known already (e.g. from the previous iteration)
        # this corresponds to u0 + QF(u^k) - QIFI(u^k) - QEFE(u^k) + tau

        # get QF(u^k)
        integral = self.integrate()

        # This is somewhat ugly, but we have to apply the mass matrix on u0 only on the finest level
        if L.level_index == 0:
            u0 = P.apply_mass_matrix(L.u[0])
        else:
            u0 = L.u[0]

        for m in range(M):
            # subtract QIFI(u^k)_m + QEFE(u^k)_m
            for j in range(M + 1):
                integral[m] -= L.dt * (self.QI[m + 1, j] * L.f[j].impl + self.QE[m + 1, j] * L.f[j].expl)
            # add initial value
            integral[m] += u0
            # add tau if associated
            if L.tau[m] is not None:
                integral[m] += L.tau[m]
        
        # do the sweep
        for m in range(0, M):
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            rhs = P.dtype_u(integral[m])
            for j in range(m + 1):
                rhs += L.dt * (self.QI[m + 1, j] * L.f[j].impl + self.QE[m + 1, j] * L.f[j].expl)

            # implicit solve with prefactor stemming from QI
            L.u[m + 1] = P.solve_system(
                rhs, L.dt * self.QI[m + 1, m + 1], L.u[m + 1], L.time + L.dt * self.coll.nodes[m],
            )
            
            # update function values
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])
           
        # indicate presence of new values at this level
        L.status.updated = True

        return None
        
    def compute_end_point(self):
        """
        Compute u at the right point of the interval

        The value uend computed here is a full evaluation of the Picard formulation unless do_full_update==False

        Returns:
            None
        """
        
        super().compute_end_point()
        
        L=self.level
                
        #breakpoint()     
        self.update_ns(L.uend)
        
  
        return None

    def update_ns(self,Un):
        
        """       
        Returns:
            None
        """
        # get current level and problem description
        tau = 1
        L = self.level
        time = L.time+L.dt
        L.uend = L.prob.NSsolve(Un,time,1)
    
        return None
    
        
        
        
        
        
        
        
    def compute_residual(self, stage=None):
        """
        Computation of the residual using the collocation matrix Q

        Args:
            stage (str): The current stage of the step the level belongs to
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # Check if we want to skip the residual computation to gain performance
        # Keep in mind that skipping any residual computation is likely to give incorrect outputs of the residual!
        if stage in self.params.skip_residual_computation:
            L.status.residual = 0.0 if L.status.residual is None else L.status.residual
            return None

        # check if there are new values (e.g. from a sweep)
        # assert L.status.updated

        # compute the residual for each node

        # build QF(u)
        res_norm = []
        res = self.integrate()
        for m in range(self.coll.num_nodes):
            # This is somewhat ugly, but we have to apply the mass matrix on u0 only on the finest level
            if L.level_index == 0:
                res[m] += P.apply_mass_matrix(L.u[0] - L.u[m + 1])
            else:
                res[m] += L.u[0] - P.apply_mass_matrix(L.u[m + 1])
            # add tau if associated
            if L.tau[m] is not None:
                res[m] += L.tau[m]
            # Due to different boundary conditions we might have to fix the residual
            if L.prob.fix_bc_for_residual:
                L.prob.fix_residual(res[m])
            # use abs function from data type here
            res_norm.append(abs(res[m]))
            
        # Compute the magnitude of the velocity field
        
        """
        fig=plt.figure(1,figsize=(16,13))
        #mesh = P.Mesh()
        #Vm = df.FunctionSpace(mesh, 'CG', 2)
         
        #u_magn = df.sqrt(df.dot(res[m].values, res[m].values))
        #u_magn = df.project(u_magn, Vm)
        
        #for i in range(len(u_magn.vector()[:])):
        #     if(u_magn.vector()[i] < 0.0):
        #        print(u_magn.vector()[i])
        
        # Split the velocity components 
        #ux,uy = res[m].values.split(deepcopy=True)  
        
        #ax=fig.add_subplot(411)    
        #c=df.plot(u_magn, mode='color',cmap = 'jet')
        #plt.colorbar(c)
       
        ax=fig.add_subplot(111)    
        c=df.plot(res[m].values,cmap = 'jet')
        plt.colorbar(c,orientation="horizontal")
        
        #ax=fig.add_subplot(413)
        #c=df.plot(ux,cmap = 'jet')
        #ax.set_title('U component')
        #plt.colorbar(c)
        #plt.draw()
            
            
        #ax=fig.add_subplot(414)
        #c=df.plot(uy,cmap = 'jet')
        #ax.set_title('V component')
        #plt.colorbar(c)
        #plt.draw()
        
        plt.show()
        """
        
        # find maximal residual over the nodes
        L.status.residual = max(res_norm)

        # indicate that the residual has seen the new values
        L.status.updated = False

        return None
            
        
        
