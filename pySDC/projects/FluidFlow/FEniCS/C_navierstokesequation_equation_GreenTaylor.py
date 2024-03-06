import numpy as np


from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

# Problem class
from pySDC.projects.FluidFlow.FEniCS.problem_classes.NavierStokes_GreenTaylor_2D_FEniCS_matrix_forced import fenics_NSE_mass, fenics_NSE_mass_timebc

# Sweepers
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
from pySDC.projects.FluidFlow.FEniCS.imex_1st_order_mass_NSE import imex_1st_order_mass_NSE

from pySDC.implementations.transfer_classes.BaseTransfer_mass import base_transfer_mass
from pySDC.implementations.transfer_classes.TransferFenicsMesh import mesh_to_mesh_fenics


from pySDC.playgrounds.FEniCS.HookClass_FEniCS_output import fenics_output

from pySDC.helpers.stats_helper import get_sorted
import pySDC.helpers.plot_helper as plt_helper

import dolfin as df
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

def run_simulation(ml=None, mass=None):

    t0 = 0.0
    dt = 1/1000
    Tend =  dt

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-15
    level_params['dt'] = dt

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 5

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = [2]

    problem_params = dict()
    problem_params['nu'] = 0.001
    problem_params['t0'] = t0  # ugly, but necessary to set up ProblemClass
    problem_params['c_nvars'] = [32]
    problem_params['family'] = 'CG'
    problem_params['order'] = [2]
    problem_params['c'] = 0.0
    problem_params['sigma'] = 0.05
    if ml:
        problem_params['refinements'] = [1, 0]
    else:
        problem_params['refinements'] = [1]

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    #controller_params['hook_class'] = fenics_output

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = fenics_NSE_mass_timebc
    description['sweeper_class'] = imex_1st_order_mass_NSE
    description['base_transfer_class'] = base_transfer_mass
    description['problem_params'] = problem_params
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['space_transfer_class'] = mesh_to_mesh_fenics

    # quickly generate block of steps
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    
    P.CloseXDMFfile()
    
    errors = get_sorted(stats, type='error', sortby='iter')
    residuals = get_sorted(stats, type='residual', sortby='iter')
    
    uex = P.u_exact(Tend)
    error_L2 = df.assemble((uend.values - uex.values)**2 * df.dx)**0.5
    print('Velocity L2-error :', error_L2)
    

    fig=plt.figure(figsize=(8,16))
    
    plt.clf()
    ax=fig.add_subplot(221)
    c=df.plot(uex.values,cmap = 'jet')
    ax.set_title('Velo. Exact')
    plt.colorbar(c)
    plt.draw()

    #ax=fig.add_subplot(222)
    #c=df.plot(pex,cmap = 'jet')
    #ax.set_title('Press. Exact')
    #plt.colorbar(c)
    #plt.draw()

    ax=fig.add_subplot(223)
    c=df.plot(uend.values,cmap = 'jet')
    ax.set_title('Velo. Num.')
    plt.colorbar(c)
    plt.draw()


    #ax=fig.add_subplot(224)
    #c=df.plot(pend,cmap = 'jet')
    #ax.set_title('Press. Num.')
    #plt.colorbar(c)
    #plt.draw()
  
         
    plt.show()
    
    return errors, residuals


if __name__ == "__main__":

    #errors_sdc_noM, _ = run_simulation(ml=False, mass=False)
    errors_sdc_M, _ = run_simulation(ml=False, mass=True)
    # errors_mlsdc_noM, _ = run_simulation(ml=True, mass=False)
    # errors_mlsdc_M, _ = run_simulation(ml=True, mass=True)
    #
    
    #np.save('errors_sdc_M.npy',  errors_sdc_M)
    #np.save('errors_sdc_noM.npy',  errors_sdc_noM)
    # np.save('errors_mlsdc_M.npy',  errors_mlsdc_M)
    # np.save('errors_mlsdc_noM.npy',  errors_mlsdc_noM)
