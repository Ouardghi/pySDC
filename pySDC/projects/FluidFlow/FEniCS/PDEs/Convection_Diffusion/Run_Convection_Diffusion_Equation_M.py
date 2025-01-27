import numpy as np


from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

from pySDC.projects.FluidFlow.FEniCS.PDEs.Convection_Diffusion.problem_classes.ConvectionDiffusion_2D_FEniCS_matrix_forced import (
    fenics_ConvDiff2D,
    fenics_ConvDiff2D_mass,
    fenics_ConvDiff2D_mass_timebc,
)

from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
from pySDC.implementations.transfer_classes.BaseTransfer_mass import base_transfer_mass
from pySDC.implementations.transfer_classes.TransferFenicsMesh import mesh_to_mesh_fenics


from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order

from pySDC.helpers.stats_helper import get_sorted
import pySDC.helpers.plot_helper as plt_helper


from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityPolynomialError
from pySDC.implementations.hooks.log_step_size import LogStepSize


import dolfin as df
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


def run_simulation(ml=None, mass=None):

    t0 = 0
    dt = 0.005
    Tend = 0.2

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-10
    level_params['dt'] = dt

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 10

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3

    problem_params = dict()
    problem_params['nu'] = 0.01
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
    controller_params['hook_class'] = [LogStepSize]
    # Fill description dictionary for easy hierarchy creation
    description = dict()
    if mass:
        description['problem_class'] = fenics_ConvDiff2D_mass_timebc
        description['sweeper_class'] = imex_1st_order_mass
        description['base_transfer_class'] = base_transfer_mass
    else:
        description['problem_class'] = fenics_ConvDiff2D
        description['sweeper_class'] = imex_1st_order
    description['problem_params'] = problem_params
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['space_transfer_class'] = mesh_to_mesh_fenics

    # description['convergence_controllers'] = {AdaptivityPolynomialError: {'e_tol': 1e-4, 'estimate_on_node':2}}

    # quickly generate block of steps
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    errors = get_sorted(stats, type='error', sortby='iter')
    residuals = get_sorted(stats, type='residual', sortby='iter')

    uex = P.u_exact(Tend)

    timestep = get_sorted(stats, type='dt', sortby='time')
    timestep_norecomputed = get_sorted(stats, type='dt', sortby='time', recomputed=False)

    # plt.figure(1)
    plt.plot([me[0] for me in timestep], [me[1] for me in timestep])
    plt.plot([me[0] for me in timestep_norecomputed], [me[1] for me in timestep_norecomputed])
    plt.show()

    error_L2 = df.assemble((uend.values - uex.values) ** 2 * df.dx) ** 0.5
    print(error_L2)

    return errors, residuals


if __name__ == "__main__":

    # errors_sdc_noM, _ = run_simulation(ml=False, mass=False)
    errors_sdc_M, _ = run_simulation(ml=False, mass=True)
    # errors_mlsdc_noM, _ = run_simulation(ml=True, mass=False)
    # errors_mlsdc_M, _ = run_simulation(ml=True, mass=True)
    #

    # np.save('errors_sdc_M.npy',  errors_sdc_M)
    # np.save('errors_sdc_noM.npy',  errors_sdc_noM)
    # np.save('errors_mlsdc_M.npy',  errors_mlsdc_M)
    # np.save('errors_mlsdc_noM.npy',  errors_mlsdc_noM)

    # visualize()
