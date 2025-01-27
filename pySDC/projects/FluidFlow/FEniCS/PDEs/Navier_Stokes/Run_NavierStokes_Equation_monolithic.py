import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.FluidFlow.FEniCS.PDEs.Navier_Stokes.problem_classes.NavierStokes_2D_FEniCS_monolithic import (
    fenics_NSE_2D_Monolithic,
)
from pySDC.projects.FluidFlow.FEniCS.PDEs.Navier_Stokes.sweeper_classes.implicit_NSE import implicit_NSE

from pySDC.implementations.transfer_classes.BaseTransfer_mass import base_transfer_mass
from pySDC.implementations.transfer_classes.TransferFenicsMesh import mesh_to_mesh_fenics
from pySDC.playgrounds.FEniCS.HookClass_FEniCS_output import fenics_output
from pySDC.helpers.stats_helper import get_sorted
import pySDC.helpers.plot_helper as plt_helper
from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityPolynomialError
from pySDC.implementations.hooks.log_step_size import LogStepSize


import dolfin as df
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


def run_simulation(mass=None):

    t0 = 0.0
    dt = 1 / 100
    Tend = 8.0

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
    sweeper_params['num_nodes'] = 4
    sweeper_params['QI'] = ['LU']
    # sweeper_params['QI'] = ['IEpar']
    # sweeper_params['QI'] = ['MIN']
    # sweeper_params['QI'] = ['MIN-SR-NS']
    # sweeper_params['QI'] = ['MIN-SR-S']
    # sweeper_params['QI'] = ['MIN-SR-FLEX']

    problem_params = dict()
    problem_params['nu'] = 0.001
    problem_params['t0'] = t0  # ugly, but necessary to set up ProblemClass
    problem_params['c_nvars'] = [64]
    problem_params['family'] = 'CG'
    problem_params['order'] = [2]
    problem_params['c'] = 0.0
    problem_params['sigma'] = 0.05
    problem_params['refinements'] = [1]

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20

    # controller_params['hook_class'] = fenics_output
    controller_params['hook_class'] = LogStepSize

    # Fill description dictionary for easy hierarchy creation
    description = dict()

    description['problem_class'] = fenics_NSE_2D_Monolithic
    description['sweeper_class'] = implicit_NSE
    description['problem_params'] = problem_params
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['space_transfer_class'] = mesh_to_mesh_fenics

    # description['convergence_controllers'] = {AdaptivityPolynomialError: {'e_tol': 1e-4, 'estimate_on_node':2, 'interpolate_between_restarts': False}}

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

    # uex = P.u_exact(Tend)
    # error_L2 = df.assemble((uend.values - uex.values)**2 * df.dx)**0.5
    # print(error_L2)

    timing = get_sorted(stats, type='timing_run', sortby='time')
    iter_counts = get_sorted(stats, type='niter', sortby='time')
    niters = np.array([item[1] for item in iter_counts])

    # timestep = get_sorted(stats, type='dt', sortby='time')
    # timestep_norecomputed = get_sorted(stats, type='dt', sortby='time', recomputed=False)

    # plt.figure(1)
    # plt.plot([me[0] for me in timestep], [me[1] for me in timestep])
    # plt.plot([me[0] for me in timestep_norecomputed], [me[1] for me in timestep_norecomputed])
    # plt.show()

    path = 'data/dataLin/data_N4_dt_0.01_LU/'
    f = open(path + 'Iter_counts.txt', 'w')

    out = f'Time to solution: {timing[0][1]:6.4f} sec.'
    f.write(out + '\n')
    print(out)

    out = '   Total number of iterations: %4i' % np.sum(niters)
    f.write(out + '\n')
    print(out)

    for i in range(len(iter_counts)):
        out = '%4i  %4.2f  %4i' % (i, iter_counts[i][0], iter_counts[i][1])
        f.write(out + '\n')
        # print(out)

    return errors, residuals


if __name__ == "__main__":

    errors_sdc_noM, _ = run_simulation(mass=False)
    # errors_sdc_M, _ = run_simulation(mass=True)
