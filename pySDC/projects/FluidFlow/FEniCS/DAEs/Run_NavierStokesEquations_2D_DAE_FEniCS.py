from pathlib import Path
import numpy as np

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.FluidFlow.FEniCS.DAEs.problem_classes.NavierStokesEquations_2D_FEniCS_matrix_implicit import (
    fenics_NSE_monolithic,
)
from pySDC.projects.FluidFlow.FEniCS.DAEs.sweepers.fully_implicit_DAE_FEniCS_NSE import fully_implicit_DAE
from pySDC.implementations.transfer_classes.TransferFenicsMesh import mesh_to_mesh_fenics

import dolfin as df
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


def setup(t0=None):
    """
    Helper routine to set up parameters

    Args:
        t0 (float): initial time
        ml (bool): use single or multiple levels

    Returns:
        description and controller_params parameter dictionaries
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-10
    level_params['dt'] = 0.05

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = [3]
    sweeper_params['QI'] = ['LU']
    # sweeper_params['QI'] = ['IEpar']
    # sweeper_params['QI'] = ['MIN']
    # sweeper_params['QI'] = ['MIN-SR-NS']
    # sweeper_params['QI'] = ['MIN-SR-S']

    problem_params = dict()
    problem_params['nu'] = 0.001
    problem_params['t0'] = t0  # ugly, but necessary to set up this ProblemClass
    problem_params['c_nvars'] = [33]
    problem_params['family'] = 'CG'
    problem_params['c'] = 0.0
    problem_params['order'] = [2]
    problem_params['refinements'] = [1]

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20

    base_transfer_params = dict()
    base_transfer_params['finter'] = True

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = None
    description['problem_class'] = fenics_NSE_monolithic
    description['problem_params'] = problem_params
    description['sweeper_class'] = fully_implicit_DAE
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['space_transfer_class'] = mesh_to_mesh_fenics
    description['base_transfer_params'] = base_transfer_params

    return description, controller_params


def main():
    """
    Main routine to run the different implementations of the heat equation with FEniCS

    Args:
        variant (str): specifies the variant
        ml (bool): use single or multiple levels
        num_procs (int): number of processors in time
    """
    Tend = 8.0
    t0 = 0.0

    description, controller_params = setup(t0=t0)

    # quickly generate block of steps
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(0.0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    timing = get_sorted(stats, type='timing_run', sortby='time')
    iter_counts = get_sorted(stats, type='niter', sortby='time')
    niters = np.array([item[1] for item in iter_counts])

    """
    # compute exact solution and compare
    uex = P.u_exact(Tend)
    """

    path = 'data/data_N3_dt_005/'
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


if __name__ == "__main__":
    main()
    plt.show()
