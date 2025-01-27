import sys
import numpy as np
from mpi4py import MPI

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.FluidFlow.FEniCSx.problem_classes.NavierStokes_2D_FEniCSx_matrix_forced_implicit import (
    fenicsx_NSE_mass,
)

from pySDC.projects.FluidFlow.FEniCSx.sweeper_classes.implicit_1st_order_mass_NSE import implicit_1st_order_mass_NSE
from pySDC.projects.FluidFlow.FEniCSx.sweeper_classes.implicit_1st_order_mass_NSE_MPI_2 import (
    implicit_1st_order_mass_NSE_MPI,
)

# from pySDC.playgrounds.FEniCSx.HookClass_FEniCS_output import fenics_output

from pySDC.helpers.stats_helper import get_sorted
import pySDC.helpers.plot_helper as plt_helper


def run_simulation(ml=None, mass=None):

    t0 = 0
    dt = 0.01
    Tend = 0.1

    # solver = 'LinSolv'
    solver = 'NonLinSolv'

    # space_comm = MPI.COMM_WORLD
    comm = MPI.COMM_WORLD

    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    # split world communicator to create space-communicators
    if len(sys.argv) >= 2:
        color = int(world_rank / int(sys.argv[1]))
    else:
        color = int(world_rank / 1)
    space_comm = comm.Split(color=color)
    space_rank = space_comm.Get_rank()

    # split world communicator to create time-communicators
    if len(sys.argv) >= 2:
        color = int(world_rank % int(sys.argv[1]))
    else:
        color = int(world_rank / world_size)
    time_comm = comm.Split(color=color)
    time_rank = time_comm.Get_rank()

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
    sweeper_params['num_nodes'] = [4]
    # sweeper_params['QI'] = ['LU']
    # sweeper_params['QI'] = ['MIN-SR-FLEX']
    sweeper_params['time_comm'] = time_comm
    sweeper_params['space_comm'] = space_comm

    problem_params = dict()
    problem_params['nu'] = 0.001
    problem_params['t0'] = t0  # ugly, but necessary to set up ProblemClass
    problem_params['nelems'] = [32]
    problem_params['family'] = 'CG'
    problem_params['order'] = [2]
    problem_params['comm'] = space_comm

    # problem_params['output_folder'] = 'Output'

    if ml:
        problem_params['refinements'] = [1, 0]
    else:
        problem_params['refinements'] = [1]

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20 if space_rank == 0 and time_rank == 0 else 99
    # controller_params['hook_class'] = fenics_output

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    if mass:
        description['problem_class'] = fenicsx_NSE_mass
        description['sweeper_class'] = implicit_1st_order_mass_NSE  # _MPI
        # description['base_transfer_class'] = base_transfer_mass
    description['problem_params'] = problem_params
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    # description['space_transfer_class'] = mesh_to_mesh_fenics

    Folder = 'data_' + solver + f'_N={sweeper_params['num_nodes'][0]}' + f'_dt={dt}' + f'_PC={sweeper_params['QI'][0]}'

    from pathlib import Path

    folder = Path('Output/' + Folder)
    folder.mkdir(exist_ok=True, parents=True)

    ofiles_lftdrag = 'Output/' + Folder + '/Liftdrag.txt'
    f = open(ofiles_lftdrag, 'w')
    f.write('# time, lift and drag coefficients' + '\n')
    # f.close

    problem_params['ofiles'] = [[ofiles_lftdrag]]

    # quickly generate block of steps
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return None


if __name__ == "__main__":

    # run_simulation(ml=False, mass=False)
    run_simulation(ml=False, mass=True)
