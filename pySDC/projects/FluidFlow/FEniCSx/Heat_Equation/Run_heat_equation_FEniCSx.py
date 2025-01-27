import sys
import numpy as np
from mpi4py import MPI

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.FluidFlow.FEniCSx.Heat_Equation.problem_classes.HeatEquation_2D_FEniCSx_matrix_forced_implicit import (
    fenicsx_heat_mass,
)
from pySDC.projects.FluidFlow.FEniCSx.Heat_Equation.sweeper_classes.implicit_1st_order_mass import (
    implicit_1st_order_mass,
)
from pySDC.projects.FluidFlow.FEniCSx.Heat_Equation.sweeper_classes.implicit_1st_order_mass_MPI_2 import (
    implicit_1st_order_mass_MPI,
)
from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
from pySDC.helpers.stats_helper import get_sorted
import pySDC.helpers.plot_helper as plt_helper


def run_simulation(ml=None, mass=None):

    t0 = 0
    dt = 0.1
    Tend = 1.0

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
    level_params['restol'] = 1e-12
    level_params['dt'] = dt
    # level_params['residual_type'] = 'last_abs'

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 10

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = [4]
    sweeper_params['QI'] = ['MIN-SR-S']
    sweeper_params['comm'] = time_comm

    problem_params = dict()
    problem_params['nu'] = 0.1
    problem_params['t0'] = t0  # ugly, but necessary to set up ProblemClass
    problem_params['nvars'] = [65]
    problem_params['family'] = 'CG'
    problem_params['order'] = [4]
    problem_params['c'] = [0.0]
    problem_params['refinements'] = [0]
    problem_params['comm'] = space_comm

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20 if space_rank == 0 and time_rank == 0 else 99
    # controller_params['hook_class'] = fenics_output

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = fenicsx_heat_mass
    description['sweeper_class'] = implicit_1st_order_mass_MPI
    # description['base_transfer_class'] = base_transfer_mass
    description['problem_params'] = problem_params
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    # description['space_transfer_class'] = mesh_to_mesh_fenics

    # quickly generate block of steps
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    uex = P.u_exact(Tend)
    err_relative = abs(uex - uend) / abs(uex)

    if space_rank == 0 and time_rank == 0:
        print('The L2 error at time ', Tend, 'is ', err_relative)

    return None


if __name__ == "__main__":
    run_simulation()
