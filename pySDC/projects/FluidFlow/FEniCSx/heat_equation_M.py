import numpy as np


from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
from pySDC.projects.FluidFlow.FEniCSx.problem_classes.HeatEquation_2D_FEniCSx_matrix_forced import fenicsx_heat_mass

from pySDC.helpers.stats_helper import get_sorted
import pySDC.helpers.plot_helper as plt_helper


def run_simulation():

    t0 = 0
    dt = 0.2
    Tend = 0.2

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-10
    level_params['dt'] = dt

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = [3]

    problem_params = dict()
    problem_params['nu'] = 0.1
    problem_params['t0'] = t0  # ugly, but necessary to set up ProblemClass
    problem_params['nelems'] = [32]
    problem_params['family'] = 'CG'
    problem_params['order'] = [4]
    problem_params['c'] = [0.0]

    problem_params['refinements'] = [1]

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    #controller_params['hook_class'] = fenics_output

    # Fill description dictionary for easy hierarchy creation
    description = dict()
   
    description['problem_class'] = fenicsx_heat_mass
    description['sweeper_class'] = imex_1st_order_mass
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

    # diff = uend.values - uex.values
    # L2_err = np.sqrt(df.assemble(df.inner(diff,diff)*df.dx))
    # L2_ex  = np.sqrt(df.assemble(df.inner(uex.values,uex.values)*df.dx))
    # L2_err_relative = L2_err/L2_ex
    # print('The L2 error at time ', Tend, 'is ', L2_err_relative)
    err_relative = abs(uex - uend) / abs(uex)
    print('The L2 error at time ', Tend, 'is ', err_relative)      

    errors = get_sorted(stats, type='error', sortby='iter')
    residuals = get_sorted(stats, type='residual', sortby='iter')
    print(errors)
    return errors, residuals


if __name__ == "__main__":

    
    run_simulation()
  