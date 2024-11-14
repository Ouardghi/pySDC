import numpy as np
from mpi4py import MPI

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.FluidFlow.FEniCSx.problem_classes.HeatEquation_2D_FEniCSx_matrix_forced_implicit import fenicsx_heat_mass

from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
from pySDC.projects.FluidFlow.FEniCSx.sweeper_classes.implicit_1st_order_mass import implicit_1st_order_mass
from pySDC.projects.FluidFlow.FEniCSx.sweeper_classes.implicit_1st_order_mass_MPI import implicit_1st_order_mass_MPI

from pySDC.helpers.stats_helper import get_sorted
import pySDC.helpers.plot_helper as plt_helper


def run_simulation(ml=None, mass=None):

    t0 = 0
    dt = 0.2
    Tend = 0.2

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-10
    level_params['dt'] = dt
    level_params['residual_type'] = 'last_abs'

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 10

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = [3]
    sweeper_params['QI'] = ['MIN-SR-S']
    sweeper_params['comm'] = MPI.COMM_WORLD

    problem_params = dict()
    problem_params['nu'] = 0.1
    problem_params['t0'] = t0  # ugly, but necessary to set up ProblemClass
    problem_params['nelems'] = [32]
    problem_params['family'] = 'CG'
    problem_params['order'] = [4]
    problem_params['c'] = [0.0]
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
    if mass:
        description['problem_class'] = fenicsx_heat_mass
        description['sweeper_class'] = implicit_1st_order_mass#_MPI 
        
        #description['base_transfer_class'] = base_transfer_mass

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


def visualize():

    errors_sdc_M = np.load('errors_sdc_M.npy')
    errors_sdc_noM = np.load('errors_sdc_noM.npy')
    errors_mlsdc_M = np.load('errors_mlsdc_M.npy')
    errors_mlsdc_noM = np.load('errors_mlsdc_noM.npy')

    plt_helper.setup_mpl()

    plt_helper.newfig(240, 1, ratio=0.8)

    plt_helper.plt.semilogy(
        [err[0] for err in errors_sdc_noM],
        [err[1] for err in errors_sdc_noM],
        lw=2,
        marker='s',
        markersize=6,
        color='darkblue',
        label='SDC without M',
    )

    plt_helper.plt.xlim([0, 11])
    plt_helper.plt.ylim([6e-09, 2e-03])
    plt_helper.plt.xlabel('iteration')
    plt_helper.plt.ylabel('error')
    plt_helper.plt.legend()
    plt_helper.plt.grid()

    plt_helper.savefig('error_SDC_noM_CG_4')

    plt_helper.newfig(240, 1, ratio=0.8)

    plt_helper.plt.semilogy(
        [err[0] for err in errors_sdc_noM],
        [err[1] for err in errors_sdc_noM],
        lw=2,
        color='darkblue',
        marker='s',
        markersize=6,
        label='SDC without M',
    )
    plt_helper.plt.semilogy(
        [err[0] for err in errors_sdc_M],
        [err[1] for err in errors_sdc_M],
        lw=2,
        marker='o',
        markersize=6,
        color='red',
        label='SDC with M',
    )

    plt_helper.plt.xlim([0, 11])
    plt_helper.plt.ylim([6e-09, 2e-03])
    plt_helper.plt.xlabel('iteration')
    plt_helper.plt.ylabel('error')
    plt_helper.plt.legend()
    plt_helper.plt.grid()

    plt_helper.savefig('error_SDC_M_CG_4')

    plt_helper.newfig(240, 1, ratio=0.8)

    plt_helper.plt.semilogy(
        [err[0] for err in errors_mlsdc_noM],
        [err[1] for err in errors_mlsdc_noM],
        lw=2,
        marker='s',
        markersize=6,
        color='darkblue',
        label='MLSDC without M',
    )

    plt_helper.plt.xlim([0, 11])
    plt_helper.plt.ylim([6e-09, 2e-03])
    plt_helper.plt.xlabel('iteration')
    plt_helper.plt.ylabel('error')
    plt_helper.plt.legend()
    plt_helper.plt.grid()

    plt_helper.savefig('error_MLSDC_noM_CG_4')

    plt_helper.newfig(240, 1, ratio=0.8)

    plt_helper.plt.semilogy(
        [err[0] for err in errors_mlsdc_noM],
        [err[1] for err in errors_mlsdc_noM],
        lw=2,
        color='darkblue',
        marker='s',
        markersize=6,
        label='MLSDC without M',
    )
    plt_helper.plt.semilogy(
        [err[0] for err in errors_mlsdc_M],
        [err[1] for err in errors_mlsdc_M],
        lw=2,
        marker='o',
        markersize=6,
        color='red',
        label='MLSDC with M',
    )

    plt_helper.plt.xlim([0, 11])
    plt_helper.plt.ylim([6e-09, 2e-03])
    plt_helper.plt.xlabel('iteration')
    plt_helper.plt.ylabel('error')
    plt_helper.plt.legend()
    plt_helper.plt.grid()

    plt_helper.savefig('error_MLSDC_M_CG_4')


if __name__ == "__main__":

    #errors_sdc_noM, _ = run_simulation(ml=False, mass=False)
    errors_sdc_M, _ = run_simulation(ml=False, mass=True)
    # errors_mlsdc_noM, _ = run_simulation(ml=True, mass=False)
    # errors_mlsdc_M, _ = run_simulation(ml=True, mass=True)
    #
    # np.save('errors_sdc_M.npy',  errors_sdc_M)
    # np.save('errors_sdc_noM.npy',  errors_sdc_noM)
    # np.save('errors_mlsdc_M.npy',  errors_mlsdc_M)
    # np.save('errors_mlsdc_noM.npy',  errors_mlsdc_noM)

    # visualize()
