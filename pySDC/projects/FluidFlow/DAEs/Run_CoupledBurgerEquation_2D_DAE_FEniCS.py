from pathlib import Path
import numpy as np

from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.FluidFlow.DAEs.problem_classes.CoupledBurgerEquation_2D_FEniCS_matrix_implicit import fenics_heat
from pySDC.projects.FluidFlow.DAEs.sweepers.fully_implicit_DAE_FEniCS import fully_implicit_DAE
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
    level_params['restol'] = 1e-20
    level_params['dt'] = 0.01

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = [3]
    sweeper_params['QI'] = ['LU']
    #sweeper_params['QI'] = ['IEpar']
    #sweeper_params['QI'] = ['MIN']
    #sweeper_params['QI'] = ['MIN-SR-NS']
    #sweeper_params['QI'] = ['MIN-SR-S']

    problem_params = dict()
    problem_params['nu'] = 0.002
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
    description['problem_params'] = problem_params
    description['sweeper_class'] = None
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['space_transfer_class'] = mesh_to_mesh_fenics
    description['base_transfer_params'] = base_transfer_params
    description['problem_class'] = fenics_heat
    description['sweeper_class'] = fully_implicit_DAE 
    
    return description, controller_params


def main():
    """
    Main routine to run the different implementations of the heat equation with FEniCS

    Args:
        variant (str): specifies the variant
        ml (bool): use single or multiple levels
        num_procs (int): number of processors in time
    """
    Tend = 1.0
    t0 = 0.0

    description, controller_params = setup(t0=t0)

    # quickly generate block of steps
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(0.0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # compute exact solution and compare
    uex = P.u_exact(Tend)
    err = abs(uex - uend) / abs(uex)

    #print(uex.values.vector()[:])
    fig=plt.figure(figsize=(8,16))
    
    ax=fig.add_subplot(221,projection='3d')
    df.plot(uend.values.sub(0), mode = 'warp')
    ax.set_xlabel('Distance x')
    ax.set_ylabel('Distance y')
    ax.set_title('SDC-FEniCS Solution')
    plt.draw()
    
    ax=fig.add_subplot(222,projection='3d')
    df.plot(uex.values.sub(0), mode = 'warp')
    ax.set_xlabel('Distance x')
    ax.set_ylabel('Distance y')
    ax.set_title('Exact Solution')
    plt.draw()


    ax=fig.add_subplot(223)
    df.plot(uend.values.sub(0), mode = 'contour', levels = 25, cmap = 'jet')
    plt.axis('scaled')
    plt.xlabel('Distance x')
    plt.ylabel('Distance y')
    plt.title('Numerical Solution')
    plt.draw()
    
    ax=fig.add_subplot(224)
    df.plot(uex.values.sub(0), mode = 'contour', levels = 25, cmap = 'jet')
    plt.axis('scaled')
    plt.xlabel('Distance x')
    plt.ylabel('Distance y')
    plt.title('Exact Solution')
    plt.draw()


    Path("data").mkdir(parents=True, exist_ok=True)
    f = open('data/step_7_A_out.txt', 'a')

    out = f'error at time {Tend}: {err}'
    f.write(out + '\n')
    print(out)

    f.write('\n')
    print()
    f.close()


if __name__ == "__main__":
    main()
    plt.show()
