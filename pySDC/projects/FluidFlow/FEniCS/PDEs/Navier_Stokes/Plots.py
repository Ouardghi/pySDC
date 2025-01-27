import numpy as np
import dolfin as df
import mshr as mshr
import json

from pathlib import Path

import matplotlib.animation as animation

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

# from pySDC.projects.FluidFlow.FEniCS.Get_pySDC_path import Get_pySDC_Path


def main():

    # Get the path to the FEniCS project directory, ensuring consistency regardless of the location
    # of the 'pySDC' folder  and the working directory
    # path0 = Get_pySDC_Path()
    # Add the data directory to the path
    datapath = '/home/ouardghi/Desktop/pySDC_data/'

    path = datapath + 'PDEs/' + 'data/dataLin/data_N4_dt_0.01_LU/'

    '''
    # load parameters
    parameters = json.load(open( path+"parameters.json", 'r' ) )

    # Get the time-step and final time of the simulation
    dt   = parameters['dt']
    Tend = parameters['Tend']
    # Get problem parameters
    c_nvars = parameters['c_nvars'][0]
    mu = 1.0/parameters['Re']
    rho = 1.0
    '''

    # Get the time-step and final time of the simulation
    dt = 1 / 100
    Tend = 8.0
    # Get problem parameters
    mu = 0.001
    rho = 1.0

    # Compute the number of time steps
    nsteps = int(Tend / dt)
    # Open XDMF file for visualization output

    # Open XDMF file for visualization output
    xdmffile_u = df.XDMFFile(path + 'Cylinder_velocity.xdmf')
    xdmffile_p = df.XDMFFile(path + 'Cylinder_pressure.xdmf')

    # Read mesh from the XDMF file
    # mesh = df.Mesh(path+'cylinder.xml.gz')
    mesh = df.Mesh('cylinder.xml')
    # for _ in range(1):
    #        mesh = df.refine(mesh)

    # df.plot(mesh)
    # plt.show()

    # define function spaces for velocity and physical quantities
    V = df.VectorFunctionSpace(mesh, 'CG', 2)
    Q = df.FunctionSpace(mesh, 'CG', 1)

    # Define variables
    un = df.Function(V)
    pn = df.Function(Q)

    print('DoFs on this mesh:', len(un.vector()[:]))

    """
    Vs = df.FunctionSpace(mesh, 'CG', 2)

    # Stiffness matrix for computing the streamline function
    u = df.TrialFunction(Vs)
    v = df.TestFunction(Vs)
    Str = df.Function(Vs)

    p = df.TrialFunction(Q)
    q = df.TestFunction(Q)

    a_s = df.dot(df.nabla_grad(u), df.nabla_grad(v)) * df.dx
    S = df.assemble(a_s)

    mp = df.inner(df.nabla_grad(p), df.as_vector((q, q))) * df.dx
    MP = df.assemble(mp)
    """

    # Define the interface of the obstacle (cylinder)
    # Normal pointing out of obstacle
    n = -df.FacetNormal(mesh)
    Cylinder = df.CompiledSubDomain('on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3')  #
    # Create MeshFunction of topological codimension 1 on given mesh.
    CylinderBoundary = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    Cylinder.mark(CylinderBoundary, 1)
    dsc = df.Measure("ds", domain=mesh, subdomain_data=CylinderBoundary, subdomain_id=1)

    u_t = df.inner(df.as_vector((n[1], -n[0])), un)
    drag = df.Form(2 / 0.1 * (mu / rho * df.inner(df.grad(u_t), n) * n[1] - pn * n[0]) * dsc)
    lift = df.Form(-2 / 0.1 * (mu / rho * df.inner(df.grad(u_t), n) * n[0] + pn * n[1]) * dsc)

    # compraison
    # turek3 = np.loadtxt("data_FEATFLOW/draglift_q2_cn_lv1-6_dt4/bdforces_lv3")
    turek4 = np.loadtxt("data_FEATFLOW/draglift_q2_cn_lv1-6_dt4/bdforces_lv4")

    # Open figure for plots
    fig = plt.figure(1, figsize=(16, 13))

    Lift_coef = []
    Drag_coef = []
    Times = []

    # Time-stepping
    t = 0.0
    for s in range(nsteps):
        # Update current time
        t += dt
        # print((s,t))

        if 0 == 0:
            # if s%10==0:
            # if s%2==0:

            # print((s,t))
            # Read the velocity field u from the XDMF file
            # xdmffile_u.read_checkpoint(un, 'un', 4*s-1)
            # xdmffile_p.read_checkpoint(pn, 'pn', 4*s-1)

            xdmffile_u.read_checkpoint(un, 'un', s)
            xdmffile_p.read_checkpoint(pn, 'pn', s)
            """
            # Compute the vorticity
            ux,uy = un.split(deepcopy=True)
            Vort = uy.dx(0)-ux.dx(1)

            # Compute the streamlines
            l = Vort*v*df.dx
            L = df.assemble(l)
            df.solve(S,Str.vector(),L)

            # Compute the magnitude of the velocity field
            u_magn = df.sqrt(df.dot(un, un))
            u_magn = df.project(u_magn, Vs)
            """

            CD = df.assemble(drag)
            CL = df.assemble(lift)

            Lift_coef.append(CL)
            Drag_coef.append(CD)
            Times.append(t)

            print([s, t, CL, CD])

            # print('Drag coeficient is CD = ',CD)
            # print('Lift coeficient is CL = ',CL)

            ax = fig.add_subplot(511)
            df.plot(un, cmap='jet')
            ax.set_xlabel('Distance x')
            ax.set_ylabel('Distance y')
            ax.set_title('Velocity field')
            ax.set_xlim(-0.01, 2.22)
            ax.set_ylim(-0.005, 0.415)
            plt.draw()

            """
            ax=fig.add_subplot(512)
            c=df.plot(pn,mode='color',cmap = 'jet')
            plt.colorbar(c)
            ax.set_xlabel('Distance x')
            ax.set_ylabel('Distance y')
            ax.set_title('pressure field')
            ax.set_xlim(-0.01,2.22)
            ax.set_ylim(-0.005,0.415)
            plt.draw()

            ax=fig.add_subplot(513)
            c=df.plot(Vort, mode='color',vmin=-30, vmax=30,cmap = 'jet')
            #c=df.plot(Vort, mode='color',cmap = 'jet')
            plt.colorbar(c)
            ax.set_xlabel('Distance x')
            ax.set_ylabel('Distance y')
            ax.set_title('Vorticity')
            ax.set_xlim(0,2.20)
            ax.set_ylim(0,0.41)
            plt.draw()

            ax=fig.add_subplot(514)
            c=df.plot(u_magn, mode='color',cmap = 'jet')
            #plt.colorbar(c)
            ax.set_xlabel('Distance x')
            ax.set_ylabel('Distance y')
            ax.set_title('Magnitude')
            ax.set_xlim(0,2.20)
            ax.set_ylim(0,0.41)
            plt.draw()

            ax=fig.add_subplot(515)
            #c=df.plot(Vort, mode='color',vmin=-50, vmax=50,cmap = 'jet')
            #c=df.plot(u_magn, mode='color',cmap = 'jet')
            #plt.colorbar(c)
            df.plot(Str,mode='contour', levels=50)
            ax.set_xlabel('Distance x')
            ax.set_ylabel('Distance y')
            ax.set_title('Magnitude = Streamlines')
            #ax.set_xlim(-0.01,2.22)
            #ax.set_ylim(-0.005,0.415)
            plt.draw()
            """

            plt.pause(0.01)
            plt.clf()

    fig = plt.figure(2, figsize=(16, 13))
    #
    ax = fig.add_subplot(211)
    plt.plot(Times, Lift_coef, color='k', ls='--')
    plt.plot(
        turek4[1:, 1],
        turek4[1:, 4],
        marker="x",
        markevery=50,
        linestyle="-",
        markersize=4,
        label="FEATFLOW (42016 dofs)",
    )
    ax.set_xlabel('Times')
    ax.set_ylabel('Lift')
    ax.set_xlim(0, 8)

    ax = fig.add_subplot(212)
    plt.plot(Times, Drag_coef, color='k', ls='--')
    plt.plot(
        turek4[1:, 1],
        turek4[1:, 3],
        marker="x",
        markevery=50,
        linestyle="-",
        markersize=4,
        label="FEATFLOW (42016 dofs)",
    )
    ax.set_xlabel('Times')
    ax.set_ylabel('Drag')
    ax.set_xlim(0, 8)


if __name__ == '__main__':
    main()
    plt.show()
