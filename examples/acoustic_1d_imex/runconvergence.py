import numpy as np
from matplotlib import pyplot as plt

import pySDC.deprecated.PFASST_stepwise as mp
from ProblemClass_conv import acoustic_1d_imex
from examples.acoustic_1d_imex.HookClass import plot_solution
from pySDC import CollocationClasses as collclass
from pySDC import Log
from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order


if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 1

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-14

    sparams = {}
  
    # This comes as read-in for the problem class
    pparams = {}
    pparams['cadv']      = 0.1
    pparams['cs']        = 1.00
    pparams['order_adv'] = 5
    pparams['waveno']    = 5

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class']     = acoustic_1d_imex
    description['problem_params']    = pparams
    description['dtype_u']           = mesh
    description['dtype_f']           = rhs_imex_mesh
    
    ### SET TYPE OF QUADRATURE NODES ###
    #description['collocation_class'] = collclass.CollGaussLobatto
    #description['collocation_class'] = collclass.CollGaussLegendre
    description['collocation_class'] = collclass.CollGaussRadau_Right
    
    description['sweeper_class']     = imex_1st_order
    description['do_coll_update']    = True
    description['level_params']      = lparams
    description['hook_class']        = plot_solution
    
    nsteps = np.zeros((3,9))
    nsteps[0,:] = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    nsteps[1,:] = nsteps[0,:]
    nsteps[2,:] = nsteps[0,:]
    
    for order in [3, 4, 5]:

      error  = np.zeros(np.shape(nsteps)[1])
    
      # setup parameters "in time"
      t0   = 0
      Tend = 1.0
      
      if order==3:
        file = open('conv-data.txt', 'w')
      else:
        file = open('conv-data.txt', 'a')

      ### SET NUMBER OF NODES DEPENDING ON REQUESTED ORDER ###
      if order==3:
        description['num_nodes'] = 3
      elif order==4:
        description['num_nodes'] = 3
      elif order==5:
        description['num_nodes'] = 3
      
      sparams['maxiter'] = order
      
      for ii in range(0,np.shape(nsteps)[1]):
      
        ns = nsteps[order-3,ii]
        if ((order==3) or (order==4)):
          pparams['nvars']     = [(2,2*ns)]
        elif order==5:
          pparams['nvars'] = [(2,2*ns)]
          
        # quickly generate block of steps
        MS = mp.generate_steps(num_procs,sparams,description)
        
        dt = Tend/float(ns)

        # get initial values on finest level
        P = MS[0].levels[0].prob
        uinit = P.u_exact(t0)
        if ii==0:
          print "Time step: %4.2f" % dt
          print "Fast CFL number: %4.2f" % (pparams['cs']*dt/P.dx) 
          print "Slow CFL number: %4.2f" % (pparams['cadv']*dt/P.dx) 

        # call main function to get things done...
        uend,stats = mp.run_pfasst(MS, u0=uinit, t0=t0, dt=dt, Tend=Tend)

        # compute exact solution and compare
        uex = P.u_exact(Tend)

        error[ii] = np.linalg.norm(uex.values-uend.values,np.inf)/np.linalg.norm(uex.values,np.inf)
        file.write(str(order)+"    "+str(ns)+"    "+str(error[ii])+"\n")

      file.close()

    if np.shape(nsteps)[1]==1:
      fig = plt.figure(figsize=(8,8))

      plt.plot(P.mesh, uex.values[0,:],  '+', color='b', label='u (exact)')
      plt.plot(P.mesh, uend.values[0,:], '-', color='b', label='u (SDC)')
      plt.plot(P.mesh, uex.values[1,:],  '+', color='r', label='p (exact)')
      plt.plot(P.mesh, uend.values[1,:], '-', color='r', label='p (SDC)')
      plt.legend()
      plt.xlim([0.0, 1.0])
      plt.ylim([-1.0, 1.0])
      plt.show()
    else:
      for ii in range(0,np.shape(nsteps)[1]):
        print('error for nsteps= %s: %s' % (nsteps[order-3,ii], error[ii]))
