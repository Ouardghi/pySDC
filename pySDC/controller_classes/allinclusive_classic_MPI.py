
import numpy as np

from pySDC.Controller import controller
from pySDC.Step import step
from pySDC.Stats import stats

class allinclusive_classic_MPI(controller):
    """

    PFASST controller, running parallel version of PFASST in blocks (MG-style)

    """

    def __init__(self, step_params, description, comm):
        """
       Initialization routine for PFASST controller

       Args:
           S: local steps
           comm: MPI communicator
       """

        # call parent's initialization routine
        super(allinclusive_classic_MPI, self).__init__()

        # create single step per processor
        self.S = step(step_params)
        self.S.generate_hierarchy(description)

        # pass communicator for future use
        self.comm = comm
        # add request handle container for isend
        self.req_send = []
        # add request handler for status send
        self.req_status = None

    def run(self, u0, t0, dt, Tend):
        """
        Main driver for running the parallel version of SDC, MSSDC, MLSDC and PFASST

        Args:
            u0: initial values
            t0: starting time
            dt: (initial) time step
            Tend: ending time

        Returns:
            end values on the finest level
            stats object containing statistics for each step, each level and each iteration
        """

        # fixme: use error classes for send/recv and stage errors

        rank = self.comm.Get_rank()
        num_procs = self.comm.Get_size()
        all_dt = self.comm.allgather(dt)
        self.S.status.dt = dt
        self.S.status.time = t0 + sum(all_dt[0:rank])
        self.S.status.step = rank
        self.S.status.slot = rank
        active = self.S.status.time < Tend - 10*np.finfo(float).eps
        comm_active = self.comm.Split(active)

        uend = u0

        while active:

            rank = comm_active.Get_rank()
            num_procs = comm_active.Get_size()

            # initialize block of steps with u0
            self.restart_block(num_procs,uend)
            # call pre-start hook
            self.S.levels[0].hooks.dump_pre(self.S.status)

            while not self.S.status.done:
                self.pfasst(comm_active,num_procs)

            tend = comm_active.bcast(self.S.status.time+self.S.status.dt, root=num_procs-1)
            uend = comm_active.bcast(self.S.levels[0].uend, root=num_procs-1)
            stepend = comm_active.bcast(self.S.status.step, root=num_procs-1)

            all_dt = comm_active.allgather(self.S.status.dt)
            self.S.status.time = stepend + rank + 1
            self.S.status.time = tend + sum(all_dt[0:rank])
            active =  self.S.status.time < Tend - 10 * np.finfo(float).eps
            comm_active = comm_active.Split(active)

        comm_active.Free()
        uend = self.comm.bcast(uend, root=num_procs-1)

        return uend, stats.return_stats()


    def restart_block(self,size,u0):
        """
        Helper routine to reset/restart block of (active) steps

        Args:
            size: number of active time steps
            u0: initial value to distribute across the steps

        Returns:
            block of (all) steps
        """

        # store link to previous step
        self.S.prev = self.S.status.slot - 1
        self.S.next = self.S.status.slot + 1
        # resets step
        self.S.reset_step()
        # determine whether I am the first and/or last in line
        self.S.status.first = self.S.prev == -1
        self.S.status.last = self.S.next == size
        # intialize step with u0
        self.S.init_step(u0)
        # reset some values
        self.S.status.done = False
        self.S.status.iter = 1
        self.S.status.stage = 'SPREAD'
        for l in self.S.levels:
            l.tag = None
        self.req_status = None
        self.req_send = []

    @staticmethod
    def recv(target,source,tag,comm):
        """
        Receive function

        Args:
            target: level which will receive the values
            source: level which initiated the send
            tag: identifier to check if this message is really for me
        """

        target.u[0] = comm.recv(source=source, tag=tag)
        target.f[0] = target.prob.eval_f(target.u[0], target.time)

    @staticmethod
    def send(source,target,tag,comm):
        """
        Send function

        Args:
            source: level which has the new values
            tag: identifier for this message
        """
        # sending here means computing uend ("one-sided communication")
        comm.send(source.uend, dest = target, tag = tag)


    def predictor(self,comm):
        """
        Predictor function, extracted from the stepwise implementation (will be also used by matrix sweppers)

        """

        # restrict to coarsest level
        for l in range(1, len(self.S.levels)):
            self.S.transfer(source=self.S.levels[l-1],target=self.S.levels[l])


        for p in range(self.S.status.slot+1):

            if not p == 0 and not self.S.status.first:
                self.recv(target=self.S.levels[-1], source=self.S.prev, tag=self.S.status.iter, comm=comm)

            # do the sweep with new values
            self.S.levels[-1].sweep.update_nodes()
            self.S.levels[-1].sweep.compute_end_point()

            if not self.S.status.last:
                self.send(source=self.S.levels[-1], target=self.S.next, tag=self.S.status.iter, comm=comm)

        # interpolate back to finest level
        for l in range(len(self.S.levels)-1,0,-1):
            self.S.transfer(source=self.S.levels[l],target=self.S.levels[l-1])

    def pfasst(self,comm,num_procs):
        """
        Main function including the stages of SDC, MLSDC and PFASST (the "controller")

        For the workflow of this controller, check out one of our PFASST talks

        Args:
            MS: all active steps

        Returns:
            all active steps
        """

        stage = self.S.status.stage
        # print(self.comm.Get_rank(),stage)
        if stage == 'SPREAD':
            # (potentially) serial spreading phase

            # first stage: spread values
            self.S.levels[0].hooks.pre_step(self.S.status)

            # call predictor from sweeper
            self.S.levels[0].sweep.predict()

            # update stage
            if (len(self.S.levels) > 1 and num_procs > 1) and self.S.params.predict:  # MLSDC or PFASST
                self.S.status.stage = 'PREDICT'
            elif num_procs > 1 and len(self.S.levels) > 1:  # PFASST
                self.S.levels[0].hooks.dump_pre_iteration(self.S.status)
                self.S.status.stage = 'IT_FINE'
            elif num_procs > 1 and len(self.S.levels) == 1:  # MSSDC
                self.S.levels[0].hooks.dump_pre_iteration(self.S.status)
                self.S.status.stage = 'IT_COARSE'
            elif num_procs == 1:  # SDC
                self.S.levels[0].hooks.dump_pre_iteration(self.S.status)
                self.S.status.stage = 'IT_FINE'
            else:
                print("Don't know what to do after spread, aborting")
                exit()

        elif stage == 'PREDICT':

            # call predictor (serial)

            self.predictor(comm)

            # update stage
            self.S.levels[0].hooks.dump_pre_iteration(self.S.status)
            self.S.status.stage = 'IT_FINE'

        elif stage == 'IT_FINE':

            # do fine sweep

            # standard sweep workflow: update nodes, compute residual, log progress
            self.S.levels[0].sweep.update_nodes()
            self.S.levels[0].sweep.compute_residual()
            self.S.levels[0].hooks.dump_sweep(self.S.status)

            # wait for pending sends before computing uend, if any
            if len(self.req_send) > 0 and not self.S.status.last and self.S.params.fine_comm:
                self.req_send[0].wait()

            self.S.levels[0].sweep.compute_end_point()

            if not self.S.status.last and self.S.params.fine_comm:
                self.req_send.append(comm.isend(self.S.levels[0].uend, dest=self.S.next, tag=0))

            # update stage
            self.S.status.stage = 'IT_CHECK'

        elif stage == 'IT_CHECK':

            # check whether to stop iterating (parallel)

            # increment iteration count here (and only here)
            self.S.status.iter += 1
            self.S.levels[0].hooks.dump_iteration(self.S.status)

            # check if an open request of the status send is pending
            if not self.req_status is None:
                self.req_status.wait()

            # check for convergence or abort
            self.S.status.done = self.check_convergence(self.S)

            # send status forward
            if not self.S.status.last:
                self.req_status = comm.isend(self.S.status.done, dest=self.S.next, tag=99)

            # recv status
            if not self.S.status.first and not self.S.status.prec_done:
                self.S.status.prec_done = comm.recv(source=self.S.prev, tag=99)

            # if I'm not done or the guy left of me is not done, keep doing stuff
            if not self.S.status.done or not self.S.status.prec_done:
                # multi-level or single-level?
                if len(self.S.levels) > 1:  # MLSDC or PFASST
                    self.S.status.stage = 'IT_UP'
                elif num_procs > 1:  # MSSDC
                    self.S.status.stage = 'IT_COARSE'
                elif num_procs == 1:  # SDC
                    self.S.status.stage = 'IT_FINE'

            else:
                self.S.levels[0].sweep.compute_end_point()
                self.S.levels[0].hooks.dump_step(self.S.status)
                self.S.status.stage = 'DONE'

        elif stage == 'IT_UP':

            # go up the hierarchy from finest to coarsest level (parallel)

            self.S.transfer(source=self.S.levels[0],target=self.S.levels[1])

            # sweep and send on middle levels (not on finest, not on coarsest, though)
            for l in range(1,len(self.S.levels)-1):
                self.S.levels[l].sweep.update_nodes()
                self.S.levels[l].sweep.compute_residual()
                self.S.levels[l].hooks.dump_sweep(self.S.status)

                # wait for pending sends before computing uend, if any
                if len(self.req_send) > l and not self.S.status.last and self.S.params.fine_comm:
                    self.req_send[l].wait()

                self.S.levels[l].sweep.compute_end_point()

                if not self.S.status.last and self.S.params.fine_comm:
                    self.req_send.append(comm.isend(self.S.levels[l].uend,dest=self.S.next,tag=l))

                # transfer further up the hierarchy
                self.S.transfer(source=self.S.levels[l],target=self.S.levels[l+1])

            # update stage
            self.S.status.stage = 'IT_COARSE'

        elif stage == 'IT_COARSE':

            # sweeps on coarsest level (serial/blocking)

            # receive from previous step (if not first)
            if not self.S.status.first and not self.S.status.prec_done:
                self.recv(target=self.S.levels[-1], source=self.S.prev, tag=len(self.S.levels)-1, comm=comm)

            # do the sweep
            self.S.levels[-1].sweep.update_nodes()
            self.S.levels[-1].sweep.compute_residual()
            self.S.levels[-1].hooks.dump_sweep(self.S.status)
            self.S.levels[-1].sweep.compute_end_point()

            # send to next step
            if not self.S.status.last:
                self.send(source=self.S.levels[-1], target=self.S.next, tag=len(self.S.levels)-1, comm=comm)

            # update stage
            if len(self.S.levels) > 1:  # MLSDC or PFASST
                self.S.status.stage = 'IT_DOWN'
            else:  # MSSDC
                self.S.status.stage = 'IT_CHECK'

        elif stage == 'IT_DOWN':

            # prolong corrections down to finest level (parallel)

            # receive and sweep on middle levels (except for coarsest level)
            for l in range(len(self.S.levels)-1,0,-1):

                if not self.S.status.first and self.S.params.fine_comm:
                    self.recv(target=self.S.levels[l-1], source=self.S.prev, tag=l-1, comm=comm)

                # prolong values
                self.S.transfer(source=self.S.levels[l],target=self.S.levels[l-1])

                # on middle levels: do sweep as usual
                if l-1 > 0:
                    self.S.levels[l-1].sweep.update_nodes()
                    self.S.levels[l-1].sweep.compute_residual()
                    self.S.levels[l-1].hooks.dump_sweep(self.S.status)

            # update stage
            self.S.status.stage = 'IT_FINE'

        else:
            #fixme: use meaningful error object here
            print('Something is wrong here, you should have hit one case statement!')
            exit()



