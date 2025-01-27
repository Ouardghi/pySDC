from mpi4py import MPI
from pySDC.core.sweeper import Sweeper, ParameterError
import logging
from pySDC.projects.FluidFlow.FEniCSx.Navier_Stokes.sweeper_classes.implicit_1st_order_mass import (
    implicit_1st_order_mass,
)


class implicit_1st_order_mass_NSE_MPI(Sweeper):
    """
    MPI based sweeper where each rank administers one collocation node. Adapt sweepers to MPI by use of multiple inheritance.
    See for example the `generic_implicit_MPI` sweeper, which has a class definition:

    ```
    class generic_implicit_MPI(SweeperMPI, generic_implicit):
    ```

    Implicit sweeper parallelized across the nodes.
    Please supply a communicator as `comm` to the parameters!


    Attributes:
        rank (int): MPI rank
    """

    def __init__(self, params):
        self.logger = logging.getLogger('sweeper')

        if 'comm' not in params.keys():
            params['comm'] = MPI.COMM_WORLD
            self.logger.debug('Using MPI.COMM_WORLD for the time communicator because none was supplied in the params.')

        if 'QI' not in params:
            params['QI'] = 'IE'

        super().__init__(params)

        if self.coll.num_nodes % self.params.comm.size != 0:
            raise NotImplementedError(
                f'The communicator in the {type(self).__name__} sweeper needs to distribute the nodes across the processes for parallel processing! The number of nodes (got {self.coll.num_nodes}) must be divisible by the number of processes (got {self.params.comm.size})'
            )

        # get QI matrix
        self.QI = self.get_Qdelta_implicit(qd_type=self.params.QI)

        # get the number of nodes per process
        self.NNPP = int(self.coll.num_nodes / self.comm.Get_size())

    @property
    def comm(self):
        return self.params.comm

    @property
    def rank(self):
        return self.comm.rank

    def integrate(self, last_only=False):
        """
        Integrates the right-hand side

        Args:
            last_only (bool): Integrate only the last node for the residual or all of them

        Returns:
            list of dtype_u: containing the integral as values
        """
        L = self.level
        P = L.prob

        uLocal = self.NNPP * [None]

        me = P.dtype_u(P.init, val=0.0)

        for m in range(self.coll.num_nodes):

            root = int(m / self.NNPP)  # The process to which the mth node belongs
            iln = int(m % self.NNPP)  # The index of the local node

            sendBuf = P.dtype_u(P.init, val=0.0)
            for j in range(self.NNPP):
                i = self.rank * self.NNPP + j
                sendBuf += L.dt * self.coll.Qmat[m + 1, i + 1] * L.f[i + 1]

            recvBuf = me if root == self.rank else None
            self.comm.Reduce(sendBuf, recvBuf, root=root, op=MPI.SUM)

            if self.rank == root:
                uLocal[iln] = me.copy()

        return uLocal

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes

        Returns:
            None
        """

        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked

        # get number of collocation nodes for easier access

        # update the MIN-SR-FLEX preconditioner
        if self.params.QI == 'MIN-SR-FLEX':
            self.QI = self.get_Qdelta_implicit(qd_type="MIN-SR-FLEX", k=L.status.sweep)

        # gather all terms which are known already (e.g. from the previous iteration)
        # this corresponds to u0 + QF(u^k) - QdF(u^k) + tau

        # get QF(u^k)
        rhs = self.integrate()

        if L.level_index == 0:
            u0 = P.apply_mass_matrix(L.u[0])
        else:
            u0 = L.u[0]

        for i in range(self.NNPP):
            m = self.rank * self.NNPP + i
            # get -QdF(u^k)
            rhs[i] -= L.dt * self.QI[m + 1, m + 1] * L.f[m + 1]

            # add initial value
            rhs[i] += u0

            # add tau if associated
            if L.tau[m] is not None:
                rhs[i] += L.tau[m]

        for i in range(self.NNPP):
            m = self.rank * self.NNPP + i

            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            # nothing todo here as we are using diagonal preconditioner QI

            # implicit solve with prefactor stemming from the diagonal of Qd
            L.u[m + 1][:] = P.solve_system(
                rhs[i],
                L.dt * self.QI[m + 1, m + 1],
                L.u[m + 1],
                L.time + L.dt * self.coll.nodes[m],
            )

            # update function values
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        # indicate presence of new values at this level
        L.status.updated = True

        return None

    def compute_end_point(self):
        """
        Compute u at the right point of the interval

        The value uend computed here is a full evaluation of the Picard formulation unless do_full_update==False

        Returns:
            None
        """

        L = self.level
        P = L.prob
        L.uend = P.dtype_u(P.init, val=0.0)

        # check if Mth node is equal to right point and do_coll_update is false, perform a simple copy
        if self.coll.right_is_node and not self.params.do_coll_update:
            # a copy is sufficient
            root = self.comm.Get_size() - 1
            if self.comm.rank == root:
                L.uend[:] = L.u[-1]
            self.comm.Bcast(L.uend, root=root)
        else:
            raise NotImplementedError('Mass matrix sweeper expect u_M = u_end')

        # L.prob.WriteFiles(L.uend, L.time)
        # L.prob.plot(L.uend)
        # L.prob.LiftDrag(L.uend, L.time + L.dt)

        return None

    def compute_residual(self, stage=None):
        """
        Computation of the residual using the collocation matrix Q

        Args:
            stage (str): The current stage of the step the level belongs to
        """

        L = self.level
        P = L.prob

        # Check if we want to skip the residual computation to gain performance
        # Keep in mind that skipping any residual computation is likely to give incorrect outputs of the residual!
        if stage in self.params.skip_residual_computation:
            L.status.residual = 0.0 if L.status.residual is None else L.status.residual
            return None

        # compute the residual for each node

        # build QF(u)
        res = self.integrate(last_only=L.params.residual_type[:4] == 'last')

        res_norm = []

        for i in range(self.NNPP):
            m = self.rank * self.NNPP + i

            if L.level_index == 0:
                res[i] += P.apply_mass_matrix(L.u[0] - L.u[m + 1])
            else:
                res[i] += L.u[0] - P.apply_mass_matrix(L.u[m + 1])

            # add tau if associated
            if L.tau[m] is not None:
                res[i] += L.tau[m]

            # Due to different boundary conditions we might have to fix the residual
            if P.fix_bc_for_residual:
                P.fix_residual(res[i])

            # use abs function from data type here
            res_norm.append(abs(res[i]))

        # find maximal residual over the nodes
        if L.params.residual_type == 'full_abs':
            L.status.residual = self.comm.allreduce(max(res_norm), op=MPI.MAX)
        elif L.params.residual_type == 'last_abs':
            L.status.residual = self.comm.bcast(max(res_norm), root=self.comm.size - 1)
        elif L.params.residual_type == 'full_rel':
            L.status.residual = self.comm.allreduce(max(res_norm) / abs(L.u[0]), op=MPI.MAX)
        elif L.params.residual_type == 'last_rel':
            L.status.residual = self.comm.bcast(max(res_norm) / abs(L.u[0]), root=self.comm.size - 1)
        else:
            raise NotImplementedError(f'residual type \"{L.params.residual_type}\" not implemented!')

        if L.time == 0.0 and L.status.residual == 0.0:
            L.status.residual = 1.0

        # indicate that the residual has seen the new values
        L.status.updated = False

        return None
