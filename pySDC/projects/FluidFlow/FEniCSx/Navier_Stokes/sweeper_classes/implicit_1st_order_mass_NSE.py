from pySDC.core.sweeper import Sweeper

import matplotlib.pyplot as plt


class implicit_1st_order_mass_NSE(Sweeper):
    """
    Generic implicit sweeper, expecting lower triangular matrix type as input

    Attributes:
        QI: lower triangular matrix
    """

    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """

        if 'QI' not in params:
            params['QI'] = 'IE'

        # call parent's initialization routine
        super().__init__(params)

        # get QI matrix
        self.QI = self.get_Qdelta_implicit(qd_type=self.params.QI)

    @property
    def space_comm(self):
        return self.params.space_comm

    @property
    def space_rank(self):
        return self.space_comm.rank

    def integrate(self):
        """
        Integrates the right-hand side

        Returns:
            list of dtype_u: containing the integral as values
        """

        L = self.level
        P = L.prob

        me = []

        # integrate RHS over all collocation nodes
        for m in range(1, self.coll.num_nodes + 1):
            # new instance of dtype_u, initialize values with 0
            me.append(P.dtype_u(P.init, val=0.0))
            for j in range(1, self.coll.num_nodes + 1):
                me[-1] += L.dt * self.coll.Qmat[m, j] * L.f[j]

        return me

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
        M = self.coll.num_nodes

        # update the MIN-SR-FLEX preconditioner
        if self.params.QI == 'MIN-SR-FLEX':
            self.QI = self.get_Qdelta_implicit(qd_type="MIN-SR-FLEX", k=L.status.sweep)

        # gather all terms which are known already (e.g. from the previous iteration)
        # this corresponds to u0 + QF(u^k) - QdF(u^k) + tau

        # get QF(u^k)
        integral = self.integrate()

        # This is somewhat ugly, but we have to apply the mass matrix on u0 only on the finest level
        if L.level_index == 0:
            u0 = P.apply_mass_matrix(
                L.u[0],
            )
        else:
            u0 = L.u[0]

        for m in range(M):
            # get -QdF(u^k)_m
            for j in range(1, M + 1):
                integral[m] -= L.dt * self.QI[m + 1, j] * L.f[j]

            # add initial value
            integral[m] += u0
            # add tau if associated
            if L.tau[m] is not None:
                integral[m] += L.tau[m]

        # do the sweep
        for m in range(0, M):
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            rhs = P.dtype_u(integral[m])
            for j in range(1, m + 1):
                rhs += L.dt * self.QI[m + 1, j] * L.f[j]

            # implicit solve with prefactor stemming from the diagonal of Qd
            alpha = L.dt * self.QI[m + 1, m + 1]
            if alpha == 0:
                L.u[m + 1] = rhs
            else:
                L.u[m + 1] = P.solve_system(rhs, alpha, L.u[m + 1], L.time + L.dt * self.coll.nodes[m])
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

        # check if Mth node is equal to right point and do_coll_update is false, perform a simple copy
        if self.coll.right_is_node and not self.params.do_coll_update:
            # a copy is sufficient
            L.uend = P.dtype_u(L.u[-1])
        else:
            raise NotImplementedError('Mass matrix sweeper expect u_M = u_end')

        # L.prob.WriteFiles(L.uend, L.time)
        # L.prob.plot(L.uend)
        L.prob.LiftDrag(L.uend, L.time + L.dt)

        return None

    def compute_residual(self, stage=None):
        """
        Computation of the residual using the collocation matrix Q

        Args:
            stage (str): The current stage of the step the level belongs to
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # Check if we want to skip the residual computation to gain performance
        # Keep in mind that skipping any residual computation is likely to give incorrect outputs of the residual!
        if stage in self.params.skip_residual_computation:
            L.status.residual = 0.0 if L.status.residual is None else L.status.residual
            return None

        # check if there are new values (e.g. from a sweep)
        # assert L.status.updated

        # compute the residual for each node

        # build QF(u)
        res_norm = []
        res = self.integrate()
        # res = [0] * (self.coll.num_nodes + 1)
        for m in range(self.coll.num_nodes):

            # This is somewhat ugly, but we have to apply the mass matrix on u0 only on the finest level

            if L.level_index == 0:
                res[m] += P.apply_mass_matrix(L.u[0] - L.u[m + 1])
            else:
                res[m] += L.u[0] - P.apply_mass_matrix(L.u[m + 1])
            # add tau if associated
            if L.tau[m] is not None:
                res[m] += L.tau[m]

            # Due to different boundary conditions we might have to fix the residual
            if L.prob.fix_bc_for_residual:
                L.prob.fix_residual(res[m])

            # use abs function from data type here
            res_norm.append(abs(res[m]))

        # find maximal residual over the nodes
        L.status.residual = max(res_norm)

        if L.time == 0.0 and L.status.residual == 0.0:
            L.status.residual = 1.0

        # indicate that the residual has seen the new values
        L.status.updated = False

        return None
