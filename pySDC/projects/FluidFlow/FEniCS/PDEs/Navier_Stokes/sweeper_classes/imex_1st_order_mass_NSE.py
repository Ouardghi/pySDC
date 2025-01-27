from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order

import dolfin as df
import matplotlib.pyplot as plt


class imex_1st_order_mass_NSE(imex_1st_order):
    """
    Custom sweeper class, implements Sweeper.py

    First-order IMEX sweeper using implicit/explicit Euler as base integrator, with mass or weighting matrix
    """

    def __init__(self, params):

        super().__init__(params)

        self.gradP = [None] * (self.coll.num_nodes + 1)
        self.p = [None] * (self.coll.num_nodes + 1)

    def integrate_new(self):
        """
        Integrates the right-hand side (here impl + expl)

        Returns:
            list of dtype_u: containing the integral as values
        """

        # get current level and problem description
        L = self.level

        me = []

        # integrate RHS over all collocation nodes
        for m in range(1, self.coll.num_nodes + 1):
            if self.gradP[1] is not None:
                me.append(L.dt * self.coll.Qmat[m, 1] * (L.f[1].impl + L.f[1].expl + self.gradP[1]))
            else:
                me.append(L.dt * self.coll.Qmat[m, 1] * (L.f[1].impl + L.f[1].expl))
            # new instance of dtype_u, initialize values with 0
            for j in range(2, self.coll.num_nodes + 1):
                if self.gradP[j] is not None:
                    me[m - 1] += L.dt * self.coll.Qmat[m, j] * (L.f[j].impl + L.f[j].expl + self.gradP[j])
                else:
                    me[m - 1] += L.dt * self.coll.Qmat[m, j] * (L.f[j].impl + L.f[j].expl)
        return me

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes

        Returns:
            None
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked

        # get number of collocation nodes for easier access
        M = self.coll.num_nodes

        # gather all terms which are known already (e.g. from the previous iteration)
        # this corresponds to u0 + QF(u^k) - QIFI(u^k) - QEFE(u^k) + tau

        # get QF(u^k)
        integral = self.integrate()

        # This is somewhat ugly, but we have to apply the mass matrix on u0 only on the finest level
        if L.level_index == 0:
            u0 = P.apply_mass_matrix(L.u[0])
        else:
            u0 = L.u[0]

        for m in range(M):
            # subtract QIFI(u^k)_m + QEFE(u^k)_m
            for j in range(M + 1):
                integral[m] -= L.dt * (self.QI[m + 1, j] * L.f[j].impl + self.QE[m + 1, j] * L.f[j].expl)
            # add initial value
            integral[m] += u0
            # add tau if associated
            if L.tau[m] is not None:
                integral[m] += L.tau[m]

        # do the sweep
        for m in range(0, M):
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            rhs = P.dtype_u(integral[m])
            for j in range(m + 1):
                rhs += L.dt * (self.QI[m + 1, j] * L.f[j].impl + self.QE[m + 1, j] * L.f[j].expl)

            # implicit solve with prefactor stemming from QI
            L.uold = L.u.copy()

            L.u[m + 1] = P.solve_system(
                rhs,
                L.dt * self.QI[m + 1, m + 1],
                L.u[m + 1],
                L.time + L.dt * self.coll.nodes[m],
                L.dt * self.coll.nodes[m],
            )

            # update function values
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

            # self.gradP[m + 1] = P.eval_gradP(self.p[m + 1])

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

        # get current level and problem description
        L = self.level
        P = L.prob

        # check if Mth node is equal to right point and do_coll_update is false, perform a simple copy
        if self.coll.right_is_node and not self.params.do_coll_update:
            # a copy is sufficient
            L.uend = P.dtype_u(L.u[-1])
        else:
            raise NotImplementedError('Mass matrix sweeper expect u_M = u_end')

        """
        time = L.time+L.dt
        if(time<=L.dt):
            L.uend = P.ApplayBC(L.uend,time)
            print('Called at time ', time)
        """
        P.WriteFiles(L.uend, L.time)

        return None

    def compute_residual(self, stage=None):
        """
        Computation of the residual using the collocation matrix Q

        Args:
            stage (str): The current stage of the step the level belongs to
        """

        # get current level and problem description
        L = self.level
        # P = L.prob

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
        # res = self.integrate_new()
        res = [0] * (self.coll.num_nodes + 1)
        for m in range(self.coll.num_nodes):

            # This is somewhat ugly, but we have to apply the mass matrix on u0 only on the finest level
            """
            if L.level_index == 0:
                res[m] += P.apply_mass_matrix(L.u[0] - L.u[m + 1])
            else:
                res[m] += L.u[0] - P.apply_mass_matrix(L.u[m + 1])
            # add tau if associated
            if L.tau[m] is not None:
                res[m] += L.tau[m]
            """
            if L.uold[m + 1] is not None:
                res[m] = L.u[m + 1] - L.uold[m + 1]
            else:
                res[m] = L.u[m + 1]

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
