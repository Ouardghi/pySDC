from pySDC.core.Errors import TransferError
from pySDC.core.SpaceTransfer import space_transfer
from pySDC.playgrounds.mpifft.FFT_datatype import fft_datatype, rhs_imex_fft
from mpi4py_fft import PFFT
import numpy as np
import time

class fft_to_fft(space_transfer):
    """
    Custon base_transfer class, implements Transfer.py

    This implementation can restrict and prolong between PMESH datatypes meshes with FFT for periodic boundaries

    """

    def __init__(self, fine_prob, coarse_prob, params):
        """
        Initialization routine

        Args:
            fine_prob: fine problem
            coarse_prob: coarse problem
            params: parameters for the transfer operators
        """
        # invoke super initialization
        super(fft_to_fft, self).__init__(fine_prob, coarse_prob, params)

        Nf = list(self.fine_prob.fft.global_shape())
        Nc = list(self.coarse_prob.fft.global_shape())
        self.ratio = [int(nf / nc) for nf, nc in zip(Nf, Nc)]
        axes = tuple(range(len(Nf)))
        self.fft_pad = PFFT(self.coarse_prob.params.comm, Nc, padding=self.ratio, axes=axes, dtype=np.float, slab=True)

    def restrict(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """
        t0 = time.time()
        if isinstance(F, fft_datatype):
            G = self.coarse_prob.dtype_u(self.coarse_prob.init)
            tmpF = self.fine_prob.fft.backward(F.values)
            tmpG = tmpF[::int(self.ratio[0]), ::int(self.ratio[1])]
            G.values = self.coarse_prob.fft.forward(tmpG, G.values)
        elif isinstance(F, rhs_imex_fft):
            raise NotImplementedError()
        else:
            raise TransferError('Unknown data type, got %s' % type(F))
        t1 = time.time()
        # print(f'Space restrict: {t1 - t0}')
        return G

    def prolong(self, G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """
        t0 = time.time()
        if isinstance(G, fft_datatype):
            F = self.fine_prob.dtype_u(self.fine_prob.init)
            tmpF = self.fft_pad.backward(G.values)
            F.values = self.fine_prob.fft.forward(tmpF, F.values)
        elif isinstance(G, rhs_imex_fft):
            F = self.fine_prob.dtype_f(self.fine_prob.init)
            tmpF = self.fft_pad.backward(G.impl.values)
            F.impl.values = self.fine_prob.fft.forward(tmpF, F.impl.values)
            tmpF = self.fft_pad.backward(G.expl.values)
            F.expl.values = self.fine_prob.fft.forward(tmpF, F.expl.values)
        else:
            raise TransferError('Unknown data type, got %s' % type(G))
        t1 = time.time()
        # print(f'Space interpolate: {t1 - t0}')
        return F
