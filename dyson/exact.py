"""Exact solver for the matrix of an unfolded self-energy.
"""

import numpy as np

from dyson import util, Solver


class ExactSymm(Solver):
    """Exact solver for the matrix of a Hermitian self-energy.

    Input
    -----
        matrix: ndarray (k, k)
            Dense array representing the self-energy unfolded into its
            full space.

    Output
    ------
        e: ndarray (k,)
            Energy of the Green's function poles.
        v: ndarray (n, k)
            Coupling of the Green's function poles.
    """

    def __init__(self, matrix, **kwargs):
        Solver.__init__(self, matrix, **kwargs)

    def eig(self, *args, **kwargs):
        return np.linalg.eigh(*args, **kwargs)

    def kernel(self):
        e, v = self.eig(self.matrix)

        self.log.info("eigenvalues:")
        for i in range(min(5, len(e))):
            self.log.info(" %s: %s", i, util.format_value(e[i]))
        self.log.info(" ...")

        return e, v

    def preamble(self):
        self.log.info("shape: %s", repr(self.matrix.shape))
        self.log.info("dtype: %s", self.dtype)

    @property
    def matrix(self):
        return self.inp[0]

    @property
    def dtype(self):
        return self.matrix.dtype

    @property
    def complex(self):
        return np.iscomplexobj(self.dtype)

    @property
    def real(self):
        return not self.complex


class ExactNoSymm(ExactSymm):
    """Exact solver for the matrix of a non-Hermitian self-energy.

    Input
    -----
        matrix: ndarray (k, k)
            Dense array representing the self-energy unfolded into its
            full space.

    Output
    ------
        e: ndarray (k,)
            Energy of the Green's function poles.
        v: ndarray (n, k)
            Coupling of the Green's function poles.
    """

    def eig(self, *args, **kwargs):
        w, v = np.linalg.eig(*args, **kwargs)
        idx = np.argsort(w.real)
        w, v = w[idx], v[:, idx]
        return w, v
