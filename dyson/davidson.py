"""Davidson for matrix-vector product of the self-energy.
"""

import numpy as np
from pyscf import lib

from dyson import util, Solver


class DavidsonSymm(Solver):
    """Davidson for the matrix-vector product of a Hermitian
    self-energy.
    
    Input
    -----
        matvec: callable
            Function which returns the dot product between the matrix
            representing the self-energy unfolded into its full space,
            with an arbitrary state vector.
        diag: ndarray (m,)
            Precomputed diagonal of the self-energy unfolded into its
            full space, for generating guesses and preconditioning.

    Options
    -------
        nroots: int
            Number of eigenvalues to compute (default value is 5).
        conv_tol: float
            Threshold for convergence in eigenvalues (default value is
            1e-9).
        lindep_tol: float
            Threshold for convergence in eigenvector linear dependency
            (default value is 1e-14).
        max_cycle: int
            Maximum number of iterations (default value is 100).
        max_space: int
            Maximum size of vector space (default value is 12).
        koopmans: bool
            Target states with large overlap with the initial guess
            (default value is False).

    Output
    ------
        e: ndarray (nroots,)
            Energy of the Green's function poles.
        v: ndarray (n, nroots)
            Coupling of the Green's function poles.
        conv: list (nroots,)
            Whether each root converged.
    """

    def __init__(self, matvec, diag, **kwargs):
        self.nroots = 5
        self.conv_tol = 1e-9
        self.lindep_tol = 1e-14
        self.max_cycle = 100
        self.max_space = 16
        self.koopmans = False

        Solver.__init__(self, matvec, diag, **kwargs)

    def get_picker(self, guesses=None):
        """Get function to pick eigenvalues.
        """

        if not self.koopmans:
            def pick(w, v, nroots, envs):
                return lib.linalg_helper.pick_real_eigs(w, v, nroots, envs)
        else:
            assert guesses is not None

            def pick(w, v, nroots, envs):
                x0 = lib.linalg_helper._gen_x0(envs["v"], envs["xs"])
                s = lib.dot(np.asarray(guesses).conj(), np.asarray(x0).T)
                s = lib.einsum("pi,pi->i", s.conj(), s)
                idx = np.argsort(-s)[:self.nroots]
                return lib.linalg_helper._eigs_cmplx2real(w, v, idx, self.real)

        return pick

    def get_guesses(self):
        """Get the guesses, by default using a vector with `nroots`
        ones placed in the index of the largest (absolute) values.
        """

        guesses = np.zeros((self.nroots, self.diag.size), dtype=self.dtype)
        inds = np.argsort(np.abs(self.diag))

        for i in range(5):
            guesses[i, inds[i]] = 1.0

        return guesses

    def get_precond(self):
        """Get the preconditioner, by default just use the diagonal.
        """

        return self.diag

    def davidson(self, *args, **kwargs):
        return lib.davidson1(*args, **kwargs)

    def kernel(self):
        matvecs = lambda xs: [self.matvec(x) for x in xs]
        precond = self.get_precond()
        guesses = list(self.get_guesses())
        pick = self.get_picker(guesses=guesses)

        conv, e, v = self.davidson(
                matvecs,
                guesses,
                precond,
                tol=self.conv_tol,
                max_cycle=self.max_cycle,
                max_space=self.max_space,
                lindep=self.lindep_tol,
                max_memory=1e9,
                nroots=self.nroots,
                pick=pick,
                verbose=0,
        )

        mask = np.argsort(e)
        conv = [conv[x] for x in mask]
        e = e[mask]
        v = np.array([v[x] for x in mask]).T

        for i in range(self.nroots):
            conv_str = "(*)" if not conv[i] else ""
            self.log.info(" %s: %s %s", i, util.format_value(e[i]), conv_str)
        if not all(conv):
            self.log.info(" (*) = not converged!")

        return e, v, conv

    def preamble(self):
        self.log.info("shape: %s", repr((self.diag.size, self.diag.size)))
        self.log.info("dtype: %s", self.dtype)
        self.log.info("nroots: %d", self.nroots)
        self.log.info("conv_tol: %s", self.conv_tol)
        self.log.info("lindep_tol: %s", self.lindep_tol)
        self.log.info("max_cycle: %d", self.max_cycle)
        self.log.info("max_space: %d", self.max_space)
        self.log.info("koopmans: %r", self.koopmans)

    @property
    def matvec(self):
        return self.inp[0]

    @property
    def diag(self):
        return self.inp[1]

    @property
    def dtype(self):
        return self.diag.dtype

    @property
    def complex(self):
        return np.iscomplexobj(self.dtype)

    @property
    def real(self):
        return not self.complex


class DavidsonNoSymm(DavidsonSymm):
    """Davidson for the matrix-vector product of a non-Hermitian
    self-energy.
    
    Input
    -----
        matvec: callable
            Function which returns the dot product between the matrix
            representing the self-energy unfolded into its full space,
            with an arbitrary state vector.
        diag: ndarray (m,)
            Precomputed diagonal of the self-energy unfolded into its
            full space, for generating guesses and preconditioning.

    Options
    -------
        nroots: int
            Number of eigenvalues to compute (default value is 5).
        conv_tol: float
            Threshold for convergence in eigenvalues (default value is
            1e-9).
        lindep_tol: float
            Threshold for convergence in eigenvector linear dependency
            (default value is 1e-14).
        maxiter: int
            Maximum number of iterations (default value is 100).
        maxspace: int
            Maximum size of vector space (default value is 12).
        koopmans: bool
            Target states with large overlap with the initial guess
            (default value is False).

    Output
    ------
        e: ndarray (nroots,)
            Energy of the Green's function poles.
        v: ndarray (n, nroots)
            Coupling of the Green's function poles.
        conv: list (nroots,)
            Whether each root converged.
    """

    def get_picker(self, guesses=None):
        """Get function to pick eigenvalues.
        """

        if not self.koopmans:
            def pick(w, v, nroots, envs):
                idx = np.argsort(w.real)
                return w, v, idx[:self.nroots]
        else:
            assert guesses is not None

            def pick(w, v, nroots, envs):
                x0 = lib.linalg_helper._gen_x0(envs["v"], envs["xs"])
                s = lib.dot(np.asarray(guesses).conj(), np.asarray(x0).T)
                s = lib.einsum("pi,pi->i", s.conj(), s)
                idx = np.argsort(-s)[:self.nroots]
                return lib.linalg_helper._eigs_cmplx2real(w, v, idx, self.real)

        return pick

    def davidson(self, *args, **kwargs):
        return lib.davidson_nosym1(*args, **kwargs)
