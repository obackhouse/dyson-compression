"""Block Lanczos for moments of the self-energy.
"""

from collections import defaultdict
import numpy as np
from pyscf import lib

from dyson import util, Solver, BlockLanczosSymmGF, BlockLanczosNoSymmGF


class V:
    def __init__(self, zero, symmetry=True):
        self.symmetry = symmetry
        self.zero = zero
        self.data = {}

    def __getitem__(self, key):
        i, n, j = key
        if i == 0 or j == 0:
            return self.zero
        elif (not self.symmetry) or i >= j:
            return self.data[i, n, j]
        else:
            return self[j, n, i].T.conj()

    def __setitem__(self, key, val):
        i, n, j = key
        if (not self.symmetry) or i >= j:
            self.data[i, n, j] = val
        else:
            self.data[j, n, i] = val.T.conj()


class BlockLanczosSymmSE(BlockLanczosSymmGF):
    """Block Lanczos for Hermitian moments of the self-energy.

    Input
    -----
        h_phys: ndarray (n, n)
            Effective single-particle Hamiltonian in the physical
            space, i.e. the Fock matrix.
        moments: ndarray (m, n, n)
            Moments of the self-energy with which one wishes to
            produce a consistent set of Green's function poles to.

    Options
    -------
        eig_tol: float
            Threshold for eigenvalues to be considered zero for singular
            matrices (default value is 1e-14).

    Output
    ------
        e: ndarray (k,)
            Energy of the Green's function poles.
        v: ndarray (n, k)
            Coupling of the Green's function poles.
    """

    def __init__(self, h_phys, moments, **kwargs):
        self.eig_tol = 1e-14

        h_phys = np.asarray(h_phys)
        moments = np.asarray(moments)
        assert h_phys.shape == moments.shape[1:]
        assert np.allclose(h_phys, h_phys.T.conj())
        assert np.allclose(moments, moments.swapaxes(1, 2).conj())

        Solver.__init__(self, h_phys, moments, **kwargs)

    def get_blocks(self):
        """Get on- and off-diagonal blocks of the block tridiagonalised
        Hamiltonian.
        """

        α = np.zeros((self.niter+1, self.norb, self.norb), dtype=self.dtype)
        β = np.zeros((self.niter, self.norb, self.norb), dtype=self.dtype)

        v = V(self.zero)

        βsq = self.moments[0]
        β[0] = self.mat_sqrt(βsq)
        βinv = self.mat_isqrt(βsq)
        for i in range(self.nmom):
            v[1, i, 1] = np.linalg.multi_dot((
                βinv.T.conj(),
                self.moments[i],
                βinv,
            ))

        def vb(i, n, j):
            return np.dot(v[i, n, j], β[j].T.conj())

        def av(i, n, j):
            return np.dot(v[i, 1, i], v[i, n, j])

        for i in range(1, self.niter):
            βsq = (
                + v[i, 2, i]
                - lib.hermi_sum(vb(i, 1, i-1))
                - np.dot(v[i, 1, i], v[i, 1, i].T.conj())
            )

            if i > 1:
                βsq += np.dot(β[i-1], β[i-1].T.conj())

            β[i] = self.mat_sqrt(βsq)
            βinv = self.mat_isqrt(βsq)

            for n in range(2 * (self.niter-i)):
                if n != (2 * (self.niter-i) - 1):
                    r = (
                        + v[i, n+1, i]
                        - vb(i, n, i-1).T.conj()
                        - av(i, n, i)
                    )
                    v[i+1, n, i] = np.dot(βinv.T.conj(), r)

                r = (
                    + v[i, n+2, i]
                    - lib.hermi_sum(vb(i, n+1, i-1))
                    - lib.hermi_sum(av(i, n+1, i))
                    + lib.hermi_sum(np.dot(v[i, 1, i], vb(i, n, i-1)))
                    + np.dot(β[i-1], vb(i-1, n, i-1))
                    + np.dot(av(i, n, i), v[i, 1, i].T.conj())
                )
                v[i+1, n, i+1] = np.linalg.multi_dot((βinv.T.conj(), r, βinv))

        for i in range(self.niter+1):
            α[i] = v[i, 1, i]

        return α, β

    def get_auxiliaries(self):
        α, β = self.get_blocks()
        orth = β[0].T.conj()
        h_tri = self.build_block_tridiagonal(α, β)

        e, u = np.linalg.eigh(h_tri[self.norb:, self.norb:])
        u = np.dot(orth, u[:self.norb])

        return e, u

    def kernel(self):
        e, u = self.get_auxiliaries()
        h_aux = np.block([
            [self.h_phys, u],
            [u.T.conj(),  np.diag(e)],
        ])

        e, v = np.linalg.eigh(h_aux)
        v = v[:self.norb]

        for i in range(min(5, len(e))):
            self.log.info(" %s: %s", i, util.format_value(e[i]))
        self.log.info(" ...")

        return e, v

    @property
    def h_phys(self):
        return self.inp[0]

    @property
    def moments(self):
        return self.inp[1]

    @property
    def niter(self):
        return (self.nmom - 2) // 2 + 1



if __name__ == "__main__":
    np.set_printoptions(edgeitems=100, linewidth=1000, precision=4)

    nmom = 4
    norb = 5
    naux = 100

    f = np.diag(np.random.random((norb,)))
    e_aux = np.random.random((naux,)) * 5.0
    v_aux = np.random.random((norb, naux)) - 0.5
    m1 = np.einsum("xk,yk,nk->nxy", v_aux, v_aux.conj(), e_aux[None]**np.arange(nmom)[:, None])
    solver = BlockLanczosSymmSE(f, m1)
    e_aux, v_aux = solver.get_auxiliaries()
    m2 = np.einsum("xk,yk,nk->nxy", v_aux, v_aux.conj(), e_aux[None]**np.arange(nmom)[:, None])
    for i, (a, b) in enumerate(zip(m1, m2)):
        a1 = a / np.max(np.abs(a))
        b1 = b / np.max(np.abs(b))
        assert np.allclose(a1, b1)

    f = np.diag(np.random.random((norb,)))
    e_aux = np.random.random((naux,)) * 5.0
    v_aux = np.random.random((norb, naux)) - 0.5 + (np.random.random((norb, naux)) - 0.5) * 1.0j
    m1 = np.einsum("xk,yk,nk->nxy", v_aux, v_aux.conj(), e_aux[None]**np.arange(nmom)[:, None])
    solver = BlockLanczosSymmSE(f, m1)
    e_aux, v_aux = solver.get_auxiliaries()
    m2 = np.einsum("xk,yk,nk->nxy", v_aux, v_aux.conj(), e_aux[None]**np.arange(nmom)[:, None])
    for i, (a, b) in enumerate(zip(m1, m2)):
        a1 = a / np.max(np.abs(a))
        b1 = b / np.max(np.abs(b))
        assert np.allclose(a1, b1)
