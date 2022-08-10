"""Block Lanczos for moments of the Green's function.
"""
# TODO caching

from collections import defaultdict
import numpy as np

from dyson import util, Solver


def BlockLanczosGF(moments, **kwargs):
    moments = np.asarray(moments)
    if np.allclose(moments, moments.swapaxes(1, 2).conj()):
        return BlockLanczosSymmGF(moments, **kwargs)
    else:
        return BlockLanczosNoSymmGF(moments, **kwargs)


class BlockLanczosSymmGF(Solver):
    """Block Lanczos for Hermitian moments of the Green's function.

    Input
    -----
        moments: ndarray (m, n, n)
            Moments of the Green's function with which one wishes to
            produce a consistent set of Green's function poles to.

    Options
    -------
        eig_tol: float
            Threshold for eigenvalues to be considered zero for singular
            matrices (default value is 1e-14).

    Output
    ------
        e: ndarray (k, )
            Energy of the Green's function poles.
        v: ndarray (n, k)
            Coupling of the Green's function poles to each orbital.
    """

    def __init__(self, moments, **kwargs):
        self.eig_tol = 1e-14

        moments = np.asarray(moments)

        Solver.__init__(self, moments, **kwargs)

    def mat_sqrt(self, array):
        """Return the square-root of a matrix.
        """

        w, v = np.linalg.eigh(array)

        if self.real:
            mask = w >= 0
            w, v = w[mask], v[:, mask]
            fac = 0.5
        else:
            fac = 0.5 + 0j

        return np.dot(v * w[None]**fac, v.T.conj())

    def mat_isqrt(self, array):
        """Return the inverse square-root of a matrix.
        """

        w, v = np.linalg.eigh(array)

        if self.real:
            mask = w > self.eig_tol
            fac = -0.5
        else:
            mask = np.abs(w) > self.eig_tol
            fac = -0.5 + 0j

        w, v = w[mask], v[:, mask]

        return np.dot(v * w[None]**fac, v.T.conj())

    def get_blocks(self):
        """Get on- and off-diagonal blocks of the block tridiagonalised
        Hamiltonian.
        """

        α = np.zeros((self.niter+1, self.norb, self.norb), dtype=self.dtype)
        β = np.zeros((self.niter, self.norb, self.norb), dtype=self.dtype)
        t = np.zeros((self.nmom, self.norb, self.norb), dtype=self.dtype)

        v = defaultdict(lambda: self.zero)
        v[0, 0] = np.eye(self.norb).astype(self.dtype)

        orth = self.mat_isqrt(self.moments[0])
        for i in range(self.nmom):
            t[i] = np.linalg.multi_dot((
                orth,
                self.moments[i],
                orth,
            ))

        α[0] = t[1]

        for i in range(self.niter):
            βsq = self.zero
            for j in range(i+2):
                for l in range(i+1):
                    βsq += np.linalg.multi_dot((
                        v[i, l].T.conj(),
                        t[j+l+1],
                        v[i, j-1],
                    ))

            βsq -= np.dot(α[i], α[i])
            if i:
                βsq -= np.dot(β[i-1], β[i-1])

            β[i] = self.mat_sqrt(βsq)
            βinv = self.mat_isqrt(βsq)

            for j in range(i+2):
                r = (
                        + v[i, j-1]
                        - np.dot(v[i, j], α[i])
                        - np.dot(v[i-1, j], β[i-1])
                )
                v[i+1, j] = np.dot(r, βinv)

            for j in range(i+2):
                for l in range(i+2):
                    α[i+1] += np.linalg.multi_dot((
                        v[i+1, l].T.conj(),
                        t[j+l+1],
                        v[i+1, j],
                    ))

        return α, β

    def build_block_tridiagonal(self, α, β):
        h = np.block([[
            α[i] if i == j else
            β[j] if j == i-1 else
            β[i].T.conj() if i == j-1 else self.zero
            for j in range(len(α))]
            for i in range(len(α))]
        )

        return h

    def kernel(self):
        α, β = self.get_blocks()
        orth = self.mat_sqrt(self.moments[0])
        h_tri = self.build_block_tridiagonal(α, β)

        e, u = np.linalg.eigh(h_tri)
        u = np.dot(orth, u[:self.norb])

        for i in range(min(5, len(e))):
            self.log.info(" %s: %s", i, util.format_value(e[i]))
        self.log.info(" ...")

        return e, u

    def preamble(self):
        self.log.info("shape: %s", repr((self.norb, self.norb)))
        self.log.info("dtype: %s", self.dtype)
        self.log.info("niter: %s", self.niter)
        self.log.info("nmom: %s", self.nmom)
        self.log.info("eig_tol: %s", self.eig_tol)

    @property
    def moments(self):
        return self.inp[0]

    @property
    def norb(self):
        return self.moments.shape[-1]

    @property
    def nmom(self):
        return self.moments.shape[0]

    @property
    def niter(self):
        return (self.nmom - 2) // 2

    @property
    def dtype(self):
        return self.moments.dtype

    @property
    def complex(self):
        return np.iscomplexobj(self.dtype)

    @property
    def real(self):
        return not self.complex

    @property
    def zero(self):
        return np.zeros((self.norb, self.norb), dtype=self.dtype)


class BlockLanczosNoSymmGF(BlockLanczosSymmGF):
    """Block Lanczos for non-Hermitian moments of the Green's function.

    Input
    -----
        moments: ndarray (m, n, n)
            Moments of the Green's function with which one wishes to
            produce a consistent set of Green's function poles to.

    Options
    -------
        eig_tol: float
            Threshold for eigenvalues to be considered zero for singular
            matrices (default value is 1e-14).

    Output
    ------
        e: ndarray (k, )
            Energy of the Green's function poles.
        v: ndarray (n, k)
            Left-hand coupling of the Green's function poles to each
            orbital.
        u: ndarray (n, k)
            Right-hand coupling of the Green's function poles to each
            orbital.
    """

    def __init__(self, moments, **kwargs):
        self.eig_tol = 1e-14

        moments = np.asarray(moments)

        Solver.__init__(self, moments, **kwargs)

    def mat_sqrt(self, array):
        """Return the square-root of a matrix.
        """

        w, vl = np.linalg.eig(array)
        vr = np.linalg.inv(vl)

        return np.dot(vl * w[None]**(0.5+0j), vr)

    def mat_isqrt(self, array):
        """Return the inverse square-root of a matrix.
        """

        w, v = np.linalg.eig(array)

        mask = np.abs(w) > self.eig_tol
        w = w[mask]
        vl = v[:, mask]
        vr = np.linalg.inv(v)[mask]

        return np.dot(vl * w[None]**(-0.5+0j), vr)

    def get_blocks(self):
        """Get on- and off-diagonal blocks of the block tridiagonalised
        Hamiltonian.
        """

        α = np.zeros((self.niter+1, self.norb, self.norb), dtype=self.dtype)
        β = np.zeros((self.niter, self.norb, self.norb), dtype=self.dtype)
        γ = np.zeros((self.niter, self.norb, self.norb), dtype=self.dtype)
        t = np.zeros((self.nmom, self.norb, self.norb), dtype=self.dtype)

        v = defaultdict(lambda: self.zero)
        w = defaultdict(lambda: self.zero)
        v[0, 0] = np.eye(self.norb).astype(self.dtype)
        w[0, 0] = np.eye(self.norb).astype(self.dtype)

        orth = self.mat_isqrt(self.moments[0])
        for i in range(self.nmom):
            t[i] = np.linalg.multi_dot((
                orth,
                self.moments[i],
                orth,
            ))

        α[0] = t[1]

        for i in range(self.niter):
            βsq = self.zero
            γsq = self.zero
            for j in range(i+2):
                for l in range(i+1):
                    βsq += np.linalg.multi_dot((
                        w[i, l],
                        t[j+l+1],
                        v[i, j-1],
                    ))
                    γsq += np.linalg.multi_dot((
                        w[i, j-1],
                        t[j+l+1],
                        v[i, l],
                    ))

            βsq -= np.dot(α[i], α[i])
            γsq -= np.dot(α[i], α[i])
            if i:
                βsq -= np.dot(γ[i-1], γ[i-1])
                γsq -= np.dot(β[i-1], β[i-1])

            β[i] = self.mat_sqrt(βsq)
            γ[i] = self.mat_sqrt(γsq)
            βinv = self.mat_isqrt(βsq)
            γinv = self.mat_isqrt(γsq)

            for j in range(i+2):
                r = (
                        + v[i, j-1]
                        - np.dot(v[i, j], α[i])
                        - np.dot(v[i-1, j], β[i-1])
                )
                v[i+1, j] = np.dot(r, γinv)

                s = (
                        + w[i, j-1]
                        - np.dot(α[i], w[i, j])
                        - np.dot(γ[i-1], w[i-1, j])
                )
                w[i+1, j] = np.dot(βinv, s)

            for j in range(i+2):
                for l in range(i+2):
                    α[i+1] += np.linalg.multi_dot((
                        w[i+1, l],
                        t[j+l+1],
                        v[i+1, j],
                    ))

        return α, β, γ

    def build_block_tridiagonal(self, α, β, γ):
        h = np.block([[
            α[i] if i == j else
            β[j] if j == i-1 else
            γ[i] if i == j-1 else self.zero
            for j in range(len(α))]
            for i in range(len(α))]
        )

        return h

    def kernel(self):
        α, β, γ = self.get_blocks()
        orth = self.mat_sqrt(self.moments[0])
        h_tri = self.build_block_tridiagonal(α, β, γ)

        e, u = np.linalg.eig(h_tri)
        ul = np.dot(orth, u[:self.norb])
        ur = np.dot(np.linalg.inv(u)[:, :self.norb], orth).T.conj()

        for i in range(min(5, len(e))):
            self.log.info(" %s: %s", i, util.format_value(e[i]))
        self.log.info(" ...")

        return e, ul, ur

    @property
    def moments(self):
        return self.inp[0]

    @property
    def norb(self):
        return self.moments.shape[-1]

    @property
    def nmom(self):
        return self.moments.shape[0]

    @property
    def niter(self):
        return (self.nmom - 2) // 2

    @property
    def dtype(self):
        # Even for real asymmetric moments, the resulting excitations
        # are in general complex
        return np.complex128

    @property
    def complex(self):
        return np.iscomplexobj(self.dtype)

    @property
    def real(self):
        return not self.complex

    @property
    def zero(self):
        return np.zeros((self.norb, self.norb), dtype=self.dtype)


if __name__ == "__main__":
    np.set_printoptions(edgeitems=100, linewidth=1000, precision=4)

    norb = 5
    naux = 100

    e = np.random.random((naux,)) * 5.0
    v = np.random.random((norb, naux)) - 0.5
    m1 = np.einsum("xk,yk,nk->nxy", v, v, e[None]**np.arange(10)[:, None])
    solver = BlockLanczosSymmGF(m1)
    e, v = solver.kernel()
    m2 = np.einsum("xk,yk,nk->nxy", v, v, e[None]**np.arange(10)[:, None])
    for i, (a, b) in enumerate(zip(m1, m2)):
        a1 = a / np.max(np.abs(a))
        b1 = b / np.max(np.abs(b))
        assert np.allclose(a1, b1)

    e = np.random.random((naux,)) * 5.0
    v = np.random.random((norb, naux)) - 0.5 + (np.random.random((norb, naux)) - 0.5) * 1.0j
    m1 = np.einsum("xk,yk,nk->nxy", v, v.conj(), e[None]**np.arange(10)[:, None])
    solver = BlockLanczosSymmGF(m1)
    e, v = solver.kernel()
    m2 = np.einsum("xk,yk,nk->nxy", v, v.conj(), e[None]**np.arange(10)[:, None])
    for i, (a, b) in enumerate(zip(m1, m2)):
        a1 = a / np.max(np.abs(a))
        b1 = b / np.max(np.abs(b))
        assert np.allclose(a1, b1)

    e = np.random.random((naux,)) * 5.0
    v = np.random.random((norb, naux)) - 0.5
    u = np.random.random((norb, naux)) - 0.5
    m1 = np.einsum("xk,yk,nk->nxy", v, u, e[None]**np.arange(10)[:, None])
    solver = BlockLanczosNoSymmGF(m1)
    e, v, u = solver.kernel()
    m2 = np.einsum("xk,yk,nk->nxy", v, u.conj(), e[None]**np.arange(10)[:, None])
    for i, (a, b) in enumerate(zip(m1, m2)):
        a1 = a / np.max(np.abs(a))
        b1 = b / np.max(np.abs(b))
        assert np.allclose(a1, b1)

    e = np.random.random((naux,)) * 5.0
    v = np.random.random((norb, naux)) - 0.5 + (np.random.random((norb, naux)) - 0.5) * 1.0j
    u = np.random.random((norb, naux)) - 0.5 + (np.random.random((norb, naux)) - 0.5) * 1.0j
    m1 = np.einsum("xk,yk,nk->nxy", v, u.conj(), e[None]**np.arange(10)[:, None])
    solver = BlockLanczosNoSymmGF(m1)
    e, v, u = solver.kernel()
    m2 = np.einsum("xk,yk,nk->nxy", v, u.conj(), e[None]**np.arange(10)[:, None])
    for i, (a, b) in enumerate(zip(m1, m2)):
        a1 = a / np.max(np.abs(a))
        b1 = b / np.max(np.abs(b))
        assert np.allclose(a1, b1)
