"""Block Lanczos via recursion of the moments of the spectral
representation of the self-energy.
"""

import numpy as np
import scipy.linalg

from dyson import linalg


class C:
    """Class to contain the recursions.

    Arguments
    ---------
    nphys: int
        Number of physical degrees of freedom.
    force_orthogonality: bool, optional
        If True, force orthogonality in the vectors composing the
        recursion terms (default value is True).
    dtype: numpy.dtype, optional
        Data-type of the moments (default value is numpy.float64).
    cache: bool, optional
        If True, cache terms (default value is True).
    allow_non_psd: bool, optional
        Allow non-positively-defined moments (default value is False).
    eps: bool, optional
        Tolerance to consider eigenvalues zero (default value is
        machine precision for the given dtype).
    """

    def __init__(
            self, nphys,
            force_orthogonality=True,
            dtype=np.float64,
            cache=True,
            allow_non_psd=False,
            eps=None,
    ):
        self._c = {}
        self._cb_cache = {} if cache else None
        self._mc_cache = {} if cache else None

        self.zero = np.zeros((nphys, nphys))
        self.eye = np.eye(nphys)

        self.force_orthogonality = force_orthogonality
        self.allow_non_psd = allow_non_psd
        self.eps = None
        self.dtype = dtype


    def __getitem__(self, key):
        """Get a recursion term.

        Arguments
        ---------
        key: tuple of int (3)
            Indices (i,n,j) of term to return.

        Returns
        -------
        term: numpy.ndarray (nphys, nphys)
            (i,n,j)th recursion term.
        """

        i, n, j = key
        if i == 0 or j == 0:
            return self.zero
        elif i < j:
            return self[j, n, i].T.conj()
        else:
            return self._c[key]


    def __setitem__(self, key, val):
        """Set a recursion term.

        Arguments
        ---------
        key: tuple of int (3)
            Order, index of the term to set.
            Indices (i,n,j) of term to return.
        val: numpy.ndarray (nphys, nphys)
            (i,n,j)th recursion term.
        """

        i, n, j = key
        if i < j:
            self[j, n, i] = val.T.conj()
        else:
            self._c[key] = val


    def check_sanity(self):
        """Check the sanity of the recursion terms.
        """

        for (i, n, j), c in self._c.items():
            if i == j:
                # Check Hermiticity
                if not np.allclose(c, c.T.conj(), rtol=1e-6, atol=1e-4):
                    raise ValueError('\nSanity check failed for C^{%d}_{%d,%d}' % (n, i, j))

            if i == 0 or j == 0:
                # Zeroth iteration Lanczos vector is zero
                if not np.allclose(c, self.zero):
                    raise ValueError('\nSanity check failed for C^{%d}_{%d,%d}' % (n, i, j))

            elif n == 0 and i == j:
                # Globally orthogonal Lanczos vectors for i==j
                if not np.allclose(c, self.eye):
                    raise ValueError('\nSanity check failed for C^{%d}_{%d,%d}' % (n, i, j))

            elif n == 0 and i != j:
                # Globally orthogonal Lanczos vectors for i!=j
                if not np.allclose(c, self.zero):
                    raise ValueError('\nSanity check failed for C^{%d}_{%d,%d}' % (n, i, j))


    def cb(self, i, n, j, env):
        """Get a cached term c[i,n,j] . b[j].T

        Arguments
        ---------
        i: int
            First index of recursion term.
        n: int
            Second index of recursion term.
        j: int
            Third index of recursion term.
        env: dict
            Dictionary containing b.

        Returns
        -------
        cb: numpy.ndarray (nphys, nphys)
            The product of the (i,n,j)th recursion term and the jth B
            term.
        """

        b = env['b']

        if self._cb_cache is None:
            return np.dot(self[i, n, j], b[j].T.conj())

        if (i, n, j) not in self._cb_cache:
            self._cb_cache[i, n, j] = np.dot(self[i, n, j], b[j].T.conj())

        return self._cb_cache[i, n, j]


    def mc(self, i, n, j, env):
        """Get a cached term c[i,1,i] . c[i,n,j]

        Arguments
        ---------
        i: int
            First index of recursion term.
        n: int
            Second index of recursion term.
        j: int
            Third index of recursion term.
        env: dict
            Dictionary containing m.

        Returns
        -------
        mc: numpy.ndarray (nphys, nphys)
            The product of the (i,1,i)th recursion term and the
            (i,n,j)th recursion terms.
        """

        if self._mc_cache is None:
            return np.dot(self[i, 1, i], self[i, n, j])

        if (i, n, j) not in self._mc_cache:
            self._mc_cache[i, n, j] = np.dot(self[i, 1, i], self[i, n, j])

        return self._mc_cache[i, n, j]


    def build_initial(self, n, env):
        """Build the orthogonalised moments.

        Arguments
        ---------
        n: int
            Order of moment to orthogonalise.
        env: dict
            Dictionary containing binv, t.
        """

        binv, t = env['binv'], env['t']

        self[1, n, 1] = np.linalg.multi_dot((binv.T.conj(), t[n], binv))

        return self


    def bump_single(self, i, n, env):
        """Compute the (i+1,n,i)th recursion term.

        Arguments
        ---------
        i: int
            First index of recursion term.
        n: int
            Second index of recursion term.
        env: dict
            Dictionary containing b, binv.
        """

        binv = env['binv']

        if n == 0 and self.force_orthogonality:
            self[i+1, n, i] = self.zero.copy()
            return self

        tmp  = self[i, n+1, i].copy()
        tmp -= self.cb(i, n, i-1, env).T.conj()
        tmp -= self.mc(i, n, i, env)

        self[i+1, n, i] = np.dot(binv.T.conj(), tmp)

        return self


    def bump_both(self, i, n, env):
        """Compute the (i+1,n,i+1)th recursion term.

        Arguments
        ---------
        i: int
            First index of recursion term.
        n: int
            Second index of recursion term.
        env: dict
            Dictionary containing b, binv.
        """

        b, binv = env['b'], env['binv']

        if n == 0 and self.force_orthogonality:
            self[i+1, n, i+1] = self.eye.copy()
            return self

        tmp  = self[i, n+2, i].copy()
        tmp -= linalg.hermi_sum(self.cb(i, n+1, i-1, env))
        tmp -= linalg.hermi_sum(self.mc(i, n+1, i, env))
        tmp += linalg.hermi_sum(np.dot(self[i, 1, i], self.cb(i, n, i-1, env)))
        tmp += np.dot(b[i-1], self.cb(i-1, n, i-1, env))
        tmp += np.dot(self.mc(i, n, i, env), self[i, 1, i].T.conj())

        self[i+1, n, i+1] = np.linalg.multi_dot((binv.T.conj(), tmp, binv))

        return self

    def compute_b(self, i, env):
        """Compute the ith B term, returning also its inverse.

        Arguments
        ---------
        i: int
            Index of term.
        env: dict
            Dictionary containing b, t.

        Returns
        -------
        b: numpy.ndarray (n, nphys, nphys)
            B matrix with the ith term set.
        binv: numpy.ndarray (nphys, nphys)
            Inverse of the ith B term.
        """

        b, t = env['b'], env['t']

        if i == 0:
            b2 = t[0]
        else:
            b2  = self[i, 2, i].copy()
            b2 -= linalg.hermi_sum(self.cb(i, 1, i-1, env))
            b2 -= np.dot(self[i, 1, i], self[i, 1, i].T.conj())
            if i > 1:
                b2 += np.dot(b[i-1], b[i-1].T.conj())

        b[i], binv = sqrt_and_inv(b2, allow_non_psd=self.allow_non_psd, eps=self.eps)

        return b, binv


def sqrt_and_inv(x, allow_non_psd=False, eps=None):
    """Returns a matrix raised to the power 1/2 and -1/2.

    Arguments
    ---------
    x: ndarray
        Matrix to decompose.
    allow_non_psd: bool, optional
        If True, allow matrix to be non-positively-defined (default
        value is False).
    eps: float, optional
        Tolerance to consider an eigenvalue zero (default value is
        machine precision for the given dtype).
    """

    if eps is None:
        eps = np.finfo(x.dtype).eps

    try:
        w, v = np.linalg.eigh(x)
    except np.linalg.LinAlgError:
        w, v = scipy.linalg.eigh(x)

    if allow_non_psd:
        mask = np.abs(w) > eps
    else:
        mask = w > eps

    w, v = w[mask], v[:, mask]

    if allow_non_psd and np.any(w < 0):
        w = w.astype(np.complex128)

    # This shouldn't happen but let's check anyway
    if np.any(w < eps) and not allow_non_psd:
        mask = w > eps
        w, v = w[mask], v[:, mask]

    bi = np.dot(v * w[None]**0.5, v.T.conj())
    binv = np.dot(v * w[None]**-0.5, v.T.conj())

    return bi, binv


def block_lanczos(t, nmom, debug=False, **kwargs):
    """Block Lanczos algorithm using recursion of the moments of the
    spectral representation of the self-energy. Performs nmom+1
    iterations of the block Lanczos algorithm. Input moments must
    index the first 2*nmom+2 moments.

    Arguments
    ---------
    t: numpy.ndarray (>=2*nmom+2, nphys, nphys)
        Moments to which consistency is achieved.
    nmom: int
        Number of iterations of block Lanczos to perform.

    Other kwargs are passed to the C object.

    Returns
    -------
    m: numpy.ndarray (nmom+1, nphys, nphys)
        On-diagonal blocks of the block tridiagonalised matrix.
    b: numpy.ndarray (nmom, nphys, nphys)
        Off-diagonal blocks of the block tridiagonalised matrix.
    """

    nphys = t[0].shape[0]
    dtype = t[0].dtype
    nblock = nmom+1

    m = np.zeros((nblock+1, nphys, nphys), dtype=dtype)
    b = np.zeros((nblock,   nphys, nphys), dtype=dtype)
    c = C(nphys, dtype=dtype, **kwargs)

    b, binv = c.compute_b(0, locals())

    for i in range(2*nmom+2):
        c = c.build_initial(i, locals())

    for i in range(1, nblock):
        b, binv = c.compute_b(i, locals())

        for n in range(2*(nblock-i)-1):
            c = c.bump_single(i, n, locals())
            c = c.bump_both(i, n, locals())
        c = c.bump_both(i, n+1, locals())

    if debug:
        c.check_sanity()

    for i in range(nblock+1):
        m[i] = c[i, 1, i]

    return m, b


def kernel(t_occ, t_vir, nmom, debug=False, **kwargs):
    """Kernel function for the block Lanczos method via the moments
    of the spectral representation of the self-energy.

    Arguments
    ---------
    t_occ: numpy.ndarray (>=2*nmom+2, nphys, nphys)
        Moments of the occupied self-energy.
    t_vir: numpy.ndarray (>=2*nmom+2, nphys, nphys)
        Moments of the virtual self-energy.
    nmom: int
        Number of iterations of block Lanczos to perform.

    Other kwargs are passed to the C object.

    Returns
    -------
    e: numpy.ndarray
        Energies of the compressed self-energy auxiliaries.
    v: numpy.ndarray
        Coupling of the compressed self-energy auxiliaries.
    """

    if t_occ is not None:
        nphys = t_occ[0].shape[0]
    if t_vir is not None:
        nphys = t_vir[0].shape[0]

    e = np.empty((0))
    v = np.empty((nphys, 0))

    if t_occ is not None:
        m, b = block_lanczos(t_occ, nmom, debug=debug, **kwargs)
        h_tri = linalg.build_block_tridiagonal(m, b)

        e_occ, v_occ = np.linalg.eigh(h_tri[nphys:, nphys:])
        v_occ = np.dot(b[0].T.conj(), v_occ[:nphys])

        e = np.concatenate([e, e_occ], axis=0)
        v = np.concatenate([v, v_occ], axis=1)

    if t_vir is not None:
        m, b = block_lanczos(t_vir, nmom, debug=debug, **kwargs)
        h_tri = linalg.build_block_tridiagonal(m, b)

        e_vir, v_vir = np.linalg.eigh(h_tri[nphys:, nphys:])
        v_vir = np.dot(b[0].T.conj(), v_vir[:nphys])

        e = np.concatenate([e, e_vir], axis=0)
        v = np.concatenate([v, v_vir], axis=1)

    return e, v


if __name__ == '__main__':
    pass
