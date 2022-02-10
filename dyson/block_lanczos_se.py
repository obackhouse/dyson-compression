'''
Block Lanczos via recursion of the moments of the spectral representation
of the self-energy.
'''

import numpy as np
import scipy.linalg
from dyson import misc, linalg


class C:
    ''' Class to contain the recursions.
    '''

    def __init__(self, nphys, force_orthogonality=True, dtype=np.float64, cache=True, allow_non_psd=False, eps=None):
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
        # Get a term
        i, n, j = key
        if i == 0 or j == 0:
            return self.zero
        elif i < j:
            return self[j,n,i].T.conj()
        else:
            return self._c[key]

    def __setitem__(self, key, val):
        # Set a term
        i, n, j = key
        if i < j:
            self[j,n,i] = val.T.conj()
        else:
            self._c[key] = val

    def check_sanity(self):
        for (i,n,j), c in self._c.items():
            if i == j:
                # Check Hermiticity
                if not np.allclose(c, c.T.conj(), rtol=1e-6, atol=1e-4):
                    raise ValueError('\nSanity check failed for C^{%d}_{%d,%d}' % (n,i,j))
            if i == 0 or j == 0:
                # Zeroth iteration Lanczos vector is zero
                if not np.allclose(c, self.zero):
                    raise ValueError('\nSanity check failed for C^{%d}_{%d,%d}' % (n,i,j))
            elif n == 0 and i == j:
                # Globally orthogonal Lanczos vectors for i==j
                if not np.allclose(c, self.eye):
                    raise ValueError('\nSanity check failed for C^{%d}_{%d,%d}' % (n,i,j))
            elif n == 0 and i != j:
                # Globally orthogonal Lanczos vectors for i!=j
                if not np.allclose(c, self.zero):
                    raise ValueError('\nSanity check failed for C^{%d}_{%d,%d}' % (n,i,j))

    def cb(self, i, n, j, env):
        # Get a cached term c[i,n,j] . b[j].T
        b = env['b']
        if self._cb_cache is None:
            return np.dot(self[i,n,j], b[j].T.conj())
        if (i,n,j) not in self._cb_cache:
            self._cb_cache[i,n,j] = np.dot(self[i,n,j], b[j].T.conj())
        return self._cb_cache[i,n,j]

    def mc(self, i, n, j, env):
        # Get a cached term c[i,1,i] . c[i,n,j]
        m = env['m']
        if self._mc_cache is None:
            return np.dot(self[i,1,i], self[i,n,j])
        if (i,n,j) not in self._mc_cache:
            self._mc_cache[i,n,j] = np.dot(self[i,1,i], self[i,n,j])
        return self._mc_cache[i,n,j]

    def build_initial(self, n, env):
        # c[1,n,1] <- None
        binv, t = env['binv'], env['t']

        tmp = np.dot(np.dot(binv.T.conj(), t[n]), binv)

        self[1,n,1] = tmp

        return self

    def bump_single(self, i, n, env):
        # [i+1,n,i] <- [i,1,i] + [i,n,i] + [i,n+1,i] + [i-1,n,i]
        b, binv = env['b'], env['binv']

        if n == 0 and self.force_orthogonality:
            self[i+1,n,i] = self.zero.copy()
            return self

        tmp  = self[i,n+1,i].copy()
        tmp -= self.cb(i,n,i-1, env).T.conj()
        tmp -= self.mc(i,n,i, env)

        self[i+1,n,i] = np.dot(binv.T.conj(), tmp)

        return self

    def bump_both(self, i, n, env):
        # [i+1,n,i+1] <- [i,1,i] + [i,n,i] + [i,n+1,i] + [i,n+2,i] + [i,n,i-1] + [i,n+1,i-1] + [i-1,n,i-1]
        b, binv = env['b'], env['binv']

        if n == 0 and self.force_orthogonality:
            self[i+1,n,i+1] = self.eye.copy()
            return self

        tmp  = self[i,n+2,i].copy()
        tmp -= linalg.hermi_sum(self.cb(i,n+1,i-1, env))
        tmp -= linalg.hermi_sum(self.mc(i,n+1,i, env))
        tmp += linalg.hermi_sum(np.dot(self[i,1,i], self.cb(i,n,i-1, env)))
        tmp += np.dot(b[i-1], self.cb(i-1,n,i-1, env))
        tmp += np.dot(self.mc(i,n,i, env), self[i,1,i].T.conj())

        self[i+1,n,i+1] = np.dot(np.dot(binv.T.conj(), tmp), binv)

        return self

    def compute_b(self, i, env):
        # b[i] <- [i,2,i] + c[i,1,i-1] + c[i,1,i]
        b, t = env['b'], env['t']

        if i == 0:
            b2 = t[0]
        else:
            b2  = self[i,2,i].copy()
            b2 -= linalg.hermi_sum(self.cb(i,1,i-1, env))
            b2 -= np.dot(self[i,1,i], self[i,1,i].T.conj())
            if i > 1:
                b2 += np.dot(b[i-1], b[i-1].T.conj())

        b[i], binv = sqrt_and_inv(b2, allow_non_psd=self.allow_non_psd, eps=self.eps)

        return b, binv


def sqrt_and_inv(x, allow_non_psd=False, eps=None):
    ''' Finds the square root (either via eigenvalues or Cholesky
        decomposition) of x and also returns the inverse of the result.
    '''

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
    ''' Block Lanczos algorithm using recursion of the moments of the
        spectral representation of the self-energy. Performs nmom+1 
        iterations of the block Lanczos algorithm. Input moments t
        must index the first 2*nmom+2 moments.
    '''

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
        m[i] = c[i,1,i]

    return m, b


def kernel(t_occ, t_vir, nmom, debug=False, **kwargs):
    ''' Kernel function for the block Lanczos method using the moments
        of the spectral representation of the self-energy. Returns the 
        reduced spectral representation.
    '''

    if t_occ is not None:
        nphys = t_occ[0].shape[0]
    if t_vir is not None:
        nphys = t_vir[0].shape[0]

    e = np.empty((0))
    v = np.empty((nphys, 0))

    if t_occ is not None:
        m, b = block_lanczos(t_occ, nmom, debug=debug, **kwargs)
        h_tri = linalg.build_block_tridiagonal(m, b)

        e_occ, v_occ = np.linalg.eigh(h_tri[nphys:,nphys:])
        v_occ = np.dot(b[0].T.conj(), v_occ[:nphys])

        e = np.concatenate([e, e_occ], axis=0)
        v = np.concatenate([v, v_occ], axis=1)

    if t_vir is not None:
        m, b = block_lanczos(t_vir, nmom, debug=debug, **kwargs)
        h_tri = linalg.build_block_tridiagonal(m, b)

        e_vir, v_vir = np.linalg.eigh(h_tri[nphys:,nphys:])
        v_vir = np.dot(b[0].T.conj(), v_vir[:nphys])

        e = np.concatenate([e, e_vir], axis=0)
        v = np.concatenate([v, v_vir], axis=1)

    return e, v


if __name__ == '__main__':
    pass
