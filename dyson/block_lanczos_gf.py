'''
Block Lanczos via recursion of the moments of the spectral representation
of the Green's function.
'''

import numpy as np
from dyson import misc, linalg
from dyson.block_lanczos_se import sqrt_and_inv


class C:
    ''' Class to contain the recursions.
    '''

    def __init__(self, nphys, force_orthogonality=True, dtype=np.float64, cache=True):
        self._c = {}

        self.zero = np.zeros((nphys, nphys))
        self.eye = np.eye(nphys)
        self[0,0] = self.eye

        self.force_orthogonality = force_orthogonality
        self.eps = None
        self.dtype = dtype

    def __getitem__(self, key):
        # Get a term
        n, i = key
        if i < 0 or i > n or n < 0:
            return self.zero
        else:
            return self._c[n,i]

    def __setitem__(self, key, val):
        # Set a term
        n, i = key
        self._c[n,i] = val

    def check_sanity(self):
        pass #TODO

    def build_initial(self, n, env):
        orth, t, s = env['orth'], env['t'], env['s']

        s[n] = np.dot(np.dot(orth, t[n]), orth)

        return s

    def bump_single(self, i, j, env):
        m, b, binv = env['m'], env['b'], env['binv']

        tmp  = self[i,j-1].copy()
        tmp -= np.dot(self[i,j], m[i])
        tmp -= np.dot(self[i-1,j], b[i-1].conj().T)

        self[i+1,j] = np.dot(tmp, binv)

        return self

    def compute_m(self, i, env):
        m, s = env['m'], env['s']

        for j in range(i+2):
            for l in range(i+2):
                m[i+1] += np.dot(np.dot(self[i+1,l].conj().T, s[j+l+1]), self[i+1,j])

        return m

    def compute_b(self, i, env):
        m, b, s = env['m'], env['b'], env['s']

        b2 = self.zero.copy()
        for j in range(i+2):
            for l in range(i+1):
                b2 += np.dot(np.dot(self[i,l].T.conj(), s[j+l+1]), self[i,j-1])

        b2 -= np.dot(m[i], m[i].T.conj())
        if i > 0:
            b2 -= np.dot(b[i-1], b[i-1].conj().T).conj().T

        b[i], binv = sqrt_and_inv(b2, eps=self.eps)

        return b, binv


def block_lanczos(t, nmom, debug=False, **kwargs):
    ''' Block Lanczos algorithm using recursion of the moments of the
        spectral representation of the Green's function. Performs 
        nmom+1 iterations of the block Lanczos algorith. Input moments 
        t must index the first 2*nmom+2 moments.
    '''

    nphys = t[0].shape[0]
    dtype = t[0].dtype
    nblock = 2*nmom+1

    m = np.zeros((nmom+1, nphys, nphys), dtype=dtype)
    b = np.zeros((nmom,   nphys, nphys), dtype=dtype)
    s = np.zeros_like(t)
    c = C(nphys, dtype=dtype, **kwargs)

    orth = linalg.power(t[0], -0.5)

    for i in range(len(t)):
        s = c.build_initial(i, locals())

    m[0] = s[1]

    for i in range(nmom):
        b, binv = c.compute_b(i, locals())

        for j in range(i+2):
            c = c.bump_single(i, j, locals())

        m = c.compute_m(i, locals())

    if debug:
        c.check_sanity()

    return m, b


def kernel(t_occ, t_vir, nmom, debug=False, **kwargs):
    ''' Kernel function for the block Lanczos method via the moments 
        of the spectral representation of the Greens' function. 
        Returns the reduced spectral representation.
    '''

    nphys = t_occ[0].shape[0]

    m, b = block_lanczos(t_occ, nmom, debug=debug, **kwargs)
    h_tri = linalg.build_block_tridiagonal(m, b)

    e_occ, u_occ = np.linalg.eigh(h_tri)
    b = sqrt_and_inv(t_occ[0], eps=kwargs.get("eps", None))[0]
    u_occ = np.dot(b.T.conj(), u_occ[:nphys])

    m, b = block_lanczos(t_vir, nmom, debug=debug, **kwargs)
    h_tri = linalg.build_block_tridiagonal(m, b)

    e_vir, u_vir = np.linalg.eigh(h_tri)
    b = sqrt_and_inv(t_vir[0], eps=kwargs.get("eps", None))[0]
    u_vir = np.dot(b.T.conj(), u_vir[:nphys])

    e = np.concatenate([e_occ, e_vir], axis=0)
    u = np.concatenate([u_occ, u_vir], axis=1).T

    norm = np.linalg.norm(u, axis=0, keepdims=True)
    norm[np.absolute(norm) == 0] = 1e-20
    u /= norm
    w, v = np.linalg.eigh(np.eye(e.size) - np.dot(u, u.T.conj()))
    u = np.block([u, w[None] * v[:, abs(w) > 1e-20]])

    h = np.dot(u.T.conj() * e[None], u)
    e, v = np.linalg.eigh(h[nphys:,nphys:])
    naux = e.size
    v = np.block([[np.eye(nphys), np.zeros((nphys, naux))],
                  [np.zeros((naux, nphys)), v]])

    h = np.dot(np.dot(v.T.conj(), h), v)
    e = np.diag(h[nphys:,nphys:])
    v = h[:nphys,nphys:]

    return e, v


if __name__ == '__main__':
    pass
