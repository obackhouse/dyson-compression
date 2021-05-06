'''
Linear algebra tools and extensions to numpy
'''

import numpy as np
from dyson import misc


def moments(e, v, n, blksize=1024):
    ''' Get the first n (or within range n) moments of the spectral
        distribution described by poles e, with transition moments v.
    '''

    squeeze = isinstance(n, int)
    if squeeze:
        n = [n,]

    nphys, naux = v.shape
    dtype = np.result_type(e.dtype, v.dtype)
    
    assert e.size == naux

    t = np.zeros((len(n), nphys, nphys), dtype=dtype)

    for p0, p1 in misc.prange(0, naux, blksize):
        en = e[p0:p1][None] ** np.array(n)[:,None]
        t[p0:p1] = np.einsum('xk,yk,nk->nxy', v, v, en)

    if squeeze:
        t = t.squeeze()

    return t


def build_block_tridiagonal(m, b):
    ''' Build a block tridiagonal matrix from a list of on- (m) and off-
        diagonal (b) blocks of matrices.
    '''

    nphys = m[0].shape[0]
    dtype = np.result_type(*([x.dtype for x in m] + [x.dtype for x in b]))

    assert all([x.shape == (nphys, nphys) for x in m])
    assert all([x.shape == (nphys, nphys) for x in b])
    assert len(m) == len(b)+1

    zero = np.zeros((nphys, nphys), dtype=dtype)

    h = np.block([[m[i]          if i == j   else
                   b[j]          if j == i-1 else
                   b[i].T.conj() if i == j-1 else
                   zero
                   for j in range(len(m))]
                   for i in range(len(m))])

    return h


def build_spectral_matrix(phys, e, v):
    ''' Builds a matrix representing the spectral representation with
        coupling to a physical block.
    '''

    return np.block([[phys, v], [v.T.conj(), np.diag(e)]])


def dot_spectral_matrix(phys, e, v, r, out=None):
    ''' Dot product of the result of build_spectral_matrix with a 
        vector r.
    '''

    nphys = phys.shape[0]
    naux = e.size
    nqmo = nphys + naux

    r = np.asarray(r)
    input_shape = r.shape
    r = r.reshape((nphys+naux, -1))
    dtype = np.result_type(v.dtype, r.dtype)

    sp = slice(None, nphys)
    sa = slice(nphys, None)

    if out is None:
        out = np.zeros(r.shape, dtype=dtype)
    out = out.reshape(r.shape)

    out[sp]  = np.dot(phys, r[sp])
    out[sp] += np.dot(v, r[sa])

    out[sa]  = np.dot(r[sp].conj().T, v).conj().T
    out[sa] += e[:,None] * r[sa]

    out = out.reshape(input_shape)

    return out


def is_posdef(x, tol=0):
    ''' Check that a matrix is positive definite, use tol to change how
        small numbers may be considered zero.
    '''

    w = np.linalg.eigvalsh(x)

    return np.all(w.real > tol) and np.all(w.imag < tol)


def force_posdef(x, tol=0, maxiter=1000):
    ''' Iteratively adjust a Hermitian matrix to find the closest positive 
        definite matrix.

        From https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite/43244194
    '''

    spacing = np.spacing(np.linalg.norm(x))
    mineig = np.min(np.real(np.linalg.eigvals(x)))
    xh = x.copy()
    i = np.eye(x.shape[0])
    k = 1

    while mineig < tol:
        mineig = np.min(np.real(np.linalg.eigvals(xh)))
        xh += i * (-mineig * k**2 + spacing)
        k += 1
        if k == maxiter:
            print('maxiter reached in force_posdef')
            break

    return xh


def hermi_sum(x):
    ''' x + x^\dagger
    '''

    return x + x.T.conj().copy()


def power(x, n):
    ''' Raise a matrix x to the power n.
    '''

    if n == 1: return x
    if n == 2: return np.dot(x, x)
    if n == 3: return np.dot(np.dot(x, x), x)
    if n == -1: return np.linalg.inv(x)

    w, v = np.linalg.eigh(x)

    if np.any(w < 0) and n < 0:
        w = w.astype(np.complex128)

    x_out = np.dot(v * w[None]**n, v.T.conj())

    return x_out


if __name__ == '__main__':
    pass
