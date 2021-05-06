'''
Projection via the moments of the spectral representation of the
Green's function.
'''

import numpy as np
from dyson import misc, linalg


def build_projector(phys, e_aux, v_aux, nmom, debug=False, tol=1e-12, blksize=256, chempot=0.0):
    ''' Builds the vectors which project the auxiliary space into a
        compressed one with consistency in the separate particle and
        hole Greens' function moments up to order 2*nmom+1.
    '''
    
    nphys, naux = v_aux.shape
    h = linalg.build_spectral_matrix(phys, e_aux, v_aux)
    w, c = np.linalg.eigh(h)

    def _part(w, c, s):
        # Up to order nmom+1 gives consistency up to order 2*nmom+1.

        p = np.zeros((naux, nphys, (nmom+2)), dtype=h.dtype)

        for p0, p1 in misc.prange(0, np.sum(s), blksize):
            e0 = w[s][p0:p1][None] ** np.arange(nmom+2)[:,None]
            c0 = c[:,s][:,p0:p1]
            p += np.einsum('xi,pi,ni->xpn', c0[nphys:], c0[:nphys], e0)

        return p.reshape(naux, -1)

    p = np.hstack((_part(w, c, w < chempot),
                   _part(w, c, w >= chempot)))

    norm = np.linalg.norm(p, axis=0, keepdims=True)
    norm[np.absolute(norm) == 0] = 1e-20

    p /= norm
    w, p = np.linalg.eigh(np.dot(p, p.T.conj()))
    p = p[:, w > tol]
    nvec = p.shape[1]

    p = np.block([[np.eye(nphys), np.zeros((nphys, nvec))],
                  [np.zeros((naux, nphys)), p]])

    return p


def kernel(phys, e, v, nmom, debug=False, chempot=0.0, tol=1e-12, blksize=256):
    ''' Kernel function for the projection method via the moments
        of the spectral representation of the Green's function.
        returns the reduced spectral representation.
    '''

    nphys = phys.shape[0]

    p = build_projector(phys, e, v, nmom, debug=debug, chempot=chempot, 
                        tol=tol, blksize=blksize)

    h = np.dot(p.T.conj(), linalg.dot_spectral_matrix(phys, e, v, p))
    del p

    e, v = np.linalg.eigh(h[nphys:,nphys:])
    v = np.dot(h[:nphys,nphys:], v)

    return e, v


if __name__ == '__main__':
    pass
