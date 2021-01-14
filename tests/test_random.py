import unittest
import numpy as np
from dyson import *


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        nphys = 20
        naux = 1000
        f_scale = 20
        e_scale = 50
        v_scale = 1.0
        rand = lambda shape, scale: (np.random.random(shape) * 2 - 1) * scale
        self.f = np.diag(rand((nphys,), f_scale))
        self.e = rand((naux,), e_scale)
        self.v = rand((nphys, naux), v_scale)
        self.w, self.c = np.linalg.eigh(linalg.build_spectral_matrix(self.f, self.e, self.v))
        self.c = self.c[:nphys]

    @classmethod
    def tearDownClass(self):
        del self.f, self.e, self.v, self.w, self.c

    def allclose(self, a, b):
        # Runs np.allclose on the arrays once normalised
        a_norm = a / np.max(a)
        b_norm = b / np.max(b)
        return np.allclose(a_norm, b_norm)

    def get_moments(self, e1, v1, nmax):
        return linalg.moments(e1, v1, range(nmax+1))

    def get_occ_moments(self, e1, v1, nmax):
        e1_occ, v1_occ = e1[e1 < 0], v1[:, e1 < 0]
        return self.get_moments(e1_occ, v1_occ, nmax)

    def get_vir_moments(self, e1, v1, nmax):
        e1_vir, v1_vir = e1[e1 >= 0], v1[:, e1 >= 0]
        return self.get_moments(e1_vir, v1_vir, nmax)

    def get_moments_errors(self, e1, v1, e2, v2, nmax):
        t1 = self.get_moments(e1, v1, nmax)
        t2 = self.get_moments(e2, v2, nmax)
        return np.array([self.allclose(x, y) for x,y in zip(t1, t2)])

    def get_occ_moments_errors(self, e1, v1, e2, v2, nmax):
        t1 = self.get_occ_moments(e1, v1, nmax)
        t2 = self.get_occ_moments(e2, v2, nmax)
        return np.array([self.allclose(x, y) for x,y in zip(t1, t2)])
        
    def get_vir_moments_errors(self, e1, v1, e2, v2, nmax):
        t1 = self.get_vir_moments(e1, v1, nmax)
        t2 = self.get_vir_moments(e2, v2, nmax)
        return np.array([self.allclose(x, y) for x,y in zip(t1, t2)])

    def _test_se_moments(self, nmom_lanczos, nmom_projection=None):
        t_occ = self.get_occ_moments(self.e, self.v, 2*nmom_lanczos+1)
        t_vir = self.get_vir_moments(self.e, self.v, 2*nmom_lanczos+1)
        e, v = kernel_se(t_occ, t_vir, nmom_lanczos, nmom_projection, phys=self.f)

        err_se_all = self.get_moments_errors(e, v, self.e, self.v, 2*nmom_lanczos+2)
        err_se_occ = self.get_occ_moments_errors(e, v, self.e, self.v, 2*nmom_lanczos+2)
        err_se_vir = self.get_occ_moments_errors(e, v, self.e, self.v, 2*nmom_lanczos+2)

        return err_se_all, err_se_occ, err_se_vir

    def _test_gf_moments(self, nmom_lanczos, nmom_projection=None):
        t_occ = self.get_occ_moments(self.w, self.c, 2*nmom_lanczos+1)
        t_vir = self.get_vir_moments(self.w, self.c, 2*nmom_lanczos+1)
        e, v = kernel_gf(t_occ, t_vir, nmom_lanczos, nmom_projection, phys=self.f)
        w, c = np.linalg.eigh(linalg.build_spectral_matrix(self.f, e, v))
        c = c[:self.f.shape[0]]

        err_gf_all = self.get_moments_errors(w, c, self.w, self.c, 2*nmom_lanczos+2)
        err_gf_occ = self.get_occ_moments_errors(w, c, self.w, self.c, 2*nmom_lanczos+2)
        err_gf_vir = self.get_vir_moments_errors(w, c, self.w, self.c, 2*nmom_lanczos+2)

        return err_gf_all, err_gf_occ, err_gf_vir

    def test_block_lanczos_se(self):
        for nmom in range(9):
            a, o, v = self._test_se_moments(nmom)
            self.assertGreaterEqual(sum(a[:2*nmom+2]), 2*nmom+2)
            self.assertGreaterEqual(sum(o[:2*nmom+2]), 2*nmom+2)
            self.assertGreaterEqual(sum(v[:2*nmom+2]), 2*nmom+2)

    def test_block_lanczos_se_plus_projection(self):
        for nmom_lanczos in range(1, 9, 2):
            for nmom_projection in range(1, nmom_lanczos-2, 2):
                a, o, v = self._test_se_moments(nmom_lanczos, nmom_projection)
                nmom = min(nmom_lanczos, nmom_projection)
                self.assertGreaterEqual(sum(a[:2*nmom+2]), 2*nmom+2)

    def test_block_lanczos_gf(self):
        for nmom in range(9):
            a, o, v = self._test_gf_moments(nmom)
            self.assertGreaterEqual(sum(a[:2*nmom+2]), 2*nmom+2)
            self.assertGreaterEqual(sum(o[:2*nmom+2]), 2*nmom+2)
            self.assertGreaterEqual(sum(v[:2*nmom+2]), 2*nmom+2)

    def test_block_lanczos_gf_plus_projection(self):
        for nmom_lanczos in range(1, 9, 2):
            for nmom_projection in range(1, nmom_lanczos-2, 2):
                a, o, v = self._test_gf_moments(nmom_lanczos, nmom_projection)
                nmom = min(nmom_lanczos, nmom_projection)
                self.assertGreaterEqual(sum(a[:2*nmom+2]), 2*nmom+2)
                self.assertGreaterEqual(sum(o[:2*nmom+2]), 2*nmom+2)
                self.assertGreaterEqual(sum(v[:2*nmom+2]), 2*nmom+2)


if __name__ == '__main__':
    unittest.main()
