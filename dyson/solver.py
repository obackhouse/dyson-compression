'''
Uses the AGF2 solver from PySCF along with user-inputted expressions
for the occupied and virtual self-energies or Green's functions to
perform self-consistent calculations.
'''

import numpy as np
import time
from dyson import misc, linalg, kernel_gf, kernel_se

try:
    from pyscf import agf2, ao2mo, lib
except:
    pass


def get_solver(rhf, nmom, occupied_se=None, virtual_se=None, occupied_gf=None, virtual_gf=None, solver_kwargs={}):
    if getattr(rhf, 'with_df', None) is None:
        cls = agf2.ragf2.RAGF2
    else:
        cls = agf2.dfragf2.DFRAGF2

    assert occupied_se or occupied_gf

    class RHFSolver(cls):
        def build_se(self, eri=None, gf=None, **kwargs):
            if eri is None: eri = self.ao2mo()
            if gf is None: gf = self.gf
            if gf is None: gf = self.init_gf()

            cput0 = (time.clock(), time.time())
            log = lib.logger.Logger(self.stdout, self.verbose)

            nmoms = range(2*nmom[1]+2)
            kernel_kwargs = {
                    'nmom_lanczos': nmom[1], 
                    'nmom_projection': nmom[0],
                    'phys': self.get_fock(eri, gf=agf2.aux.combine(gf_occ, gf_vir)) \
                            if nmom[0] is not None else None,
            }

            gf_occ = gf.get_occupied()
            gf_vir = gf.get_virtual()

            if gf_occ.naux == 0 or gf_vir.naux == 0:
                logger.warn(self, 'Attempting to build a self-energy with '
                                  'no (i,j,a) or (a,b,i) configurations.')

            use_se = occupied_se is not None
            ph_symm = virtual_se is None if use_se else virtual_gf is None

            if use_se:
                t_occ = occupied_se(self, eri, gf_occ, gf_vir, nmoms, **kwargs)
                if ph_symm:
                    t_vir = occupied_se(self, eri, gf_vir, gf_occ, nmoms, **kwargs)
                else:
                    t_vir = virtual_se(self, eri, gf_occ, gf_vir, nmoms, **kwargs)

                e, v = kernel_se(t_occ, t_vir, **kernel_kwargs)

            else:
                t_occ = occupied_gf(self, eri, gf_occ, gf_vir, nmoms, **kwargs)
                if ph_symm:
                    t_vir = occupied_gf(self, eri, gf_vir, gf_occ, nmoms, **kwargs)
                else:
                    t_vir = virtual_gf(self, eri, gf_occ, gf_vir, nmoms, **kwargs)

                e, v = kernel_gf(t_occ, t_vir, **kernel_kwargs)

            se = agf2.SelfEnergy(e, v, chempot=gf.chempot)
            se.remove_uncoupled(tol=self.weight_tol)

            if not (self.frozen is None or self.frozen == 0):
                mask = get_frozen_mask(self)
                coupling = np.zeros((nmo, se.naux))
                coupling[mask] = se.coupling
                se = aux.SelfEnergy(se.energy, coupling, chempot=se.chempot)

            log.timer('se part', *cput0)

            return se

    solver = RHFSolver(rhf, **solver_kwargs)

    return solver


def mp2_occupied_se(solver, eri, gf_occ, gf_vir, nmoms, **kwargs):
    log = lib.logger.Logger(solver.stdout, solver.verbose)

    coeffs = (np.eye(solver.nmo), gf_occ.coupling, gf_occ.coupling, gf_vir.coupling)
    xija = ao2mo.incore.general(eri.eri, coeffs, compact=False, verbose=log)
    xija = xija.reshape([x.shape[1] for x in coeffs])

    va = xija.reshape(gf_occ.nphys, -1)
    vb = (2*xija-xija.swapaxes(1,2)).reshape(gf_occ.nphys, -1)
    e = lib.direct_sum('i+j-a->ija', gf_occ.energy, gf_occ.energy, gf_vir.energy)
    e = e.ravel()

    t = lib.einsum('xk,yk,nk->nxy', va, vb, e[None]**np.array(nmoms)[:,None])

    return t


def adc2_occupied_se(solver, eri, gf_occ, gf_vir, nmoms, **kwargs):
    # Equivalent to non-Dyson MP2

    log = lib.logger.Logger(solver.stdout, solver.verbose)

    coeffs = (gf_occ.coupling, gf_occ.coupling, gf_occ.coupling, gf_vir.coupling)
    ijka = ao2mo.incore.general(eri.eri, coeffs, compact=False, verbose=log)
    ijka = ijka.reshape([x.shape[1] for x in coeffs])

    va = ijka.reshape(gf_occ.nphys, -1)
    vb = (2*ijka-ijka.swapaxes(1,2)).reshape(gf_occ.nphys, -1)
    e = lib.direct_sum('j+k-a->jka', gf_occ.energy, gf_occ.energy, gf_vir.energy)
    e = e.ravel()

    t = lib.einsum('xk,yk,nk->nxy', va, vb, e[None]**np.array(nmoms)[:,None])

    return t


if __name__ == '__main__':
    from pyscf import gto, scf
    mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz', verbose=4)
    rhf = scf.RHF(mol).run(conv_tol=1e-12)

    dyson_solver = get_solver(rhf, (None,0), occupied_se=mp2_occupied_se)
    dyson_solver.run()
