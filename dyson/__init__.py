'''
Methods to compress spectral representations of self-energies, which
are linked to Green's functions via a Dyson equation.

Only the moments of the representations are required, and the module
supports similar functionality starting from the moments of the 
self-energy or of the Green's function.
'''

from dyson import misc, linalg
from dyson import block_lanczos_se, block_lanczos_gf, project_gf


def kernel_se(t_occ, t_vir, nmom_lanczos, nmom_projection=None, phys=None, chempot=0.0, debug=False, lanczos_kwargs={}, projection_kwargs={}):
    ''' Kernel function for the module starting from moments of the 
        spectral representation of the self-energy. 
        
        Resulting representation will be consistent in separate occupied 
        and virtual self-energy moments up to order: 

            2 * nmom_lanczos + 1

        If nmom_projection is passed, then only the central (occupied +
        virtual) will be consistent, up to order:

            2 * min(nmom_lanczos, nmom_projection) + 1

    Arguments
    ---------
    t_occ : (2*nmom_lanczos+2, nphys, nphys) ndarray
        Moments of the occupied (hole) self-energy. Must give up to
        order 2*nmom_lanczos+1 moments.
    t_vir : (2*nmom_lanczos+2, nphys, nphys) ndarray
        Moments of the virtual (particle) self-energy. Must give up
        to order 2*nmom_lanczos+1 moments.
    nmom_lanczos: int or 2-tuple of ints
        Order of consistency in the block Lanczos recursion, can be
        passed as a 2-tuple to give separate (occupied, virtual) 
        self-energy values.
    nmom_projection: int, optional
        Order of consistency in the projection of the Green's function,
        if None then not performed. If passed, then phys must also be
        passed.
    phys : ndarray, optional
        Matrix representing the physical space. Must be the same shape
        as the moments, and required only if nmom_projection != None.
    chempot : float, optional
        Adjustable chemical potential for the projection of the Green's
        function.
    debug : bool, optional
        Enable debugging flags, may severely impact performance.
    lanczos_kwargs: dict, optional
        Additional keyword arguments for the block Lanczos function.
    projection_kwargs: dict, optional
        Additional keyword arguments for the projection function.

    Returns
    -------
    e : ndarray
        Pole positions (energies) of the reduced spectral representation
        of the self-energy.
    v : ndarray
        Pole strengths (couplings to the physical space) of the reduced
        spectral representation of the self-energy.
    '''
        
    if min(len(t_occ), len(t_vir)) < (2*nmom_lanczos+2):
        raise ValueError('Not enough moments passed for nmom_lanczos = %d. '
                         '2*nmom_lanczos+2 = %d are required, representing '
                         'order 0 through 2*nmom+lanczos+1'
                         % (nmom_lanczos, 2*nmom_lanczos+2))

    if nmom_projection is not None and phys is None:
        raise ValueError('phys must be passed if nmom_projection != None')

    e, v = block_lanczos_se.kernel(t_occ, t_vir, nmom_lanczos, 
                                   debug=debug, **lanczos_kwargs)

    if nmom_projection is not None:
        e, v = project_gf.kernel(phys, e, v, nmom_projection, debug=debug, 
                                 chempot=chempot, **projection_kwargs)

    return e, v


def kernel_gf(t_occ, t_vir, nmom_lanczos, nmom_projection=None, phys=None, chempot=0.0, debug=False, lanczos_kwargs={}, projection_kwargs={}):
    ''' Kernel function for the module starting from moments of the 
        spectral representation of the Green's function. 
        
        Resulting representation will be consistent in separate occupied 
        and virtual Green's function moments up to order: 

            2 * nmom_lanczos + 1

        If nmom_projection is passed, up to order:

            2 * min(nmom_lanczos, nmom_projection) + 1

    Arguments
    ---------
    t_occ : (2*nmom_lanczos+2, nphys, nphys) ndarray
        Moments of the occupied (hole) Green's function. Must give up 
        to order 2*nmom_lanczos+1 moments.
    t_vir : (2*nmom_lanczos+2, nphys, nphys) ndarray
        Moments of the virtual (particle) Green's function. Must give 
        up to order 2*nmom_lanczos+1 moments.
    nmom_lanczos: int or 2-tuple of ints
        Order of consistency in the block Lanczos recursion, can be
        passed as a 2-tuple to give separate (occupied, virtual) 
        self-energy values.
    nmom_projection: int, optional
        Order of consistency in the projection of the Green's function,
        if None then not performed. If passed, then phys must also be
        passed.
    phys : ndarray, optional
        Matrix representing the physical space. Must be the same shape
        as the moments, and required only if nmom_projection != None.
    chempot : float, optional
        Adjustable chemical potential for the projection of the Green's
        function.
    debug : bool, optional
        Enable debugging flags, may severely impact performance.
    lanczos_kwargs: dict, optional
        Additional keyword arguments for the block Lanczos function.
    projection_kwargs: dict, optional
        Additional keyword arguments for the projection function.

    Returns
    -------
    e : ndarray
        Pole positions (energies) of the reduced spectral representation
        of the self-energy.
    v : ndarray
        Pole strengths (couplings to the physical space) of the reduced
        spectral representation of the self-energy.
    '''

    if min(len(t_occ), len(t_vir)) < (2*nmom_lanczos+2):
        raise ValueError('Not enough moments passed for nmom_lanczos = %d. '
                         '2*nmom_lanczos+2 = %d are required, representing '
                         'order 0 through 2*nmom+lanczos+1'
                         % (nmom_lanczos, 2*nmom_lanczos+2))

    if nmom_projection is not None and phys is None:
        raise ValueError('phys must be passed if nmom_projection != None')

    e, v = block_lanczos_gf.kernel(t_occ, t_vir, nmom_lanczos, 
                                   debug=debug, **lanczos_kwargs)

    if nmom_projection is not None:
        e, v = project_gf.kernel(phys, e, v, nmom_projection, debug=debug, 
                                 chempot=chempot, **projection_kwargs)

    return e, v
