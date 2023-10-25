from qutip import Qobj
import scipy.sparse as sp
import numpy as np

def cos_phi(N, phi_ext, m = 1):
    """
    Compute the cosine phi operator matrix in a complex sparse representation.

    The operator is calculated based on the given system size N and external phase factor phi_ext. 

    Parameters
    ----------
    N : int
        The size of the matrix to be created.
    phi_ext : float
        The external phase factor.
    m : int, optional
        The diagonal offset. Default is 1.

    Returns
    -------
    Qobj
        The cosine phi operator represented as a QuTiP Qobj with the CSR sparse matrix format.
    """
    diags = [np.exp(1j*phi_ext/2)*np.ones(N-m,dtype=int),np.exp(-1j*phi_ext/2)*np.ones(N-m,dtype=int)]
    T = sp.diags(diags,[m,-m],format='csr', dtype=complex)
    return Qobj(T, isherm=True)/2

def sin_phi(N, phi_ext, m = 1):
    """
    Compute the sine phi operator matrix in a complex sparse representation.

    The operator is calculated based on the given system size N and external phase factor phi_ext.

    Parameters
    ----------
    N : int
        The size of the matrix to be created.
    phi_ext : float
        The external phase factor.
    m : int, optional
        The diagonal offset. Default is 1.

    Returns
    -------
    Qobj
        The sine phi operator represented as a QuTiP Qobj with the CSR sparse matrix format.
    """
    diags = [np.exp(1j*phi_ext/2)*np.ones(N-m,dtype=int),-np.exp(-1j*phi_ext/2)*np.ones(N-m,dtype=int)]
    T = sp.diags(diags,[m,-m],format='csr', dtype=complex)
    return Qobj(T, isherm=True) /2/1j 