from qutip import Qobj
import scipy.sparse as sp
import numpy as np

import numpy as np

def destroy(dimension: int) -> np.ndarray:
    """
    Returns the annihilation (lowering) operator for a given dimension.

    Parameters
    ----------
    dimension : int
        Dimension of the Hilbert space.

    Returns
    -------
    np.ndarray
        The annihilation operator.
    """
    indices = np.arange(1, dimension)
    data = np.sqrt(indices)
    return np.diag(data, k=1)

def creation(dimension: int) -> np.ndarray:
    """
    Returns the creation (raising) operator for a given dimension.

    Parameters
    ----------
    dimension : int
        Dimension of the Hilbert space.

    Returns
    -------
    np.ndarray
        The creation operator.
    """
    return destroy(dimension).T.conj()

def sigma_x() -> np.ndarray:
    """
    Returns the Pauli-X (sigma_x) operator.

    Returns
    -------
    np.ndarray
        The Pauli-X operator.
    """
    return np.array([[0, 1], [1, 0]], dtype=complex)

def sigma_y() -> np.ndarray:
    """
    Returns the Pauli-Y (sigma_y) operator.

    Returns
    -------
    np.ndarray
        The Pauli-Y operator.
    """
    return np.array([[0, -1j], [1j, 0]], dtype=complex)

def sigma_z() -> np.ndarray:
    """
    Returns the Pauli-Z (sigma_z) operator.

    Returns
    -------
    np.ndarray
        The Pauli-Z operator.
    """
    return np.array([[1, 0], [0, -1]], dtype=complex)

###########

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