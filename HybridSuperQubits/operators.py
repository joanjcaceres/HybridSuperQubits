from qutip import Qobj
import scipy.sparse as sp
import numpy as np

def state_to_density_matrix(state_vector: np.ndarray) -> np.ndarray:
    """
    Convert a state vector to a density matrix.

    Parameters
    ----------
    state_vector : numpy.ndarray
        The state vector to be converted.

    Returns
    -------
    numpy.ndarray
        The density matrix.
    """
    return np.outer(state_vector, state_vector.conj())

def ptrace(rho: np.ndarray, dims: tuple, subsys: int) -> np.ndarray:
    """
    Compute the partial trace of a density matrix over a specified subsystem.

    The partial trace is a method used in quantum mechanics to obtain the reduced 
    density matrix of a subsystem by tracing out the degrees of freedom of the 
    other subsystem.

    Parameters
    ----------
    rho : numpy.ndarray
        The density matrix to be traced. It should be a square matrix of shape 
        (dimA * dimB, dimA * dimB).
    dims : tuple
        A tuple (dimA, dimB) specifying the dimensions of the subsystems A and B.
    subsys : int
        The subsystem to trace out. Use 0 to trace out subsystem A and 1 to trace 
        out subsystem B.

    Returns
    -------
    numpy.ndarray
        The reduced density matrix after tracing out the specified subsystem.

    Raises
    ------
    ValueError
        If the subsys parameter is not 0 or 1.
    """
    dimA, dimB = dims
    rho_reshaped = rho.reshape([dimA, dimB, dimA, dimB])
    if subsys == 0:
        return np.trace(rho_reshaped, axis1=0, axis2=2)
    elif subsys == 1:
        return np.trace(rho_reshaped, axis1=1, axis2=3)
    else:
        raise ValueError("subsys must be 0 (to trace out A) or 1 (to trace out B).")

def purity(density_matrix: np.ndarray) -> float:
    """
    Calculate the purity of a quantum state.

    The purity is defined as Tr(rho^2), where rho is the density matrix.
    For a pure state, the purity is 1. For a mixed state of dimension `d`, 1/d <= purity < 1.

    Parameters
    ----------
    density_matrix : numpy.ndarray
        The density matrix of the quantum state.

    Returns
    -------
    float
        The purity of the quantum state.
    """
    return np.trace(np.dot(density_matrix, density_matrix)).real

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