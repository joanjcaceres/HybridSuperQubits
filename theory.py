import numpy as np

def bloch_waves_generator(Ej: float, phase:float, r: float, Ec: float, q: float, Nmax: int) -> np.ndarray:
    """
    Generate the matrix representation of the Andreev Hamiltonian in the Bloch waves representation.

    Parameters:
    - Ej (float): Josephson energy.
    - phase (float): Phase.
    - r (float): Reflectivity.
    - Ec (float): Charging energy.
    - q (float): Quasimomentum.
    - N (int): Size parameter for the matrix. The actual size will be 2*(N+1) x 2*(N+1).

    Returns:
    - np.ndarray: Generated matrix.
    """
    dimension = 2 * (2* Nmax + 1)
    matrix = np.zeros((dimension, dimension))
    N_list = np.arange(-Nmax, Nmax + 1) / 2

    G = np.repeat(N_list,2)  #this results in i.e.: [...,-1/2,-1/2,0/2,0/2,1/2,1/2,...]
    q_vals = q - G
    matrix[np.diag_indices(dimension)] = 4 * Ec * q_vals ** 2

    off_diag1 = np.zeros(dimension - 1)
    off_diag1[1::2] = -r * Ej / 2 # interleaved with zeros.
    np.fill_diagonal(matrix[1:], off_diag1)
    np.fill_diagonal(matrix[:, 1:], off_diag1)

    off_diag2 = np.zeros(dimension - 2)
    off_diag2[::2] = Ej / 2
    off_diag2[1::2] = -Ej / 2
    np.fill_diagonal(matrix[2:], off_diag2)
    np.fill_diagonal(matrix[:, 2:], off_diag2)

    off_diag3 = np.zeros(dimension - 3)
    off_diag3[::2] = r * Ej / 2 # interleaved with zeros.
    np.fill_diagonal(matrix[3:], off_diag3)
    np.fill_diagonal(matrix[:, 3:], off_diag3)

    eigvals,eigvecs = np.linalg.eigh(matrix)
    eigvecs_reshaped = eigvecs.reshape(dimension, dimension//2, 2)
    phase_factor = np.exp(1j * N_list * phase)
    bloch_waves = np.einsum('ijk,j->ik',eigvecs_reshaped,phase_factor)

    return bloch_waves


def andreev_bloch_waves(phi: float, Ej: float, r: float, Ec: float, q: float, N: int) -> np.ndarray:

    hamiltonian = bloch_andreev_hamiltonian(Ej = Ej, r = r, Ec = Ec, q = q, N = N)
    _, eigenvectors = np.linalg.eigh(hamiltonian)
    # eigenvectors_reshaped = eigenvectors.reshape(2*(N+1), -1, 2)

    m_vals = np.arange(-N//2, N//2 + 1)
    factor = np.exp(1j * m_vals/2 * phi)

    bloch_waves = [eigenvector * factor[:,np.newaxis] for eigenvector in eigenvectors_reshaped]
    bloch_waves = np.array(bloch_waves)
    return bloch_waves

def omega_matrix(Ej: float, r: float, Ec: float, q: float, N: int):
    q_list = np.linspace(0,1/2,100)
    phi_list = np.linspace(0,4*np.pi, 100)
    list_of_u_list = []
    list_of_dudq_list = []
    product_list = []
    for phi in phi_list:
        for q in q_list:
            andreev_bloch_eigenvectors = andreev_bloch_waves(phi, Ej, r, Ec, q, N)
            list_of_u_list.append(andreev_bloch_eigenvectors)
        list_of_u_list = np.array(list_of_u_list)
        list_of_dudq_list = np.gradient(list_of_u_list,q_list)
        #multiplicar aca los u con los dudq usando einsum
