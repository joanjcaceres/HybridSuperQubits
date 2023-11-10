import numpy as np

# All the function below are valid for the Andreev Hamiltonian.

def bloch_andreev_hamiltonian(Ej: float, r: float, Ec: float, q: float, N: int) -> np.ndarray:
    """
    Generate the matrix representation of the Andreev Hamiltonian in the Bloch waves representation.

    Parameters:
    - Ej (float): Josephson energy.
    - r (float): Reflectivity.
    - Ec (float): Charging energy.
    - q (float): Quasimomentum.
    - N (int): Size parameter for the matrix. The actual size will be 2*(N+1) x 2*(N+1).

    Returns:
    - np.ndarray: Generated matrix.
    """
    dimension = 2 * (N + 1)
    matrix = np.zeros((dimension, dimension))

    for i, idx in enumerate(range(-N, N + 2)):

        q_val = q - (idx // 2) / 2
        matrix[i, i] = 4 * Ec * q_val ** 2

        if i + 1 < dimension:
            matrix[i, i + 1] = matrix[i + 1, i] = (-r * Ej / 2) if idx % 2 != 0 else 0

        if i + 2 < dimension:
            matrix[i, i + 2] = matrix[i + 2, i] = (Ej / 2) if idx % 2 == 0 else (-Ej / 2)

        if i + 3 < dimension:
            matrix[i, i + 3] = matrix[i + 3, i] = (r * Ej / 2) if idx % 2 == 0 else 0

    return matrix

import numpy as np


def andreev_bloch_waves(phi: float, Ej: float, r: float, Ec: float, q: float, N: int) -> np.ndarray:

    hamiltonian = bloch_andreev_hamiltonian(Ej = Ej, r = r, Ec = Ec, q = q, N = N)
    _, eigenvectors = np.linalg.eigh(hamiltonian)
    eigenvectors_reshaped = eigenvectors.reshape(2*(N+1), -1, 2)

    m_vals = np.arange(-N/2, N/2 + 1)
    factor = np.exp(1j * m_vals/2 * phi)

    # andreev_bloch_eigenvectors = eigenvectors * factors

    return eigenvectors_reshaped,factor

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
