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
    dimension = 2*(N+1)
    # Initialize the matrix with zeros. The matrix has dimensions [dimension, dimension]
    matrix = np.zeros((dimension, dimension))
    
    # Populate the main diagonal and the sub-diagonals.
    for i, idx in enumerate(np.arange(-N, N + 2)):
        # Populate the main diagonal with 4 * Ec * (q + i/2)^2
        q_val = q - (idx // 2) / 2  # Calculate q + i/2 for each pair of indices
        matrix[i, i] = 4 * Ec * q_val ** 2
        
        # Populate the sub-diagonals and super-diagonals.
        if i + 1 < 2 * (N + 1):
            matrix[i, i + 1] = 0 if i % 2 == 0 else -r * Ej / 2
            matrix[i + 1, i] = 0 if i % 2 == 0 else -r * Ej / 2
            
        if i + 2 < 2 * (N + 1):
            matrix[i, i + 2] = Ej / 2 if i % 2 == 0 else -Ej / 2
            matrix[i + 2, i] = Ej / 2 if i % 2 == 0 else -Ej / 2

        if i + 3 < 2 * (N + 1):
            matrix[i, i + 3] = - r * Ej / 2 if i % 2 == 0 else 0
            matrix[i + 3, i] = r * Ej / 2 if i % 2 == 0 else 0
            
    return matrix

def andreev_eigensystem(hamiltonian: np.ndarray):
    eigenvalues, eigenvectors = np.linalg.eig(hamiltonian)
    return eigenvalues, eigenvectors