"""
Fluxonium Parameter Calculation Module

This module provides functions to calculate the parameters of the Hamiltonian for a fluxonium qubit.
These functions are based on the methodologies described in the paper:

Di Paolo et al. (2021), "Efficient modeling of superconducting quantum circuits with tensor networks",
npj Quantum Information 7, 11. https://doi.org/10.1038/s41534-020-00352-4

The fluxonium qubit is a type of superconducting qubit that utilizes a Josephson junction array to
create a superinductance. This module helps in generating matrices that are essential for the 
Hamiltonian calculations described in the paper.

Functions:
    - create_matrix_R: Eq (9).
    - generate_matriz_Cphi: Generate the Cphi matrix.
    - generate_matriz_Ctheta: Generate the Ctheta matrix.
    - calculate_a_coeff: Eq (17).
    - generate_R_1: Eq (16).
    - generate_C_x_1: Eq (19).
"""

import numpy as np

def create_matrix_R(N):
    """
    Create an (N+1)x(N+1) identity matrix and modify it according to the rules specified.
    
    Parameters:
    N (int): The size of the matrix (N+1)x(N+1)
    
    Returns:
    np.ndarray: The resulting matrix R.
    """
    R = np.eye(N + 1)

    for i in range(N):
        R[i, i + 1] = -1

    R[-1, :] = 1

    return R

def generate_matriz_Cphi(N, CJ, C0, CJb):
    """
    Generate the Cphi matrix.
    
    Parameters:
    N (int): The size of the matrix (N+1)x(N+1)
    CJ (float): Capacitance value for CJ
    C0 (float): Capacitance value for C0
    CJb (float): Capacitance value for CJb
    
    Returns:
    np.ndarray: The resulting Cphi matrix.
    """
    matriz = np.zeros((N + 1, N + 1))

    for i in range(N + 1):
        matriz[i, i] = 2 * CJ + C0
        if i == 0 or i == N:
            matriz[i, i] = CJb + CJ + C0

    for i in range(N):
        matriz[i, i + 1] = -CJ
        matriz[i + 1, i] = -CJ

    matriz[0, N] = -CJb
    matriz[N, 0] = -CJb

    return matriz

def generate_matriz_Ctheta(N, CJ, C0, CJb):
    """
    Generate the Ctheta matrix using the R matrix and Cphi matrix.
    
    Parameters:
    N (int): The size of the matrix (N+1)x(N+1)
    CJ (float): Capacitance value for CJ
    C0 (float): Capacitance value for C0
    CJb (float): Capacitance value for CJb
    
    Returns:
    np.ndarray: The resulting Ctheta matrix.
    """
    R_matrix = create_matrix_R(N)
    R_matrix_inv = np.linalg.inv(R_matrix)
    R_matrix_inv_T = R_matrix_inv.T
    Cphi = generate_matriz_Cphi(N, CJ, C0, CJb)
    return R_matrix_inv_T @ Cphi @ R_matrix_inv

def calculate_a_coeff(N, CJ, C0, CJb):
    """
    Calculate the a coefficients for the R_1 matrix.
    
    Parameters:
    N (int): The size of the matrix (N+1)x(N+1)
    CJ (float): Capacitance value for CJ
    C0 (float): Capacitance value for C0
    CJb (float): Capacitance value for CJb
    
    Returns:
    np.ndarray: The resulting a coefficients.
    """
    Ctheta = generate_matriz_Ctheta(N, CJ, C0, CJb)
    return np.sum(Ctheta[1:N, 1:N], axis=0) / np.sum(Ctheta[:N, :N]) - 1

def generate_R_1(N, CJ, C0, CJb):
    """
    Generate the R_1 matrix.
    
    Parameters:
    N (int): The size of the matrix (N+1)x(N+1)
    CJ (float): Capacitance value for CJ
    C0 (float): Capacitance value for C0
    CJb (float): Capacitance value for CJb
    
    Returns:
    np.ndarray: The resulting R_1 matrix.
    """
    a_coeff = calculate_a_coeff(N, CJ, C0, CJb)
    R_1_matrix = np.eye(N + 1)
    R_1_matrix[1:-1, 0] = -1
    R_1_matrix[0, 0] = 1 - np.sum(a_coeff)
    R_1_matrix[0, 1:-1] = 1 + a_coeff

    return R_1_matrix

def generate_C_x_1(N, CJ, C0, CJb):
    """
    Generate the C_x_1 matrix.
    
    Parameters:
    N (int): The size of the matrix (N+1)x(N+1)
    CJ (float): Capacitance value for CJ
    C0 (float): Capacitance value for C0
    CJb (float): Capacitance value for CJb
    
    Returns:
    np.ndarray: The resulting C_x_1 matrix.
    """
    R_1_matrix = generate_R_1(N, CJ, C0, CJb)
    R_1_matrix_inv = np.linalg.inv(R_1_matrix)
    R_1_matrix_inv_T = R_1_matrix_inv.T

    C_X_0_matrix = generate_matriz_Ctheta(N, CJ, C0, CJb)

    return R_1_matrix_inv_T @ C_X_0_matrix @ R_1_matrix_inv