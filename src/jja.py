import numpy as np
from scipy.sparse import diags
from scipy.linalg import eigh

def C_matrix(N,Cjj,Cg,Cg_big,Cin,Cout):
        #TODO: #5 Add documentation and adapt the introduction of Cg_big
        matrix_diagonals = [(-Cjj)*np.ones(N),(2*Cjj+Cg)*np.ones(N+1),(-Cjj)*np.ones(N)]
        matrix = diags(matrix_diagonals,offsets=[-1,0,1]).toarray()
        matrix[0,0] = Cjj + Cin
        matrix[-1,-1] = Cjj + Cout
        matrix[77,77] = 2*Cjj+ Cg_big
        matrix[-78,-78] = 2*Cjj+ Cg_big
        return matrix

def L_inv_matrix(N,Ljj):
    matrix_diagonals = [(-1/Ljj)*np.ones(N),(2/Ljj)*np.ones(N+1),(-1/Ljj)*np.ones(N)]
    matrix = diags(matrix_diagonals,[-1,0,1]).toarray()
    matrix[0,0] = 1/Ljj
    matrix[-1,-1] = 1/Ljj
    return matrix

def jja_resonances(params):
    Ljj = params[0]*1e-9
    Cjj = params[1]*1e-15
    Cg = params[2]*1e-18
    Cg_big = params[3]*1e-15
    Cin = params[4]*1e-15
    
    N = 170
    Cout = 0

    # RESOLVER EL PROBLEMA DE REDONDEO DE NUMEROS PEQUEÑOS.

    # Eigensolution of the linearized circuit matrix for a Josephson junction chain with the following parameters:
    # N: number of junctions
    # Cjj: junction capacitance
    # Ljj: junction inductance
    # Cg: ground capacitance
    # Cin: input capacitance
    # Cout: output capacitance

    # Return the frequency in Hz.

    # References:
    #   https://theses.hal.science/tel-01369020
    #   https://journals.aps.org/prb/abstract/10.1103/PhysRevB.92.104508

    # Solving C^-1/2 L^-1 C^-1/2 phi = \omega^2 phi

    eigenvalues_C, eigenvectors_C = eigh(C_matrix(N,Cjj,Cg,Cg_big,Cin,Cout))
    Lambda_inv_sqrt = np.diag(1 / np.sqrt(eigenvalues_C))
    C_inv_sqrt = np.dot(eigenvectors_C, np.dot(Lambda_inv_sqrt, eigenvectors_C.T)) # spectral decomposition of C^-1/2
    matrix_operation = np.dot(C_inv_sqrt, np.dot(L_inv_matrix(N,Ljj), C_inv_sqrt)) # C^-1/2 L^-1 C^-1/2
    eigvals = eigh(matrix_operation, eigvals_only=True) # eigenvalues and eigvecs of C^-1/2 L^-1 C^-1/2

    return np.sqrt(eigvals)/2/np.pi

def jja_eigensys(params, **kwargs):
    Ljj = params[0]*1e-9
    Cjj = params[1]*1e-15
    Cg = params[2]*1e-18
    Cg_big = params[3]*1e-15
    Cin = params[4]*1e-15
    
    N = 170
    Cout = 0

    eigenvalues_C, eigenvectors_C = eigh(C_matrix(N,Cjj,Cg,Cg_big,Cin,Cout))
    Lambda_inv_sqrt = np.diag(1 / np.sqrt(eigenvalues_C))
    C_inv_sqrt = np.dot(eigenvectors_C, np.dot(Lambda_inv_sqrt, eigenvectors_C.T)) # spectral decomposition of C^-1/2
    matrix_operation = np.dot(C_inv_sqrt, np.dot(L_inv_matrix(N,Ljj), C_inv_sqrt)) # C^-1/2 L^-1 C^-1/2
    eigvals, eigvecs = eigh(matrix_operation, **kwargs) # eigenvalues and eigvecs of C^-1/2 L^-1 C^-1/2

    return np.sqrt(eigvals), eigvecs # Returns \omega_k, \Phi_k





