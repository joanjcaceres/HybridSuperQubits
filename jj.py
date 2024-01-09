import numpy as np
from scipy.sparse import diags
from scipy.linalg import eigh

def frequency_array_fit(xdata, Cjj,Cg,Cin):
     #To change the values below each time to fit
     N = 150
     Ljj = 4.1e-9
     return frequency_array(N,Cjj,Ljj,Cg,Cin,Cout = 0)[1:7]

def frequency_array_fit2(params):
     Cjj,Cg,Cin = params
     #To change the values below each time to fit
     N = 150
     Ljj = 4.1e-9
     return frequency_array(N,Cjj,Ljj,Cg,Cin,Cout = 0)[1:7]

def frequency_array_fit3(params):
     Cg,Cin = params
     #To change the values below each time to fit
     N = 150
     Cjj = 32.4e-15
     Ljj = 4.1e-9
     return frequency_array(N,Cjj,Ljj,Cg,Cin,Cout = 0)[1:7]

def resonance_model(j, fp, cg_over_cj):
    N = 150
    return fp * np.sqrt((1-np.cos(j*np.pi/N))/(1-np.cos(j*np.pi/N) + cg_over_cj/2))

def resonance_model2(j, cg_over_cj):
    N = 150
    Cjj = 32.4e-15
    Ljj = 4.1e-9
    fp = 1/np.sqrt(Cjj*Ljj)/2/np.pi
    return fp * np.sqrt((1-np.cos(j*np.pi/N))/(1-np.cos(j*np.pi/N) + cg_over_cj/2))

def resonance_model3(j, cg_over_cj , geo_L):
    N = 150
    Cjj = 32.4e-15
    Ljj = 4.1e-9
    fp = 1/np.sqrt(Cjj*(Ljj+geo_L))/2/np.pi
    return fp * np.sqrt((1-np.cos(j*np.pi/N))/(1-np.cos(j*np.pi/N) + cg_over_cj/2))

def frequency_array(N,Cjj,Ljj,Cg,Cin,Cout):

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

    def C_matrix(N,Cjj,Cg,Cin,Cout):
            matrix_diagonals = [np.ones(N-1)*(-Cjj),(2*Cjj+Cg)*np.ones(N),np.ones(N-1)*(-Cjj)]
            matrix = diags(matrix_diagonals,offsets=[-1,0,1]).toarray()
            matrix[0,0] = Cjj + Cin
            matrix[-1,-1] = Cjj + Cout
            return matrix
        
    def L_inv_matrix(N,Ljj):
        matrix_diagonals = [(-1/Ljj)*np.ones(N-1),(2/Ljj)*np.ones(N),(-1/Ljj)*np.ones(N-1)]
        matrix = diags(matrix_diagonals,[-1,0,1]).toarray()
        matrix[0,0] = 1/Ljj
        matrix[-1,-1] = 1/Ljj
        return matrix

    # Solving C^-1/2 L^-1 C^-1/2 phi = \omega^2 phi
    eigenvalues_C, eigenvectors_C = eigh(C_matrix(N,Cjj,Cg,Cin,Cout))
    # epsilon = 1e-100  # Un pequeño valor positivo para reemplazar valores propios negativos o muy pequeños
    # eigenvalues_C_safe = np.where(eigenvalues_C > epsilon, eigenvalues_C, epsilon)
    # Lambda_inv_sqrt = np.diag(1 / np.sqrt(eigenvalues_C_safe))

    Lambda_inv_sqrt = np.diag(1 / np.sqrt(eigenvalues_C))
    C_inv_sqrt = np.dot(eigenvectors_C, np.dot(Lambda_inv_sqrt, eigenvectors_C.T)) # spectral decomposition of C^-1/2
    matrix_operation = np.dot(C_inv_sqrt, np.dot(L_inv_matrix(N,Ljj), C_inv_sqrt)) # C^-1/2 L^-1 C^-1/2
    eigvals, eigvecs = eigh(matrix_operation) # eigenvalues and eigvecs of C^-1/2 L^-1 C^-1/2

    return np.sqrt(eigvals)/2/np.pi