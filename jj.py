import numpy as np
from scipy.sparse import diags
from scipy.linalg import eigh
import scipy.constants as const

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

def frequency_array_fit4(params):
     Cjj, Cg,Cg_big = params
     #To change the values below each time to fit
     N = 150
     Ljj = 4.1e-9
     return frequency_array_Cgbig(N,Cjj,Ljj,Cg,Cg_big,Cin=0,Cout = 0)[1:8]

def frequency_array_fit5(params):
     Cjj, Cg,Cg_big = params
     #To change the values below each time to fit
     N = 600
     Ljj = 4.1e-9
     return frequency_array_Cgbig_half(N,Cjj,Ljj,Cg,Cg_big,Cin=0,Cout = 0)[1:8]

def frequency_array_fit6(params):
     Cjj, Cg, Cin = params
     Cjj = Cjj*1e-15; Cg = Cg*1e-18; Cin = Cin*1e-18
     #To change the values below each time to fit
     N = 150
     Ljj = 4.1e-9
     return frequency_array_Cin_ini(N,Cjj,Ljj,Cg,Cin,Cout = 0)[1:11]

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
            matrix_diagonals = [np.ones(N)*(-Cjj),(2*Cjj+Cg)*np.ones(N+1),np.ones(N)*(-Cjj)]
            matrix = diags(matrix_diagonals,offsets=[-1,0,1]).toarray()
            matrix[0,0] = Cjj + Cin
            matrix[-1,-1] = Cjj + Cout
            matrix[N//2, N//2] = 2*Cjj
            return matrix
        
    def L_inv_matrix(N,Ljj):
        matrix_diagonals = [(-1/Ljj)*np.ones(N),(2/Ljj)*np.ones(N+1),(-1/Ljj)*np.ones(N)]
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

def frequency_array_Cgbig(N,Cjj,Ljj,Cg,Cg_big,Cin,Cout):

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
            matrix_diagonals = [np.ones(N)*(-Cjj),(2*Cjj+Cg)*np.ones(N+1),np.ones(N)*(-Cjj)]
            matrix = diags(matrix_diagonals,offsets=[-1,0,1]).toarray()
            matrix[0,0] = Cjj + Cin
            matrix[-1,-1] = Cjj + Cout
            matrix[N//2, N//2] = 2*Cjj+Cg_big
            return matrix
        
    def L_inv_matrix(N,Ljj):
        matrix_diagonals = [(-1/Ljj)*np.ones(N),(2/Ljj)*np.ones(N+1),(-1/Ljj)*np.ones(N)]
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

def frequency_array_Cgbig_half(N,Cjj,Ljj,Cg,Cg_big,Cin,Cout):

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
            matrix_diagonals = [np.ones(N)*(-Cjj),(2*Cjj+Cg)*np.ones(N+1),np.ones(N)*(-Cjj)]
            matrix = diags(matrix_diagonals,offsets=[-1,0,1]).toarray()
            matrix[0,0] = Cjj + Cin
            matrix[-1,-1] = Cjj + Cout
            matrix[-N//4, -N//4] = 2*Cjj+Cg_big
            matrix[N//4, N//4] = 2*Cjj+Cg_big
            return matrix
        
    def L_inv_matrix(N,Ljj):
        matrix_diagonals = [(-1/Ljj)*np.ones(N),(2/Ljj)*np.ones(N+1),(-1/Ljj)*np.ones(N)]
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

def frequency_array_Cin_ini(N,Cjj,Ljj,Cg,Cin,Cout):

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
            matrix_diagonals = [np.ones(N)*(-Cjj),(2*Cjj+Cg)*np.ones(N+1),np.ones(N)*(-Cjj)]
            matrix = diags(matrix_diagonals,offsets=[-1,0,1]).toarray()
            matrix[0,0] = Cjj + Cin
            matrix[-1,-1] = Cjj + Cout
            return matrix
        
    def L_inv_matrix(N,Ljj):
        matrix_diagonals = [(-1/Ljj)*np.ones(N),(2/Ljj)*np.ones(N+1),(-1/Ljj)*np.ones(N)]
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

def frequency_array_Cgbig_half_phi0(N,Cjj,Ljj,Cg,Cg_big,Cin,Cout):

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
            matrix_diagonals = [np.ones(N)*(-Cjj),(2*Cjj+Cg)*np.ones(N+1),np.ones(N)*(-Cjj)]
            matrix = diags(matrix_diagonals,offsets=[-1,0,1]).toarray()
            matrix[0,0] = Cjj + Cin
            matrix[-1,-1] = Cjj + Cout
            matrix[-N//4, -N//4] = 2*Cjj+Cg_big
            matrix[N//4, N//4] = 2*Cjj+Cg_big
            return matrix
        
    def L_inv_matrix(N,Ljj):
        matrix_diagonals = [(-1/Ljj)*np.ones(N),(2/Ljj)*np.ones(N+1),(-1/Ljj)*np.ones(N)]
        matrix = diags(matrix_diagonals,[-1,0,1]).toarray()
        matrix[0,0] = 1/Ljj
        matrix[-1,-1] = 1/Ljj
        return matrix

    # Solving C^-1/2 L^-1 C^-1/2 phi = \omega^2 phi
    eigenvalues_C, eigenvectors_C = eigh(C_matrix(N,Cjj,Cg,Cin,Cout))
    Lambda_inv_sqrt = np.diag(1 / np.sqrt(eigenvalues_C))
    C_inv_sqrt = np.dot(eigenvectors_C, np.dot(Lambda_inv_sqrt, eigenvectors_C.T)) # spectral decomposition of C^-1/2
    
    matrix_operation = np.dot(C_inv_sqrt, np.dot(L_inv_matrix(N,Ljj), C_inv_sqrt)) # C^-1/2 L^-1 C^-1/2
    eigvals, eigvecs = eigh(matrix_operation) # eigenvalues and eigvecs of C^-1/2 L^-1 C^-1/2

    phase_fluctuation = np.sqrt(const.h/2/eigvals[1])*(np.dot(C_inv_sqrt,eigvecs)[-1] - np.dot(C_inv_sqrt,eigvecs)[0]) #Note that the first mode is eigvals[1] not eigvals[0]

    return np.sqrt(eigvals)/2/np.pi, phase_fluctuation

def frequency_array_Cgbig_half_phi0_impedance(N,Cjj,Ljj,Cg,Cg_big,Cin=0,Cout=0):

    def C_matrix(N,Cjj,Cg,Cin,Cout):
            matrix_diagonals = [np.ones(N)*(-Cjj),(2*Cjj+Cg)*np.ones(N+1),np.ones(N)*(-Cjj)]
            matrix = diags(matrix_diagonals,offsets=[-1,0,1]).toarray()
            matrix[0,0] = Cjj + Cin
            matrix[-1,-1] = Cjj + Cout
            matrix[-N//4, -N//4] = 2*Cjj+Cg_big
            matrix[N//4, N//4] = 2*Cjj+Cg_big
            return matrix
        
    def L_inv_matrix(N,Ljj):
        matrix_diagonals = [(-1/Ljj)*np.ones(N),(2/Ljj)*np.ones(N+1),(-1/Ljj)*np.ones(N)]
        matrix = diags(matrix_diagonals,[-1,0,1]).toarray()
        matrix[0,0] = 1/Ljj
        matrix[-1,-1] = 1/Ljj
        return matrix

    # Solving C^-1/2 L^-1 C^-1/2 phi = \omega^2 phi
    eigenvalues_C, eigenvectors_C = eigh(C_matrix(N,Cjj,Cg,Cin,Cout))
    Lambda_inv_sqrt = np.diag(1 / np.sqrt(eigenvalues_C))
    C_inv_sqrt = np.dot(eigenvectors_C, np.dot(Lambda_inv_sqrt, eigenvectors_C.T)) # spectral decomposition of C^-1/2

    matrix_operation = np.dot(C_inv_sqrt, np.dot(L_inv_matrix(N,Ljj), C_inv_sqrt)) # C^-1/2 L^-1 C^-1/2
    eigvals, eigvecs = eigh(matrix_operation) # eigenvalues and eigvecs of C^-1/2 L^-1 C^-1/2
    resonances = np.sqrt(eigvals) #Hz.rad

    impedance_array = []
    for j in range(1,N+1):
        # phase_fluctuation2 = np.sqrt(const.hbar/2/resonances[1])*(np.dot(C_inv_sqrt,eigvecs[:,j])[-1] - np.dot(C_inv_sqrt,eigvecs[:,j])[0]) #Note that the first mode is eigvals[1] not eigvals[0]
        impedance_array.append((np.sqrt(1/resonances[j])*(np.dot(C_inv_sqrt,eigvecs[:,j])[-1] - np.dot(C_inv_sqrt,eigvecs[:,j])[0]))**2) #Note that the first mode is eigvals[1] not eigvals[0])

    impedance_array = np.array(impedance_array)
    return resonances/2/np.pi, impedance_array


def frequency_array_Cgbig_phi0_impedance(N,Cjj,Ljj,Cg,Cg_big,Cin,Cout):

    def C_matrix(N,Cjj,Cg,Cin,Cout):
            matrix_diagonals = [np.ones(N)*(-Cjj),(2*Cjj+Cg)*np.ones(N+1),np.ones(N)*(-Cjj)]
            matrix = diags(matrix_diagonals,offsets=[-1,0,1]).toarray()
            matrix[0,0] = Cjj + Cin
            matrix[-1,-1] = Cjj + Cout
            matrix[N//2, N//2] = 2*Cjj+Cg_big
            return matrix
        
    def L_inv_matrix(N,Ljj):
        matrix_diagonals = [(-1/Ljj)*np.ones(N),(2/Ljj)*np.ones(N+1),(-1/Ljj)*np.ones(N)]
        matrix = diags(matrix_diagonals,[-1,0,1]).toarray()
        matrix[0,0] = 1/Ljj
        matrix[-1,-1] = 1/Ljj
        return matrix

    # Solving C^-1/2 L^-1 C^-1/2 phi = \omega^2 phi
    eigenvalues_C, eigenvectors_C = eigh(C_matrix(N,Cjj,Cg,Cin,Cout))
    Lambda_inv_sqrt = np.diag(1 / np.sqrt(eigenvalues_C))
    C_inv_sqrt = np.dot(eigenvectors_C, np.dot(Lambda_inv_sqrt, eigenvectors_C.T)) # spectral decomposition of C^-1/2

    matrix_operation = np.dot(C_inv_sqrt, np.dot(L_inv_matrix(N,Ljj), C_inv_sqrt)) # C^-1/2 L^-1 C^-1/2
    eigvals, eigvecs = eigh(matrix_operation) # eigenvalues and eigvecs of C^-1/2 L^-1 C^-1/2
    resonances = np.sqrt(eigvals)

    impedance_array = []
    for j in range(1,N+1):
        # phase_fluctuation2 = np.sqrt(const.hbar/2/resonances[1])*(np.dot(C_inv_sqrt,eigvecs[:,j])[-1] - np.dot(C_inv_sqrt,eigvecs[:,j])[0]) #Note that the first mode is eigvals[1] not eigvals[0]
        impedance_array.append((np.sqrt(1/resonances[j])*(np.dot(C_inv_sqrt,eigvecs[:,j])[-1] - np.dot(C_inv_sqrt,eigvecs[:,j])[0]))**2) #Note that the first mode is eigvals[1] not eigvals[0])

    impedance_array = np.array(impedance_array)
    return resonances/2/np.pi, impedance_array

def phase_slip_rate(wp,El,N=150):
    Es = np.sqrt(2/np.pi) * np.sqrt(8*El*N/wp)* (const.hbar * wp) * np.exp(-8*El*N/wp)
    return np.pi * np.sqrt(N)*Es/const.h