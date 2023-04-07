from qutip import *
# from tqdm import tqdm
from scipy import special
import numpy as np
from scipy.interpolate import interp1d
from scipy.special import hermite
# Constants

h = 6.62607015e-34
hbar = h/2/np.pi
k = 1.380649e-23
T_k = 36.8e-3 #imposing detailed balanced in the even transitions rates at Fig. 10.7 of Cyril PhD Thesis.
e = 1.602e-19
phi_0 = h/2/e

def eigensystem_fbq(Ec,El,EDelta,phi_ext,r, N = 200, eigvals = 0):
    # EIGENVALUES AND EIGENSTATES OF THE FERMIONIC-BOSONIC QUBIT 

    phi_ZPF=(8.0 * Ec / El) ** 0.25
    N_op  = 1j * (destroy(N).dag() - destroy(N)) / phi_ZPF /np.sqrt(2)
    phi_op= (destroy(N).dag() + destroy(N)) * phi_ZPF* np.sqrt(2)

    delta = phi_op - phi_ext
    ReZ = (phi_op/2).cosm()*(r*phi_op/2).cosm()+r*(phi_op/2).sinm()*(r*phi_op/2).sinm()
    ImZ = -(phi_op/2).cosm()*(r*phi_op/2).sinm()+r*(phi_op/2).sinm()*(r*phi_op/2).cosm()
    H = 4*Ec*tensor(N_op**2,qeye(2)) + 0.5*El*tensor((delta)**2,qeye(2)) - EDelta*(tensor(ReZ,sigmaz())+tensor(ImZ,sigmay())) #Hamiltonian.
    evals,ekets=H.eigenstates(eigvals=eigvals)
    return evals,ekets


def eigenenergies_fbq(Ec,El,EDelta,phi_ext,r, N = 200, eigvals = 0):
    # ONLY EIGENVALUES OF THE FERMIONIC-BOSONIC QUBIT 

    phi_ZPF=(8.0 * Ec / El) ** 0.25
    N_op  = 1j * (destroy(N).dag() - destroy(N)) / phi_ZPF /np.sqrt(2)
    phi_op= (destroy(N).dag() + destroy(N)) * phi_ZPF*np.sqrt(2)
    delta = phi_op - phi_ext
    ReZ = (phi_op/2).cosm()*(r*phi_op/2).cosm()+r*(phi_op/2).sinm()*(r*phi_op/2).sinm()
    ImZ = -(phi_op/2).cosm()*(r*phi_op/2).sinm()+r*(phi_op/2).sinm()*(r*phi_op/2).cosm()
    H = 4*Ec*tensor(N_op**2,qeye(2)) + 0.5*El*tensor((delta)**2,qeye(2)) + EDelta*(tensor(ReZ,sigmaz())+tensor(ImZ,sigmay())) #Hamiltonian.
    evals=H.eigenenergies(eigvals=eigvals)

    return evals


def phi_n(EC,EL, n: int,phi_list):
    # CONVERT THE FOCK SPACE STATES IN HARMONIC OSCILLATOR WAVEFUNCTIONS

    phi_ZPF=(8.0 * EC / EL) ** 0.25      
    return 1/np.sqrt(np.sqrt(np.pi)*(2**n) * 1.0* phi_ZPF * np.math.factorial(n)) * np.exp(-(phi_list/phi_ZPF)**2 / 2.0) * np.polyval(hermite(n), (phi_list/phi_ZPF))


def wavefunction_phi_fbq_up(EC,EL,phi_list,phi,N):
    # Wavefunction(\varphi) FOR \sigmaz = +1 OF AN EIGENSTATE OF THE Fermionic-Bosonic Qubit.

    wfunc = np.zeros(len(phi_list),dtype = complex)
    for n in range(np.int(N/2)+1):
        wfunc = wfunc + phi_n(EC,EL,n,phi_list)*phi.full()[2*n,0]
    return wfunc

def wavefunction_phi_fbq_down(EC,EL,phi_list,phi,N):
    # Wavefunction(\varphi) FOR \sigmaz = -1 OF AN EIGENSTATE OF THE Fermionic-Bosonic Qubit.

    wfunc = np.zeros(len(phi_list),dtype = complex)
    for n in range(np.int(N/2)):
        wfunc = wfunc + phi_n(EC,EL,n,phi_list)*phi.full()[2*n+1,0]
    return wfunc

def eigensystem_and_matrix_elements_Josephson(Ec,El,EDelta,phi_ext,r, N = 200, eigvals = 0):
    # Obtain the eigenvalues, eigenkets and the |<1|O|0>|^2 of the n operator, phi operator and dH/dphi_ext operator.

    phi_ZPF=(8.0 * Ec / El) ** 0.25
    N_op  = 1j * (destroy(N).dag() - destroy(N)) / phi_ZPF /np.sqrt(2)
    phi_op= (destroy(N).dag() + destroy(N)) * phi_ZPF *np.sqrt(2)

    delta = phi_op-phi_ext

    H = 4*Ec*N_op**2 + 0.5*El*delta**2 + EDelta*(1-(1-r**2)*(phi_op/2).sinm()**2).sqrtm() #Hamiltonian.
    evals,ekets=H.eigenstates(eigvals=eigvals)
    evals = np.real(evals)

    N_op01 = N_op.matrix_element(ekets[1],ekets[0]) 
    phi_op01 = phi_op.matrix_element(ekets[1],ekets[0])

    dH_dr_numerator = np.sqrt(2)*r*(phi_op/2).sinm()**2
    dH_dr_denominator = (1+r**2+phi_op.cosm() - r**2*phi_op.cosm()).sqrtm()
    dHdr_op01 = (dH_dr_numerator*dH_dr_denominator.inv()).matrix_element(ekets[1],ekets[0])

    matrix_op_sqr_list = np.array([N_op01,phi_op01,dHdr_op01],dtype = complex)
    return evals,ekets,matrix_op_sqr_list


# COHERENCE TIME CALCULATION

def eigensystem_and_matrix_elements_sqr_fbq(Ec,El,EDelta,phi_ext,r, N = 200, eigvals = 0):
    # Obtain the eigenvalues, eigenkets and the |<1|O|0>|^2 of the n operator, phi operator and dH/dphi_ext operator.

    phi_ZPF=(8.0 * Ec / El) ** 0.25
    N_op  = 1j * (destroy(N).dag() - destroy(N)) / phi_ZPF /np.sqrt(2)
    phi_op= (destroy(N).dag() + destroy(N)) * phi_ZPF*np.sqrt(2)

    delta = phi_op-phi_ext
    ReZ = (phi_op/2).cosm()*(r*phi_op/2).cosm()+r*(phi_op/2).sinm()*(r*phi_op/2).sinm() #Re(Z) of the Hamiltonian
    ImZ = -(phi_op/2).cosm()*(r*phi_op/2).sinm()+r*(phi_op/2).sinm()*(r*phi_op/2).cosm() ##Im(Z) of the Hamiltonian

    H = 4*Ec*tensor(N_op,qeye(2))**2 + 0.5*El*tensor(delta,qeye(2))**2 + EDelta*(tensor(ReZ,sigmaz())+tensor(ImZ,sigmay())) #Hamiltonian.
    evals,ekets=H.eigenstates(eigvals=eigvals)
    evals = np.real(evals)

    dReZdr = 1/2*(r*phi_op*(r*phi_op/2).cosm()*(phi_op/2).sinm()+(-phi_op*(phi_op/2).cosm()+2*(phi_op/2).sinm())*(r*phi_op/2).sinm())
    dImZdr = -1/2*(r*phi_op/2).cosm()*(phi_op*(phi_op/2).cosm()-2*(phi_op/2).sinm())-1/2*r*phi_op*(phi_op/2).sinm()*(r*phi_op/2).sinm()
    dHdr = EDelta*(tensor(dReZdr,sigmaz())+tensor(dImZdr,sigmay()))

    N_op01 = tensor(N_op,qeye(2)).matrix_element(ekets[1],ekets[0]) 
    phi_op01 = tensor(phi_op,qeye(2)).matrix_element(ekets[1],ekets[0])
    dHdr_op01 = dHdr.matrix_element(ekets[1],ekets[0])

    matrix_op_sqr_list = np.array([N_op01,phi_op01,dHdr_op01])
    return evals,ekets,matrix_op_sqr_list


def eigensystem_and_gamma_fbq(Ec,El,EDelta,phi_ext,r, N = 200, eigvals = 0):
    # Obtain the eigenvalues, eigenkets, Gamma_1 and the matrix_elements_{01} of the n, phi and sigma_z operators

    phi_ZPF=(8.0 * Ec / El) ** 0.25
    N_op  = 1j * (destroy(N).dag() - destroy(N)) / phi_ZPF /np.sqrt(2)
    phi_op= (destroy(N).dag() + destroy(N)) * phi_ZPF*np.sqrt(2)

    delta = phi_op - phi_ext
    ReZ = (phi_op/2).cosm()*(r*phi_op/2).cosm()+r*(phi_op/2).sinm()*(r*phi_op/2).sinm()
    ImZ = -(phi_op/2).cosm()*(r*phi_op/2).sinm()+r*(phi_op/2).sinm()*(r*phi_op/2).cosm()

    dReZdr = 1/2*(r*phi_op*(r*phi_op/2).cosm()*(phi_op/2).sinm()+(-phi_op*(phi_op/2).cosm()+2*(phi_op/2).sinm())*(r*phi_op/2).sinm())
    dImZdr = -1/2*(r*phi_op/2).cosm()*(phi_op*(phi_op/2).cosm()-2*(phi_op/2).sinm())-1/2*r*phi_op*(phi_op/2).sinm()*(r*phi_op/2).sinm()
    dHdr = EDelta*(tensor(dReZdr,sigmaz())+tensor(dImZdr,sigmay()))

    H = 4*Ec*tensor(N_op**2,qeye(2)) + 0.5*El*tensor((delta)**2,qeye(2)) + EDelta*(tensor(ReZ,sigmaz())+tensor(ImZ,sigmay())) #Hamiltonian.
    evals,ekets=H.eigenstates(eigvals=eigvals)
    E01 = np.abs(evals[1]-evals[0])

    N_op01 = tensor(N_op,qeye(2)).matrix_element(ekets[1],ekets[0]) 
    phi_op01 = tensor(phi_op,qeye(2)).matrix_element(ekets[1],ekets[0])
    dHdr01 = dHdr.matrix_element(ekets[1],ekets[0])

    Gamma_capacitive = np.abs(N_op01)**2 * S_capacitive(Ec,E01) *(2*e/hbar)**2
    Gamma_inductive = np.abs(phi_op01)**2 * S_inductive(El,E01) * (phi_0/2/np.pi/hbar)**2
    Gamma_chargeCoupledImpedance = np.abs(N_op01)**2 * S_charge_coupled_impedance(E01) *(2*e/hbar)**2
    Gamma_FluxBiasLine = np.abs(phi_op01)**2 * S_flux_bias_line(E01) * (2*np.pi/phi_0*El*h/hbar)**2
    Gamma_1overf = np.abs(phi_op01)**2 * S_1f_flux(E01) * (h*El*2*np.pi/phi_0/hbar)**2
    Gamma_Andreev_1f = np.abs(dHdr01)**2 * S_Andreev_1f(E01)*(h/hbar)**2

    Gamma_T1 = np.real(np.array([Gamma_capacitive,Gamma_inductive,Gamma_chargeCoupledImpedance,Gamma_FluxBiasLine,Gamma_1overf,Gamma_Andreev_1f]))
    operator_list = np.array([N_op01,phi_op01,dHdr01])
    return evals,ekets,Gamma_T1,operator_list


def gamma_dephasing_flux(A_lambda,x_list,e01_vs_x):
    # Obtain the Gamma_phi for a noise amplitud A_lambda
    # e01_vs_x should be in Hz.
    # Reference: Eq. (13): Peter Groszkowski et al 2018 New J. Phys. 20 043053
    
    x_list2 = np.concatenate([-x_list[::-1][0:-1],x_list])
    e01_vs_x2 = np.concatenate([e01_vs_x[::-1][0:-1],e01_vs_x])
    dE01dx = np.gradient(e01_vs_x2,x_list2,edge_order=2)
    d2E01dx2 = np.gradient(dE01dx,x_list2,edge_order=2)

    w_ir = 2*np.pi
    w_uv = 2*np.pi*3e9
    t = 10e-6

    first_term = 2*A_lambda**2*(2*np.pi*dE01dx)**2*np.abs(np.log(w_ir*t))
    second_term = 2*A_lambda**4*(2*np.pi*d2E01dx2)**2 * (np.log(w_uv/w_ir)**2 + 2*np.log(w_ir*t)**2)

    return np.sqrt(first_term + second_term)

def gamma_dephasing_r(A_lambda,x_list,e01_vs_x):
    # Obtain the Gamma_phi for a noise amplitud A_lambda
    # e01_vs_x should be in Hz.
    # This is without the concatenate (we don't need to mirror the data to obtain the Gamma for r)
    # we don't calculate the derivative of H wrt r at r = 0 to make the mirror.

    dE01dx = np.gradient(e01_vs_x,x_list,edge_order=2)
    d2E01dx2 = np.gradient(dE01dx,x_list,edge_order=2)

    w_ir = 2*np.pi
    w_uv = 2*np.pi*3e9
    t = 10e-6

    first_term = 2*A_lambda**2*(2*np.pi*dE01dx)**2*np.abs(np.log(w_ir*t))
    second_term = 2*A_lambda**4*(2*np.pi*d2E01dx2)**2 * (np.log(w_uv/w_ir)**2 + 2*np.log(w_ir*t)**2)

    return np.sqrt(first_term + second_term)


def Temp_factor(E01):
    return 1/np.tanh(h*np.abs(E01)/2 /k /T_k)

def S_capacitive(Ec,E01):

    def Q_cap(E01):
        return (0.35e6*(6e9/E01)**0.15)

    Cj = e**2/2/Ec/h
    return 2*hbar/Cj/Q_cap(E01)*Temp_factor(E01)

def S_inductive(El,E01):
        
    def Q_ind_fun(E01):
        return 500e6* (special.k0(h*0.5e9/2/k/T_k) * np.sinh(h*0.5e9/2/k/T_k) / special.k0(h*np.abs(E01)/2/k/T_k) / np.sinh(h*np.abs(E01)/2/k/T_k))
    
    Lj = (phi_0/2/np.pi)**2/El/h

    return 2*hbar/Lj/Q_ind_fun(E01)*Temp_factor(E01)

def S_charge_coupled_impedance(E01):
    ReZ = 6.9e-3
    return 2*ReZ*h*E01*Temp_factor(E01)

def S_flux_bias_line(E01):
    Mut_imp = 1/1.6*1e3*phi_0 #21e-12 #
    Res = 26
    return Mut_imp**2*4*h*E01/Res*Temp_factor(E01)

def S_1f_flux(E01):
    A_phi = 1.18e-6*phi_0
    return A_phi**2/E01

def S_Andreev_1f(E01):

    A_r = 4.34e-6
    
    return A_r**2/E01*Temp_factor(E01)

# EIGENENERGIES OF THE BLOCHNIUM (take care of the factor /2 in the cos)
def eigenenergies_bloch(Ec,El,Ej,phi_ext, N = 200, eigvals = 0):
    phi_ZPF=(8.0 * Ec / El) ** 0.25
    N_op  = 1j * (destroy(N).dag() - destroy(N)) / phi_ZPF / np.sqrt(2)
    phi_op= (destroy(N).dag() + destroy(N)) * phi_ZPF *np.sqrt(2)

    delta = phi_op-phi_ext
    H = 4*Ec*N_op**2+0.5*El*(delta)**2-Ej*(phi_op/2).cosm()
    evals=H.eigenenergies(eigvals=eigvals)
    return evals

def eigensystem_fluxonium(Ec,El,Ej,phi_ext, N = 200, eigvals = 0):
    phi_ZPF=(8.0 * Ec / El) ** 0.25
    N_op  = 1j * (destroy(N).dag() - destroy(N)) / phi_ZPF / np.sqrt(2)
    phi_op= (destroy(N).dag() + destroy(N)) * phi_ZPF *np.sqrt(2)

    delta = phi_op-phi_ext
    H = 4*Ec*N_op**2+0.5*El*(delta)**2-Ej*(phi_op).cosm()
    evals,ekets=H.eigenstates(eigvals=eigvals)
    return evals,ekets

#######################################################
# THE FUNCTIONS BELOW ARE NOT USEFUL FOR ME ANYMORE.

def eigensystem_flux(r,phase_list, EC,EL,EDelta, N = 200,eigvals = 0, interpol = False):

    phi_ZPF=(8.0 * EC / EL) ** 0.25
    
    #definition of the quantum operator
    N_op  = 1j * (destroy(N).dag() - destroy(N)) / (phi_ZPF/ np.sqrt(2))
    phi_op= (destroy(N).dag() + destroy(N))  * phi_ZPF *np.sqrt(2)
    
    #creation of the range of the flux and the list that will save the matrix elements of different operators.
    evalsf_list=[]
    eketsf_list=[]

    for phi_ext in phase_list:

        evals,ekets = eigensystem_fbq(EC,EL,EDelta,phi_ext,r, N)
        
        #saving the eigenvalues and the matrix elements.
        evalsf_list.append(evals)
        eketsf_list.append(ekets)
        
    
    #making a mirror to the negative values of the external flux.
    phase_list = np.concatenate([-phase_list[::-1][0:-1],phase_list])
    evalsf_list = np.concatenate([evalsf_list[::-1][0:-1],evalsf_list])
        
    #converting the list to numpy array to facilitate the operations.
    evalsf_list = np.real(np.array(evalsf_list))
    
    #interpolation of the values is it's required.
    if interpol == True:
        x = np.linspace(phase_list[0],phase_list[-1],5001)
        ##DANGEROUS ZONE (we don't need to interpolate all the eigenvalues)
        evalsf = []
        for i in range(6): ##you can change the 6 if you want to see more levels.
            evalsf_list_f = interp1d(phase_list, evalsf_list[:,i],kind='cubic')
            evalsf.append(evalsf_list_f(x))
        evalsf = np.array(evalsf)
        evalsf_list = evalsf
        ##END OF THE DANGEROUS ZONE
        
        E01f_list_f = interp1d(phase_list, E01f_list,kind='cubic')
        E01f_list = E01f_list_f(x)

        phase_list = x 
    
    return phase_list, evalsf_list,eketsf_list

def eigensystem_r(phi_ext,r_list, EL,EC,EDelta, N = 200,eigvals = 0):

    phi_ZPF=(8.0 * EC / EL) ** 0.25
    
    #definition of the quantum operator
    N_op  = 1j * (destroy(N).dag() - destroy(N)) / (phi_ZPF * np.sqrt(2))
    phi_op= (destroy(N).dag() + destroy(N)) / np.sqrt(2) * phi_ZPF
    
    #creation of the range of the flux and the list that will save the matrix elements of different operators.
    evalsr_list=[]
    eketsr_list=[]

    for r in r_list:
        
        evals,ekets = eigensystem_fbq(EC,EL,EDelta,phi_ext,r, N)
        
        #saving the eigenvalues and the matrix elements.
        evalsr_list.append(evals)
        eketsr_list.append(ekets)
        
    #converting the list to numpy array to facilitate the operations.
    evalsr_list = np.real(np.array(evalsr_list))    
    
    return evalsr_list,eketsr_list




def evals_flux_blochnium(phase_list, EL,EC,EJ, N = 200, eigvals = 0):

    phi_ZPF=(8.0 * EC / EL) ** 0.25
    N_op  = 1j * (destroy(N).dag() - destroy(N)) / (phi_ZPF * np.sqrt(2))
    phi_op= (destroy(N).dag() + destroy(N)) / np.sqrt(2) * phi_ZPF
    
    #creation of the range of the flux and the list that will save the matrix elements of different operators.
    evalsf_list=[]
    eketsf_list=[]

    for phi_ext in tqdm(phase_list):
        
        delta = phi_op-phi_ext
        H = 4*EC*N_op**2+0.5*EL*phi_op**2-EJ*(delta).cosm() #Hamiltonian.
    
        evals,ekets =H.eigenstates(eigvals=eigvals) #saving the eigenvalues and eigenvectors of H at different external flux.
        
        #saving the eigenvalues and the matrix elements.
        evalsf_list.append(evals)
        eketsf_list.append(ekets)
        
    
    #making a mirror to the negative values of the external flux.
    phase_list = np.concatenate([-phase_list[::-1][0:-1],phase_list])
    evalsf_list = np.concatenate([evalsf_list[::-1][0:-1],evalsf_list])
        
    #converting the list to numpy array to facilitate the operations.
    evalsf_list = np.real(np.array(evalsf_list))
    
    return phase_list, evalsf_list,eketsf_list



# WAVEFUNCTION (\varphi) FOR A WAVEFUNCTION "phi" GIVEN AS A KET. 
def wavefunction_phi(EC,EL,phi_list,phi):
    wfunc = np.zeros(len(phi_list),dtype = complex)
    for n in range(45):
        wfunc = wfunc + phi_n(EC,EL,n,phi_list)*phi.full()[n,0]
    wfunc
    return np.real(wfunc)

# PROBABILITY (\varphi) FOR A WAVEFUNCTION "phi" GIVEN AS A KET. 
def probability_phi(EC,EL,phi_list,phi):
    wfunc = np.zeros(len(phi_list),dtype = complex)
    for n in range(70):
        wfunc = wfunc + phi_n(EC,EL,n,phi_list)*phi.full()[n,0]
    prob = np.abs(wfunc)**2
    return prob

def probability_phi_fbq(EC,EL,phi_list,phi):
    wfunc = np.zeros(len(phi_list),dtype = complex)
    for n in range(70):
        wfunc = wfunc + phi_n(EC,EL,n,phi_list)*phi.full()[2*n,0]
    prob = np.abs(wfunc)**2
    return prob

def probability_phi_fbq1(EC,EL,phi_list,phi):
    wfunc = np.zeros(len(phi_list),dtype = complex)
    for n in range(65):
        wfunc = wfunc + phi_n(EC,EL,n,phi_list)*phi.full()[2*n+1,0]
    prob = np.abs(wfunc)**2
    return prob