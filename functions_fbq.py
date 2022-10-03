from qutip import *
from tqdm import tqdm
from scipy import special
import numpy as np
from scipy.interpolate import interp1d
from scipy.special import hermite


# EIGENVALUES OF THE FERMIONIC-BOSONIC QUBIT
def eigensystem_flux(r,phase_list, EL,EC,EDelta, N = 200,eigvals = 0, interpol = False):

    phi_ZPF=(8.0 * EC / EL) ** 0.25
    
    #definition of the quantum operator
    N_op  = 1j * (destroy(N).dag() - destroy(N)) / (phi_ZPF * np.sqrt(2))
    phi_op= (destroy(N).dag() + destroy(N)) / np.sqrt(2) * phi_ZPF
    
    #creation of the range of the flux and the list that will save the matrix elements of different operators.
    evalsf_list=[]
    eketsf_list=[]

    for phi_ext in tqdm(phase_list):
        
        delta = phi_op-phi_ext
        ReZ = EDelta*((delta/2).cosm()*(r*delta/2).cosm()+r*(delta/2).sinm()*(r*delta/2).sinm()) #Re(Z) of the Hamiltonian
        ImZ = EDelta*(-(delta/2).cosm()*(r*delta/2).sinm()+r*(delta/2).sinm()*(r*delta/2).cosm()) ##Im(Z) of the Hamiltonian
        H = 4*EC*tensor(N_op,qeye(2))**2+0.5*EL*tensor(phi_op,qeye(2))**2-tensor(ReZ,sigmaz())+tensor(ImZ,sigmax()) #Hamiltonian.
        evals,ekets=H.eigenstates(eigvals=eigvals) #saving the eigenvalues and eigenvectors of H at different external flux.
        
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

    for r in tqdm(r_list):
        
        delta = phi_op-phi_ext
        ReZ = EDelta*((delta/2).cosm()*(r*delta/2).cosm()+r*(delta/2).sinm()*(r*delta/2).sinm()) #Re(Z) of the Hamiltonian
        ImZ = EDelta*(-(delta/2).cosm()*(r*delta/2).sinm()+r*(delta/2).sinm()*(r*delta/2).cosm()) ##Im(Z) of the Hamiltonian
        H = 4*EC*tensor(N_op,qeye(2))**2+0.5*EL*tensor(phi_op,qeye(2))**2-tensor(ReZ,sigmaz())+tensor(ImZ,sigmax()) #Hamiltonian.
        evals,ekets=H.eigenstates(eigvals=eigvals) #saving the eigenvalues and eigenvectors of H at different external flux.
        
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
    
    return evalsf_list,eketsf_list

def phi_n(EC,EL, n: int,phi_list):
    phi_ZPF=(8.0 * EC / EL) ** 0.25      
    return 1/np.sqrt(phi_ZPF)/np.sqrt(2**n *1.0* np.math.factorial(n)) * np.exp(-(phi_list/phi_ZPF)**2 / 2.0) * np.polyval(hermite(n), (phi_list/phi_ZPF))

def probability_phi(EC,EL,phi_list,phi):
    wfunc = np.zeros(len(phi_list),dtype = complex)
    for n in range(70):
        wfunc = wfunc + phi_n(EC,EL,n,phi_list)*phi.full()[n,0]
    prob = np.abs(wfunc)**2
    return prob

def probability_phi_fbq(EC,EL,phi_list,phi):
    wfunc = np.zeros(len(phi_list),dtype = complex)
    for n in tqdm(range(70)):
        wfunc = wfunc + phi_n(EC,EL,n,phi_list)*phi.full()[2*n,0]
    prob = np.abs(wfunc)**2
    return prob

def probability_phi_fbq1(EC,EL,phi_list,phi):
    wfunc = np.zeros(len(phi_list),dtype = complex)
    for n in tqdm(range(70)):
        wfunc = wfunc + phi_n(EC,EL,n,phi_list)*phi.full()[2*n+1,0]
    prob = np.abs(wfunc)**2
    return prob

def potentials(r, phase_range_pot,phi_ext,EL = 0.05, EDelta = 20):
    E_phi_p = 0.5*EL*phase_range_pot**2+EDelta*np.sqrt(np.cos((phase_range_pot-phi_ext)/2)**2+(r**2)*np.sin((phase_range_pot-phi_ext)/2)**2)

    E_phi_m = 0.5*EL*phase_range_pot**2-EDelta*np.sqrt(np.cos((phase_range_pot-phi_ext)/2)**2+(r**2)*np.sin((phase_range_pot-phi_ext)/2)**2)
    return E_phi_m, E_phi_p

def potential_blochnium(phase_list,phi_ext,EL = 0.05, EJ = 20):
    pot = 0.5*EL*phase_list**2 - EJ*np.cos(phase_list - phi_ext)
    return pot