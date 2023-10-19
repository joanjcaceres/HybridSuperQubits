import numpy as np
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor
import copy
from qutip import Qobj, destroy, tensor, qeye, sigmaz, sigmay

class Ferbo:
    def __init__(self, Ec, El, Delta, r = 0.05, phi_ext = 0):
        self.Ec = Ec
        self.El = El
        self.Delta = Delta
        self.r = r
        self.phi_ext = phi_ext

    def charge_number_operator(self, dimension = 100):
        phi_zpf = (8.0 * self.Ec / self.El) ** 0.25
        n_operator  = 1j * (destroy(dimension).dag() - destroy(dimension)) / phi_zpf /np.sqrt(2)
        return n_operator
    
    def phase_operator(self, dimension = 100):
        phi_zpf = (8.0 * self.Ec / self.El) ** 0.25
        phase_operator = (destroy(dimension).dag() + destroy(dimension)) * phi_zpf/ np.sqrt(2)
        return phase_operator

    def get_hamiltonian(self, dimension = 100) -> Qobj:

        phi_zpf = (8.0 * self.Ec / self.El) ** 0.25
        n_operator = 1j * (destroy(dimension).dag() - destroy(dimension)) / phi_zpf /np.sqrt(2)
        phase_operator = (destroy(dimension).dag() + destroy(dimension)) * phi_zpf/ np.sqrt(2)
        ReZ = (phase_operator/2).cosm()*(self.r*phase_operator/2).cosm()+self.r*(phase_operator/2).sinm()*(self.r*phase_operator/2).sinm()
        ImZ = -(phase_operator/2).cosm()*(self.r*phase_operator/2).sinm()+self.r*(phase_operator/2).sinm()*(self.r*phase_operator/2).cosm()
        
        delta = phase_operator - self.phi_ext
        hamiltonian = 4*self.Ec*tensor(n_operator**2,qeye(2)) + 0.5*self.El*tensor((delta)**2,qeye(2)) + self.Delta*(tensor(ReZ,sigmaz())+tensor(ImZ,sigmay()))
        return hamiltonian
    
    def get_eigenenergies(self, dimension = 100, eigvals=0) -> np.ndarray:
        hamiltonian = self.get_hamiltonian(dimension = dimension)
        eigenenergies = hamiltonian.eigenenergies(eigvals=eigvals)
        return np.real(eigenenergies)

    def get_eigenenergies_vs_phase(self, dimension=100, eigvals=6, phi_ext_array=np.linspace(-2*np.pi, 2*np.pi, 100)) -> np.ndarray:
        eigenenergies = []
        
        def task(phi_ext):
            local_self = copy.deepcopy(self)
            local_self.phi_ext = phi_ext
            return local_self.get_eigenenergies(dimension=dimension, eigvals=eigvals)
        
        with ThreadPoolExecutor() as executor:
            eigenenergies = list(executor.map(task, phi_ext_array))
            
        return np.array(eigenenergies)
    
    def get_eigenenergies_vs_r(self, dimension = 100, eigvals=6, r_array = np.linspace(0,1,100)) -> np.ndarray:
        eigenenergies = []
        for r in r_array:
            self.r = r
            eigenenergies.append(self.get_eigenenergies(dimension = dimension, eigvals=eigvals))
        return np.array(eigenenergies)
    
    def get_eigensystem(self, dimension = 100, eigvals=0):
        hamiltonian = self.get_hamiltonian(dimension=dimension)
        eigenenergies, eigenstates = hamiltonian.eigenstates(eigvals=eigvals)
        return eigenenergies, np.real(eigenstates)
    
    def get_operator_matrix_element(self, operator: Qobj, eigvals=0, i=0, j=1):
        hamiltonian = self.get_hamiltonian(dimension=100)
        eigenstates = hamiltonian.eigenstates(eigvals=eigvals)[1]
        matrix_element = operator.matrix_element(eigenstates[i],eigenstates[j])
        return matrix_element
    
    def get_operator_matrix_element_vs_external_phase(self, operator: Qobj, eigvals=0, i=0, j=1, phi_ext_array = np.linspace(-2*np.pi,2*np.pi,100)):
        matrix_elements = []
        for phi_ext in phi_ext_array:
            self.phi_ext = phi_ext
            matrix_elements.append(self.get_operator_matrix_element(operator=operator, eigvals=eigvals, i=i, j=j))
        return np.array(matrix_elements)
    
    def get_operator_matrix_element_vs_r(self, operator: Qobj, eigvals=0, i=0, j=1, r_array = np.linspace(0,1,100)):
        matrix_elements = []
        for r in r_array:
            self.r = r
            matrix_elements.append(self.get_operator_matrix_element(operator=operator, eigvals=eigvals, i=i, j=j))
        return np.array(matrix_elements)


