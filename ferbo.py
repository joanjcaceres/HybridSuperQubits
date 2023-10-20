import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from qutip import Qobj, destroy, tensor, qeye, sigmaz, sigmay

class Ferbo:
    def __init__(self, Ec, El, Delta, r = 0.05, phi_ext = 0, dimension = 100):
        self.Ec = Ec
        self.El = El
        self.Delta = Delta
        self.r = r
        self.phi_ext = phi_ext
        self.dimension = dimension

        self.phi_zpf = (8.0 * self.Ec / self.El) ** 0.25
        self.charge_number_operator = 1j * (destroy(self.dimension).dag() - destroy(self.dimension)) / self.phi_zpf /np.sqrt(2)
        self.phase_operator = (destroy(self.dimension).dag() + destroy(self.dimension)) * self.phi_zpf/ np.sqrt(2)
        self.ReZ = (self.phase_operator/2).cosm()*(self.r*self.phase_operator/2).cosm()+self.r*(self.phase_operator/2).sinm()*(self.r*self.phase_operator/2).sinm()
        self.ImZ = -(self.phase_operator/2).cosm()*(self.r*self.phase_operator/2).sinm()+self.r*(self.phase_operator/2).sinm()*(self.r*self.phase_operator/2).cosm()

    def get_hamiltonian(self) -> Qobj:
        delta = self.phase_operator - self.phi_ext
        hamiltonian = 4*self.Ec*tensor(self.charge_number_operator**2,qeye(2)) + 0.5*self.El*tensor((delta)**2,qeye(2)) + self.Delta*(tensor(self.ReZ,sigmaz())+tensor(self.ImZ,sigmay()))
        return hamiltonian
    
    def get_eigenenergies(self, phi_ext, eigvals=0) -> np.ndarray:
        delta = self.phase_operator - phi_ext
        hamiltonian = 4*self.Ec*tensor(self.charge_number_operator**2,qeye(2)) + 0.5*self.El*tensor((delta)**2,qeye(2)) + self.Delta*(tensor(self.ReZ,sigmaz())+tensor(self.ImZ,sigmay()))
        eigenenergies = hamiltonian.eigenenergies(eigvals=eigvals)
        return np.real(eigenenergies)
    
    def get_eigenenergies_vs_external_phase(self, phi_ext_array=np.linspace(-2*np.pi, 2*np.pi, 100), eigvals=6, plot=True) -> np.ndarray:
        eigenenergies = []
        for phi_ext in tqdm(phi_ext_array):
            delta = self.phase_operator - phi_ext
            hamiltonian = 4*self.Ec*tensor(self.charge_number_operator**2,qeye(2)) + 0.5*self.El*tensor((delta)**2,qeye(2)) + self.Delta*(tensor(self.ReZ,sigmaz())+tensor(self.ImZ,sigmay()))
            eigenenergies.append(np.real(hamiltonian.eigenenergies(eigvals=eigvals)))

        eigenenergies = np.array(eigenenergies)
        if plot == True:
            plot_eigenenergies(phi_ext_array, eigenenergies, r"External phase  $\varphi_{ext}$ (rad)")
        return eigenenergies
    
    def get_eigenenergies_vs_r(self, r_array = np.linspace(0,1,100), eigvals=6, plot=True) -> np.ndarray:
        eigenenergies = []
        delta = self.phase_operator - self.phi_ext
        for r in tqdm(r_array):
            ReZ = (self.phase_operator/2).cosm()*(r*self.phase_operator/2).cosm()+r*(self.phase_operator/2).sinm()*(r*self.phase_operator/2).sinm()
            ImZ = -(self.phase_operator/2).cosm()*(r*self.phase_operator/2).sinm()+r*(self.phase_operator/2).sinm()*(r*self.phase_operator/2).cosm()
            hamiltonian = 4*self.Ec*tensor(self.charge_number_operator**2,qeye(2)) + 0.5*self.El*tensor((delta)**2,qeye(2)) + self.Delta*(tensor(ReZ,sigmaz())+tensor(ImZ,sigmay()))
            eigenenergies.append(np.real(hamiltonian.eigenenergies(eigvals=eigvals)))

        eigenenergies = np.array(eigenenergies)
        if plot == True:
            plot_eigenenergies(r_array, eigenenergies, r"$r$")
        return eigenenergies
    
    def get_eigenenergies_vs_Ec(self, Ec_array, eigvals=6, plot=True) -> np.ndarray:
        eigenenergies = []
        for Ec in tqdm(Ec_array):
            phi_zpf = (8.0 * Ec / self.El) ** 0.25
            charge_number_operator = 1j * (destroy(self.dimension).dag() - destroy(self.dimension)) / phi_zpf /np.sqrt(2)
            phase_operator = (destroy(self.dimension).dag() + destroy(self.dimension)) * phi_zpf/ np.sqrt(2)
            ReZ = (phase_operator/2).cosm()*(self.r*phase_operator/2).cosm()+self.r*(phase_operator/2).sinm()*(self.r*phase_operator/2).sinm()
            ImZ = -(phase_operator/2).cosm()*(self.r*phase_operator/2).sinm()+self.r*(phase_operator/2).sinm()*(self.r*phase_operator/2).cosm()
            delta = phase_operator - self.phi_ext
            hamiltonian = 4*Ec*tensor(charge_number_operator**2,qeye(2)) + 0.5*self.El*tensor((delta)**2,qeye(2)) + self.Delta*(tensor(ReZ,sigmaz())+tensor(ImZ,sigmay()))
            eigenenergies.append(np.real(hamiltonian.eigenenergies(eigvals=eigvals)))

        eigenenergies = np.array(eigenenergies)
        if plot == True:
            plot_eigenenergies(Ec_array, eigenenergies, r"$E_c$ (GHz)")
        return eigenenergies
    
    def get_eigenenergies_vs_Delta(self, Delta_array, eigvals=6, plot=True) -> np.ndarray:
        eigenenergies = []
        for Delta in tqdm(Delta_array):
            phi_zpf = (8.0 * self.Ec / self.El) ** 0.25
            charge_number_operator = 1j * (destroy(self.dimension).dag() - destroy(self.dimension)) / phi_zpf /np.sqrt(2)
            phase_operator = (destroy(self.dimension).dag() + destroy(self.dimension)) * phi_zpf/ np.sqrt(2)
            ReZ = (phase_operator/2).cosm()*(self.r*phase_operator/2).cosm()+self.r*(phase_operator/2).sinm()*(self.r*phase_operator/2).sinm()
            ImZ = -(phase_operator/2).cosm()*(self.r*phase_operator/2).sinm()+self.r*(phase_operator/2).sinm()*(self.r*phase_operator/2).cosm()
            delta = phase_operator - self.phi_ext
            hamiltonian = 4*self.Ec*tensor(charge_number_operator**2,qeye(2)) + 0.5*self.El*tensor((delta)**2,qeye(2)) + Delta*(tensor(ReZ,sigmaz())+tensor(ImZ,sigmay()))
            eigenenergies.append(np.real(hamiltonian.eigenenergies(eigvals=eigvals)))

        eigenenergies = np.array(eigenenergies)
        if plot == True:
            plot_eigenenergies(Delta_array, eigenenergies, r"$\Delta_{eff}$ (GHz)")
        return eigenenergies

    def get_eigenenergies_vs_El(self, El_array, eigvals=6, plot=True) -> np.ndarray:
        eigenenergies = []
        for El in tqdm(El_array):
            phi_zpf = (8.0 * self.Ec / El) ** 0.25
            charge_number_operator = 1j * (destroy(self.dimension).dag() - destroy(self.dimension)) / phi_zpf /np.sqrt(2)
            phase_operator = (destroy(self.dimension).dag() + destroy(self.dimension)) * phi_zpf/ np.sqrt(2)
            ReZ = (phase_operator/2).cosm()*(self.r*phase_operator/2).cosm()+self.r*(phase_operator/2).sinm()*(self.r*phase_operator/2).sinm()
            ImZ = -(phase_operator/2).cosm()*(self.r*phase_operator/2).sinm()+self.r*(phase_operator/2).sinm()*(self.r*phase_operator/2).cosm()
            delta = phase_operator - self.phi_ext
            hamiltonian = 4*self.Ec*tensor(charge_number_operator**2,qeye(2)) + 0.5*El*tensor((delta)**2,qeye(2)) + self.Delta*(tensor(ReZ,sigmaz())+tensor(ImZ,sigmay()))
            eigenenergies.append(np.real(hamiltonian.eigenenergies(eigvals=eigvals)))

        eigenenergies = np.array(eigenenergies)
        if plot == True:
            plot_eigenenergies(El_array, eigenenergies, r"$E_L$ (GHz)")
        return eigenenergies
    
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

def plot_eigenenergies(x_value, eigenenergies, x_label):
    for i in range(eigenenergies.shape[1]):
        plt.plot(x_value, eigenenergies[:,i])
    plt.xlabel(x_label)
    plt.ylabel("Eigenenergies (GHz)")
    plt.show()

