import numpy as np
from .qubit_base import QubitBase
from typing import Union, Tuple, Dict, Any, Iterable, Optional
import matplotlib.pyplot as plt
from scipy.integrate import quad
from .utilities import cos_kphi_operator


class Gatemon(QubitBase):
    
    PARAM_LABELS = {
        'Ec': r'$E_C$',
        'Delta': r'$\Delta$',
        'T': r'$T$',
        'ng': r'$n_g$'
        }
    
    OPERATOR_LABELS = {
    'n_operator': r'\hat{n}',
    'phase_operator': r'\hat{\phi}',
    'd_hamiltonian_d_ng': r'\partial \hat{H} / \partial n_g',
    }
    
    def __init__(self, Ec, Delta, T, ng, n_cut):
        
        self.Ec = Ec
        self.Delta = Delta
        self.T = T
        self.ng = ng
        self.n_cut = n_cut
        self.num_coef = 4
        self.dimension = 2 * n_cut + 1
        
        super().__init__(dimension = self.dimension)
        
    def n_operator(self):
        """
        Generate the number operator matrix \hat{n} in a (2N+1)-dimensional truncated space.
        
        Returns:
            numpy.ndarray: Diagonal matrix representing the number operator.
        """
        n_values = np.arange(-self.n_cut, self.n_cut+1)
        return np.diag(n_values)
    
    def junction_potential(self):
        
        def f(phi, T, Delta):
            return -Delta * np.sqrt(1 - T * np.sin(phi/2)**2)

        # Cálculo numérico de A_k para k >= 1
        def A_k(k, T, Delta):
            integral, error = quad(lambda x: f(x, T, Delta) * np.cos(k*x), 0, np.pi)
            return 2 * integral / np.pi
        
        A_coeffs = [A_k(k, self.T, self.Delta) for k in range(self.num_coef + 1)]
        
        junction_term = A_coeffs[0] / 2 * np.eye(self.dimension, dtype=np.complex128)
        for k in range(1, self.num_coef + 1 ):
            junction_term += A_coeffs[k] * cos_kphi_operator(k, self.dimension)
            
        return junction_term
    
    def hamiltonian(self):
        n_op = self.n_operator() - self.ng * np.eye(self.dimension)      
        kinetic_term = 4 * self.Ec * (n_op @ n_op)
        junction_term = self.junction_potential()

        return kinetic_term + junction_term
            
    def potential(self, phi: Union[float, np.ndarray]):
        raise NotImplementedError("Potential method not implemented for this class.")

    def d_hamiltonian_d_ng(self):
        n_op = self.n_operator() - self.ng * np.eye(self.dimension)
        return - 8 * self.Ec * n_op
        
    def wavefunction(self, which: int = 0, phi_grid: np.ndarray = None, esys: Tuple[np.ndarray, np.ndarray] = None) -> Dict[str, Any]:
        raise NotImplementedError("Wavefunction method not implemented for this class.")
    
    def plot_wavefunction(
        self, 
        which: Union[int, Iterable[int]] = 0, 
        phi_grid: np.ndarray = None, 
        esys: Tuple[np.ndarray, np.ndarray] = None, 
        scaling: Optional[float] = None,
        **kwargs
        ) -> Tuple[plt.Figure, plt.Axes]:
        raise NotImplementedError("Wavefunction plotting not implemented for this class.")
        

        