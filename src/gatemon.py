import numpy as np
from .qubit_base import QubitBase
from scipy.linalg import sinm, cosm, sqrtm, solve
from .operators import destroy, creation
from typing import Union, Tuple, Dict, Any, Iterable, Optional
import matplotlib.pyplot as plt
from scipy.linalg import pinv
from scipy.integrate import quad


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
        
        # self.phi_grid = np.linspace(- 8 * np.pi, 8 * np.pi, self.dimension, endpoint=False)
        # self.dphi = self.phi_grid[1] - self.phi_grid[0] 
        
        super().__init__(dimension = self.dimension)
        
    def n_operator(self):
        """
        Generate the number operator matrix \hat{n} in a (2N+1)-dimensional truncated space.
        
        Returns:
            numpy.ndarray: Diagonal matrix representing the number operator.
        """
        n_values = np.arange(-self.n_cut, self.n_cut+1)
        return np.diag(n_values)
    
    def phase_operator(self):
        pass
    
    def cos_kphi_operator(self, k):
        """
        Generate the matrix representation of the \cos(k\hat{\phi}) operator in the number basis.
        
        Parameters:
            k (int): The integer multiplier of \hat{\phi}.
        
        Returns:
            numpy.ndarray: Matrix representation of \cos(k\hat{\phi}).
        """
        cos_kphi = np.zeros((self.dimension, self.dimension))
        indices = np.arange(self.dimension)
        
        mask_up = indices + k < self.dimension
        mask_down = indices - k >= 0
        
        cos_kphi[indices[mask_up], indices[mask_up] + k] = 0.5
        cos_kphi[indices[mask_down], indices[mask_down] - k] = 0.5
        
        return cos_kphi
    
    def junction_potential(self):
        phase_op = self.phase_operator()
        
        junction_term = 0
        def f(phi, T, Delta, phi_ext):
            return -Delta * np.sqrt(1 - T * np.sin((phi - phi_ext)/2)**2)
        
        def A0(T, Delta):
            integral, error = quad(lambda x: f(x, T, Delta, 0), 0, np.pi)
            return integral / np.pi

        # Cálculo numérico de A_k para k >= 1
        def A_k(k, T, Delta):
            integral, error = quad(lambda x: f(x, T, Delta, 0) * np.cos(k*x), 0, np.pi)
            return 2 * integral / np.pi
        
        A_coeffs = [A0(self.T, self.Delta)] + [A_k(k, self.T, self.Delta) for k in range(1, self.num_coef + 1)]
        
        for k in range(self.num_coef):
            junction_term += A_coeffs[k] * self.cos_kphi_operator(k)
            
        return junction_term
    
    def hamiltonian(self):
        n_op = self.n_operator() - self.ng * np.eye(self.dimension)      
        kinetic_term = 4 * self.Ec * (n_op @ n_op)
        junction_term = self.junction_potential()

        return kinetic_term + junction_term
            
    def potential(self, phi: Union[float, np.ndarray]):
        raise NotImplementedError("Potential method not implemented for this class.")
        
    def wavefunction(self, which: int = 0, phi_grid: np.ndarray = None, esys: Tuple[np.ndarray, np.ndarray] = None) -> Dict[str, Any]:
        """
        Returns a wave function in the phi basis.

        Parameters
        ----------
        which : int, optional
            Index of desired wave function (default is 0).
        phi_grid : np.ndarray, optional
            Custom grid for phi; if None, a default grid is used.

        Returns
        -------
        Dict[str, Any]
            Wave function data containing basis labels, amplitudes, and energy.
        """
        if esys is None:
            evals_count = max(which + 1, 3)
            evals, evecs = self.eigensys(evals_count)
        else:
            evals, evecs = esys
            
        dim = self.n_cut
                        
        if phi_grid is None:
            phi_grid = np.linspace(-5 * np.pi, 5 * np.pi, 151)

        phi_basis_labels = phi_grid
        wavefunc_osc_basis_amplitudes = evecs[:, which]
        phi_wavefunc_amplitudes = np.zeros_like(phi_grid, dtype=np.complex_)
        phi_osc = self.phi_osc()
        
        for n in range(dim):
            phi_wavefunc_amplitudes += wavefunc_osc_basis_amplitudes[n] * self.harm_osc_wavefunction(n, phi_basis_labels, phi_osc)

        return {
            "basis_labels": phi_basis_labels,
            "amplitudes": phi_wavefunc_amplitudes,
            "energy": evals[which]
        }

    def plot_wavefunction(
        self, 
        which: Union[int, Iterable[int]] = 0, 
        phi_grid: np.ndarray = None, 
        esys: Tuple[np.ndarray, np.ndarray] = None, 
        scaling: Optional[float] = None,
        **kwargs
        ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the wave function in the phi basis.

        Parameters
        ----------
        which : Union[int, Iterable[int]], optional
            Index or indices of desired wave function(s) (default is 0).
        phi_grid : np.ndarray, optional
            Custom grid for phi; if None, a default grid is used.
        esys : Tuple[np.ndarray, np.ndarray], optional
            Precomputed eigenvalues and eigenvectors.
        **kwargs
            Additional arguments for plotting. Can include:
            - fig_ax: Tuple[plt.Figure, plt.Axes], optional
                Figure and axes to use for plotting. If not provided, a new figure and axes are created.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            The figure and axes of the plot.
        """
        if isinstance(which, int):
            which = [which]
            
        potential = self.potential(phi=phi_grid)

        fig_ax = kwargs.get("fig_ax")
        if fig_ax is None:
            fig, ax = plt.subplots()
            fig.suptitle(self._generate_suptitle())
        else:
            fig, ax = fig_ax
        
        ax.plot(phi_grid/2/np.pi, potential, color='black', label='Potential')

        for idx in which:
            wavefunc_data = self.wavefunction(which=idx, phi_grid=phi_grid, esys=esys)
            phi_basis_labels = wavefunc_data["basis_labels"]
            wavefunc_amplitudes = wavefunc_data["amplitudes"]
            wavefunc_energy = wavefunc_data["energy"]

            ax.plot(
                phi_basis_labels/2/np.pi,
                wavefunc_energy + scaling * (wavefunc_amplitudes.real + wavefunc_amplitudes.imag),
                # color="blue",
                label=rf"$\Psi_{idx}$"
                )

        ax.set_xlabel(r"$\Phi / \Phi_0$")
        ax.set_ylabel(r"$\psi(\varphi)$, Energy [GHz]")
        ax.legend()
        ax.grid(True)

        return fig, ax
        

        