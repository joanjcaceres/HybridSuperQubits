import numpy as np
from .qubit_base import QubitBase
from scipy.linalg import sinm, cosm
from .operators import destroy, creation
from typing import Union, Tuple, Dict, Any, Iterable, Optional
import matplotlib.pyplot as plt
from scipy.integrate import quad



class Gatemonium(QubitBase):
    
    PARAM_LABELS = {
        'Ec': r'$E_C$',
        'El': r'$E_L$',
        'Delta': r'$\Delta$',
        'T': r'$T$',
        'phase': r'$\Phi_{ext} / \Phi_0$'
    }
    
    OPERATOR_LABELS = {
    'n_operator': r'\hat{n}',
    'phase_operator': r'\hat{\phi}',
    'd_hamiltonian_d_ng': r'\partial \hat{H} / \partial n_g',
    'd_hamiltonian_d_phase': r'\partial \hat{H} / \partial \phi_{{ext}}',
    }
    
    def __init__(self, Ec, El, Delta, T, phase, dimension, flux_grouping):
        
        if flux_grouping not in ['L', 'ABS']:
            raise ValueError("Invalid flux grouping; must be 'L' or 'ABS'.")
        
        self.Ec = Ec
        self.El = El
        self.Delta = Delta
        self.T = T
        self.phase = phase
        self.dimension = dimension
        self.flux_grouping = flux_grouping
        self.num_coef = 4
        
        # self.phi_grid = np.linspace(- 8 * np.pi, 8 * np.pi, self.dimension, endpoint=False)
        # self.dphi = self.phi_grid[1] - self.phi_grid[0] 
        
        super().__init__(dimension = dimension)
        
    @property
    def phase_zpf(self) -> float:
        """
        Returns the zero-point fluctuation of the phase.

        Returns
        -------
        float
            Zero-point fluctuation of the phase.
        """
        return (2 * self.Ec / self.El) ** 0.25
    
    @property
    def n_zpf(self) -> float:
        """
        Returns the zero-point fluctuation of the charge number.

        Returns
        -------
        float
            Zero-point fluctuation of the charge number.
        """
        return 1/2 * (self.El / 2 / self.Ec) ** 0.25
        
    def phi_osc(self) -> float:
        """
        Returns the oscillator length for the LC oscillator composed of the inductance and capacitance.

        Returns
        -------
        float
            Oscillator length.
        """
        return (8.0 * self.Ec / self.El) ** 0.25
        
    def n_operator(self):
        dimension = self.dimension
        return 1j/2 * (self.El/2/self.Ec)**0.25 * (creation(dimension) - destroy(dimension))
    
    def phase_operator(self):
        dimension = self.dimension
        return (2*self.Ec/self.El)**0.25 * (creation(dimension) + destroy(dimension))
    
    def junction_potential(self):
        phase_op = self.phase_operator()
        
        if self.flux_grouping == 'ABS':
            phase_op -= self.phase * np.eye(self.dimension)
        
        junction_term = 0
        def f(phi, T, Delta):
            return - Delta * np.sqrt(1 - T * np.sin(phi/2)**2)
        
        def A_k(k, T, Delta):
            integral, error = quad(lambda x: f(x, T, Delta) * np.cos(k*x), 0, np.pi)
            return 2 * integral / np.pi
        
        A_coeffs = [A_k(k, self.T, self.Delta) for k in range(self.num_coef + 1)]
        
        junction_term = A_coeffs[0] / 2  * np.eye(self.dimension)
        for k in range(1, self.num_coef + 1):
            junction_term += A_coeffs[k] * cosm(k * phase_op)            
        return junction_term
    
    def hamiltonian(self):
        n_op = self.n_operator()
        phase_op = self.phase_operator()

        if self.flux_grouping == 'L':
            phase_op += self.phase * np.eye(self.dimension)
        
        kinetic_term = 4 * self.Ec * (n_op @ n_op)
        inductive_term = 0.5 * self.El * (phase_op @ phase_op)
        junction_term = self.junction_potential()

        return kinetic_term + inductive_term + junction_term
    
    def d_hamiltonian_d_phase(self) -> np.ndarray:
        phase_op = self.phase_operator()
        ext_phase = self.phase * np.eye(self.dimension)
        
        if self.flux_grouping == 'L':
            return self.El * (phase_op + ext_phase)
        elif self.flux_grouping == 'ABS':
            phase_op = self.phase_operator() - self.phase * np.eye(self.dimension)
            # numerator = -self.T * self.Delta * sinm(phase_op - ext_phase)
            # sin_op = sinm((phase_op - ext_phase) / 2)
            # denominator = 4 * sqrtm(np.eye(self.dimension) - self.T * sin_op.conj().T @ sin_op)
            
            # return solve(numerator , denominator)
            
            def f(phi, T, Delta):
                return -Delta * np.sqrt(1 - T * np.sin(phi/2)**2)
            
            def A_k(k, T, Delta):
                integral, error = quad(lambda x: f(x, T, Delta) * np.cos(k*x), 0, np.pi)
                return 2 * integral / np.pi
            
            A_coeffs = [A_k(k, self.T, self.Delta) for k in range(0, self.num_coef + 1)]
                    
            dH_dPhi = 0
            for k in range(self.num_coef):
                dH_dPhi += A_coeffs[k] * k * sinm(k * phase_op)
                
            return dH_dPhi
            
            
        
    def potential(self, phi: Union[float, np.ndarray]):
        
        phi_array = np.atleast_1d(phi)
        
        if self.flux_grouping == 'L':
            inductive_term = 0.5 * self.El * (phi_array + self.phase)**2
            junction_term = -self.Delta * np.sqrt(1 - self.T * np.sin(phi_array/2)**2)
        elif self.flux_grouping == 'ABS':
            inductive_term = 0.5 * self.El * phi_array**2
            junction_term = -self.Delta * np.sqrt(1 - self.T * np.sin((phi_array - self.phase)/2)**2)
            
        return inductive_term + junction_term
        
    def wavefunction(
        self,
        which: int = 0, 
        phi_grid: np.ndarray = None, 
        esys: Tuple[np.ndarray, np.ndarray] = None
        ) -> Dict[str, Any]:
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
            
        dim = self.dimension
                        
        if phi_grid is None:
            phi_grid = np.linspace(-5 * np.pi, 5 * np.pi, 151)

        phi_basis_labels = phi_grid
        wavefunc_osc_basis_amplitudes = evecs[:, which]
        phi_wavefunc_amplitudes = np.zeros_like(phi_grid, dtype=np.complex128)
        
        for n in range(dim):
            phi_wavefunc_amplitudes += wavefunc_osc_basis_amplitudes[n] * self.harm_osc_wavefunction(n, phi_basis_labels, self.phase_zpf)

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
        

        