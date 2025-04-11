import matplotlib.pyplot as plt
import numpy as np
from .qubit_base import QubitBase
from scipy.linalg import cosm, sinm
from typing import Any, Dict, Optional, Tuple, Union, Iterable
from .operators import destroy, creation

class Fluxonium(QubitBase):
    PARAM_LABELS = {
        'Ec': r'$E_C$',
        'El': r'$E_L$',
        'Ej': r'$E_J$',
        'phase': r'$\Phi_{\mathrm{ext}} / \Phi_0$'
    }
    
    OPERATOR_LABELS = {
    'n_operator': r'\hat{n}',
    'phase_operator': r'\hat{\phi}',
    'd_hamiltonian_d_ng': r'\partial \hat{H} / \partial n_g',
    'd_hamiltonian_d_phase': r'\partial \hat{H} / \partial \phi_{{ext}}',
    'd_hamiltonian_d_EL': r'\partial \hat{H} / \partial E_L',
    }
    
    def __init__(self, Ec, El, Ej, phase, dimension, flux_grouping: str = 'EL'):
        """
        Initializes the Ferbo class with the given parameters.

        Parameters
        ----------
        Ec : float
            Charging energy.
        El : float
            Inductive energy.
        Ej : float
            Josephson energy.
        phase : float
            External magnetic phase.
        dimension : int
            Dimension of the Hilbert space.
        flux_grouping : str, optional
            Flux grouping ('EL' or 'EJ') (default is 'EL').
        """
        if flux_grouping not in ['EL', 'EJ']:
            raise ValueError("Invalid flux grouping; must be 'EL' or 'EJ'.")
        
        self.Ec = Ec
        self.El = El
        self.Ej = Ej
        self.phase = phase
        self.dimension = dimension
        self.flux_grouping = flux_grouping
        super().__init__(self.dimension)
        
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
    
    def n_operator(self) -> np.ndarray:
        """
        Returns the charge number operator.

        Returns
        -------
        np.ndarray
            The charge number operator.
        """
        return 1j * self.n_zpf * (creation(self.dimension) - destroy(self.dimension))
    
    def phase_operator(self) -> np.ndarray:
        """
        Returns the total phase operator.

        Returns
        -------
        np.ndarray
            The total phase operator.
        """
        return self.phase_zpf * (creation(self.dimension) + destroy(self.dimension)) 

        
    def hamiltonian(self) -> np.ndarray:
        """
        Returns the Hamiltonian of the system.

        Returns
        -------
        np.ndarray
            The Hamiltonian of the system.
        """
        n_op = self.n_operator()
        charge_term = 4 * self.Ec * n_op @ n_op
        phase_op = self.phase_operator()
        ext_phase_op = self.phase * np.eye(self.dimension)
                    
        if self.flux_grouping == 'EL':
            inductive_term = 0.5 * self.El * (phase_op + ext_phase_op) @ (phase_op + ext_phase_op)
            josephson_term = - self.Ej * cosm(phase_op)
        elif self.flux_grouping == 'EJ':
            inductive_term = 0.5 * self.El * phase_op @ phase_op
            josephson_term = - self.Ej * cosm((phase_op - ext_phase_op))
            
        return charge_term + inductive_term + josephson_term
    
    def d_hamiltonian_d_EL(self) -> np.ndarray:
        
        if self.flux_grouping == 'EL':
            phase_op = self.phase_operator() + self.phase * np.eye(self.dimension)
        elif self.flux_grouping == 'EJ':
            phase_op = self.phase_operator()

        return 1/2 * np.dot(phase_op, phase_op)
    
    def d_hamiltonian_d_ng(self) -> np.ndarray:
        """
        Returns the derivative of the Hamiltonian with respect to the number of charge offset.
        
        Returns
        -------
        np.ndarray
            The derivative of the Hamiltonian with respect to the number of charge offset.
        
        """
        return - 8 * self.Ec * self.n_operator()
    
    def d2_hamiltonian_d_ng2(self) -> np.ndarray:
        """
        Returns the second derivative of the Hamiltonian with respect to the number of charge offset.
        
        Returns
        -------
        np.ndarray
            The second derivative of the Hamiltonian with respect to the number of charge offset.
        
        """
        return 8 * self.Ec * np.eye(self.dimension)
    
    def d_hamiltonian_d_phase(self) -> np.ndarray:
        """
        Returns the derivative of the Hamiltonian with respect to the external magnetic phase.

        Returns
        -------
        np.ndarray
            The derivative of the Hamiltonian with respect to the external magnetic phase.
        """
        if self.flux_grouping == 'EL':
            return self.El * (self.phase_operator() + self.phase * np.eye(self.dimension))
        elif self.flux_grouping == 'EJ':
            return  - self.Ej * sinm(self.phase_operator() - self.phase * np.eye(self.dimension))
        
    def d2_hamiltonian_d_phase2(self) -> np.ndarray:
        """
        Returns the second derivative of the Hamiltonian with respect to the external magnetic phase.

        Returns
        -------
        np.ndarray
            The second derivative of the Hamiltonian with respect to the external magnetic phase.
        """
        if self.flux_grouping == 'EL':
            return self.El * np.eye(self.dimension)
        elif self.flux_grouping == 'EJ':
            return self.Ej * cosm(self.phase_operator() - self.phase * np.eye(self.dimension))
        
    def d_hamiltonian_d_EJ(self) -> np.ndarray:
        """
        Returns the derivative of the Hamiltonian with respect to the Josephson energy Ej.

        Returns
        -------
        np.ndarray
            The derivative of the Hamiltonian with respect to Ej.
        """
        phase_op = self.phase_operator()
        if self.flux_grouping == 'EJ':
            phase_op -= self.phase * np.eye(self.dimension)
        return - cosm(phase_op)
    
    def wavefunction(
        self, 
        which: int = 0,
        phi_grid: np.ndarray = None,
        esys: Tuple[np.ndarray, np.ndarray] = None,
        basis: str = 'phase',
        ) -> Dict[str, Any]:
        """
        Returns a wave function in the phi basis.

        Parameters
        ----------
        which : int, optional
            Index of desired wave function (default is 0).
        phi_grid : np.ndarray, optional
            Custom grid for phi; if None, a default grid is used.
        basis : str, optional
            Basis in which to return the wavefunction ('phase' or 'charge') (default is 'phase').
        rotate : bool, optional
            Whether to rotate the basis (default is False).
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
        evecs = evecs.T
        
        if basis == 'phase':
            l_osc = self.phase_zpf
        elif basis == 'charge':
            l_osc = self.n_zpf     
                                
        if phi_grid is None:
            phi_grid = np.linspace(-5 * np.pi, 5 * np.pi, 151)

        phi_basis_labels = phi_grid
        wavefunc_osc_basis_amplitudes = evecs[which, :]
        phi_wavefunc_amplitudes = np.zeros(len(phi_grid), dtype=np.complex128)
        
        for n in range(dim):
            phi_wavefunc_amplitudes += wavefunc_osc_basis_amplitudes[n] * self.harm_osc_wavefunction(n, phi_basis_labels, l_osc)

        if basis == 'charge':
            phi_wavefunc_amplitudes /= np.sqrt(self.n_zpf)
        elif basis == 'phase':
            phi_wavefunc_amplitudes /= np.sqrt(self.phase_zpf)
            
        return {
            "basis_labels": phi_basis_labels,
            "amplitudes": phi_wavefunc_amplitudes,
            "energy": evals[which]
        }
        
                
    def potential(self, phi: Union[float, np.ndarray]) -> np.ndarray:
        """
        Calculates the potential energy for given values of phi.

        Parameters
        ----------
        phi : Union[float, np.ndarray]
            The phase values at which to calculate the potential.

        Returns
        -------
        np.ndarray
            The potential energy values.
        """
        phi_array = np.atleast_1d(phi)
        
        if self.flux_grouping == 'EL':
            inductive_term = self.El / 2 * (phi_array + self.phase) ** 2
            josephson_term = - self.Ej * np.cos(phi_array)
        elif self.flux_grouping == 'EJ':
            inductive_term = self.El / 2 * phi_array ** 2
            josephson_term = - self.Ej * np.cos(phi_array - self.phase)
        return inductive_term + josephson_term
    
    def tphi_1_over_f_flux(
        self, 
        A_noise: float = 1e-6,
        esys: Tuple[np.ndarray, np.ndarray] = None, 
        get_rate: bool = False, 
        **kwargs
        ) -> float:
        return self.tphi_1_over_f(A_noise, ['d_hamiltonian_d_phase', 'd2_hamiltonian_d_phase'], esys=esys, get_rate=get_rate, **kwargs)

    def plot_wavefunction(
        self, 
        which: Union[int, Iterable[int]] = 0, 
        phi_grid: np.ndarray = None, 
        esys: Tuple[np.ndarray, np.ndarray] = None, 
        scaling: Optional[float] = 1,
        plot_potential: bool = False,
        basis: str = 'phase',
        mode: str = 'abs',
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
        scaling : float, optional
            Scaling factor for the wavefunction (default is 1).
        plot_potential : bool, optional
            Whether to plot the potential (default is False).
        basis: str, optional
            Basis in which to return the wavefunction ('phase' or 'charge') (default is 'phase').
        mode: str, optional
            Mode of the wavefunction ('abs', 'real', or 'imag') (default is 'abs').
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
            
        if phi_grid is None:
            phi_grid = np.linspace(-5 * np.pi, 5 * np.pi, 151)
            
        fig_ax = kwargs.get("fig_ax")
        if fig_ax is None:
            fig, ax = plt.subplots()
            fig.suptitle(self._generate_suptitle())
        else:
            fig, ax = fig_ax
        
        if plot_potential:
            potential = self.potential(phi=phi_grid)
            ax.plot(phi_grid, potential, color='black', label='Potential')

        for idx in which:
            wavefunc_data = self.wavefunction(which=idx, phi_grid=phi_grid, esys=esys, basis=basis)
            phi_basis_labels = wavefunc_data["basis_labels"]
            wavefunc_amplitudes = wavefunc_data["amplitudes"]
            wavefunc_energy = wavefunc_data["energy"]
            
            if mode == 'abs':
                y_values = np.abs(wavefunc_amplitudes)
            elif mode == 'real':
                y_values = wavefunc_amplitudes.real
            elif mode == 'imag':
                y_values = wavefunc_amplitudes.imag
            else:
                raise ValueError("Invalid mode; must be 'abs', 'real', or 'imag'.")

            ax.plot(
                phi_basis_labels,
                wavefunc_energy + scaling * y_values,
                label=rf"$\Psi_{idx}$"
                )
            
        if basis == 'phase':
            ax.set_xlabel(r"$2 \pi \Phi / \Phi_0$")
            ax.set_ylabel(r"$\psi(\varphi)$, Energy [GHz]")
        elif basis == 'charge':
            ax.set_xlabel(r"$n$")
            ax.set_ylabel(r"$\psi(n)$, Energy [GHz]")
        
        ax.legend()
        ax.grid(True)

        return fig, ax      